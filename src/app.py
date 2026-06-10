"""
app.py — Crazyhouse Tactical Retrieval Web App
Lucene BM25 backend + Fairy-Stockfish re-ranking.

Start order:
  1. java -Xmx2g -cp ".;../lucene/*" CrazyhouseLuceneServer
  2. python app.py
"""

from pathlib import Path
from collections import Counter
import json
import io
import re
import time
import threading
import urllib.parse
import urllib.request

import numpy as np
from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS

from encode   import encode_static, encode_pockets
from encodev2 import encode_dynamic_v2, encode_corpus_fields

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
ASSETS_DIR = BASE_DIR.parent / "assets"
LUCENE_URL = "http://localhost:8983"
CACHE_DIR  = BASE_DIR.parent / "data" / "models"

# Path to Fairy-Stockfish binary — adjust if yours is elsewhere
ENGINE_PATH = str(BASE_DIR.parent / "engines" / "fairy-stockfish-largeboard_x86-64.exe")
if not Path(ENGINE_PATH).exists():
    # Try common Windows name
    _alt = BASE_DIR.parent / "engine" / "fairy-stockfish.exe"
    if _alt.exists():
        ENGINE_PATH = str(_alt)

# ---------------------------------------------------------------------------
# Lucene client
# ---------------------------------------------------------------------------

def _lucene_search(query_tokens: str, field: str = "text_all",
                   topk: int = 10, exclude_id: str = "") -> list[dict]:
    params = urllib.parse.urlencode({
        "q":          query_tokens,
        "field":      field,
        "topk":       topk,
        "exclude_id": exclude_id,
    })
    try:
        with urllib.request.urlopen(f"{LUCENE_URL}/search?{params}", timeout=10) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"Lucene search error: {e}")
        return []


def _lucene_doc(doc_id: str) -> dict | None:
    params = urllib.parse.urlencode({"id": doc_id})
    try:
        with urllib.request.urlopen(f"{LUCENE_URL}/doc?{params}", timeout=5) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"Lucene doc error: {e}")
        return None


def _lucene_status() -> dict:
    try:
        with urllib.request.urlopen(f"{LUCENE_URL}/status", timeout=3) as r:
            return json.loads(r.read().decode())
    except Exception:
        return {"status": "unavailable"}


_status = _lucene_status()
if _status.get("status") == "ok":
    print(f"Lucene server connected: {_status.get('docs', '?'):,} docs indexed")
else:
    print("WARNING: Lucene server not reachable at", LUCENE_URL)

# ---------------------------------------------------------------------------
# Fairy-Stockfish engine (singleton, thread-safe via lock)
# ---------------------------------------------------------------------------
_engine      = None
_engine_lock = threading.Lock()
_engine_ok   = False


def _get_engine():
    """Lazy-load the engine; returns None if unavailable."""
    global _engine, _engine_ok
    if _engine_ok:
        return _engine
    if _engine is not None:
        return None   # already failed

    engine_bin = Path(ENGINE_PATH)
    if not engine_bin.exists():
        print(f"Engine not found at {ENGINE_PATH} — re-rank search will be unavailable.")
        print("  Set ENGINE_PATH in app.py to your Fairy-Stockfish binary.")
        _engine = None
        return None

    try:
        from engine_wrapper import FairyStockfish
        _engine = FairyStockfish(str(engine_bin))
        _engine_ok = True
        print(f"Fairy-Stockfish loaded: {engine_bin}")
        return _engine
    except Exception as e:
        print(f"Engine load failed: {e}")
        _engine = None
        return None


def _engine_best_line(fen: str, movetime_ms: int = 1500) -> list[str]:
    """
    Run Fairy-Stockfish on a Crazyhouse FEN, return PV as UCI move list.
    Thread-safe. Returns [] on failure.
    """
    eng = _get_engine()
    if eng is None:
        return []

    with _engine_lock:
        try:
            eng.new_game()
            # Use position fen <fen> instead of startpos moves
            eng._send(f"position fen {fen}")
            eng._send(f"go movetime {movetime_ms}")

            last_pv = []
            while True:
                line = eng._readline(timeout=movetime_ms / 1000 + 5)
                if line.startswith("info ") and "pv" in line:
                    parts = line.split()
                    j = parts.index("pv")
                    last_pv = parts[j + 1:]
                if line.startswith("bestmove"):
                    break

            return last_pv[:10]   # cap at 10 moves
        except Exception as e:
            print(f"Engine analysis failed: {e}")
            return []


_get_engine()   # attempt load at startup

# ---------------------------------------------------------------------------
# Feature cache (lazy-loaded for re-ranking)
# ---------------------------------------------------------------------------
_feat_mm   = None
_feat_idx  = None
_feat_cols = None


def _load_cache() -> bool:
    global _feat_mm, _feat_idx, _feat_cols
    if _feat_mm is not None:
        return True

    candidates = [
        ("feat_matrix_checkmates5.npy", "feat_index_checkmates5.json", "feat_cols_checkmates5.json"),
        ("feat_matrix.npy",             "feat_index.json",              "feat_cols.json"),
    ]
    for npy_name, idx_name, cols_name in candidates:
        npy_path  = CACHE_DIR / npy_name
        idx_path  = CACHE_DIR / idx_name
        cols_path = CACHE_DIR / cols_name
        if npy_path.exists() and idx_path.exists():
            try:
                with open(idx_path) as f:
                    _feat_idx = json.load(f)
                if cols_path.exists():
                    with open(cols_path) as f:
                        _feat_cols = json.load(f)
                _feat_mm = np.load(str(npy_path), mmap_mode="r")
                print(f"Feature cache loaded: {_feat_mm.shape[0]:,} docs × {_feat_mm.shape[1]} features  ({npy_name})")
                return True
            except Exception as e:
                print(f"Feature cache load failed ({npy_name}): {e}")
                _feat_mm = _feat_idx = _feat_cols = None

    print("Feature cache not found — re-ranking will use token-based fallback.")
    return False


_load_cache()

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=str(STATIC_DIR))
CORS(app)


def _parse_pocket(p):
    if isinstance(p, dict): return p
    if isinstance(p, str):
        try: return json.loads(p)
        except: return {}
    return {}


def _parse_arr(v):
    if isinstance(v, list): return v
    if isinstance(v, str):
        try: return json.loads(v)
        except: return []
    return []


def _format_hit(h: dict, rank: int) -> dict:
    site = h.get("site") or ""
    ply  = h.get("ply") or 0
    lichess_url = ""
    if site and "lichess.org" in site:
        game_id = site.split("/")[-1]
        lichess_url = f"https://lichess.org/{game_id}#{ply}"

    pw = _parse_pocket(h.get("pockets_white", {}))
    pb = _parse_pocket(h.get("pockets_black", {}))
    solution_san = _parse_arr(h.get("solution_san", []))
    solution_uci = _parse_arr(h.get("solution_uci", []))
    mate_in      = h.get("mate_in") or h.get("mate_before")

    return {
        "id":               h.get("id", ""),
        "rank":             rank,
        "score":            round(float(h.get("score", 0)), 3),
        "site":             site,
        "ply":              ply,
        "lichess_url":      lichess_url,
        "board_fen":        h.get("board_fen", ""),
        "fen":              h.get("fen", ""),
        "pockets_white":    pw,
        "pockets_black":    pb,
        "turn":             h.get("turn", "white"),
        "mate_in":          mate_in,
        "mate_before":      mate_in,
        "solution_san":     solution_san,
        "solution_uci":     solution_uci,
        "engine_verified":  h.get("engine_verified", False),
        "text_dynamic":     h.get("text_dynamic", ""),
        "white":            h.get("white"),
        "black":            h.get("black"),
        "white_elo":        h.get("white_elo"),
        "black_elo":        h.get("black_elo"),
        "game_rating_avg":  h.get("game_rating_avg") or h.get("meta_avg_rating"),
        "time_control":     h.get("time_control"),
        "utc_date":         h.get("utc_date"),
        "result":           h.get("result"),
        "source_pgn":       h.get("source_pgn"),
        "event":            h.get("event"),
        "delta":            None,
        "bestmove":         solution_san[0] if solution_san else None,
        "pv":               solution_san,
        "played_move":      None,
    }


# ---------------------------------------------------------------------------
# Re-ranking helpers
# ---------------------------------------------------------------------------

def _levenshtein_sim(a: list, b: list) -> float:
    if not a and not b: return 1.0
    if not a or not b:  return 0.0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for ca in a:
        curr = [prev[0] + 1] + [0] * lb
        for j, cb in enumerate(b):
            curr[j + 1] = min(prev[j+1]+1, curr[j]+1, prev[j]+(0 if ca==cb else 1))
        prev = curr
    return 1.0 - prev[lb] / max(la, lb)


def _feature_jaccard_vec(qv: np.ndarray, cv: np.ndarray) -> float:
    """Jaccard on binary dynamic feature vectors."""
    q_bin = qv > 0
    c_bin = cv > 0
    inter = float(np.logical_and(q_bin, c_bin).sum())
    union = float(np.logical_or(q_bin, c_bin).sum())
    return inter / union if union > 0 else 0.0


def _get_dyn_mask():
    if _feat_cols is None:
        return None
    if not hasattr(_get_dyn_mask, "_mask"):
        _get_dyn_mask._mask = np.array([
            c.startswith(("DG_", "DS_", "dyn_", "sum_"))
            for c in _feat_cols
        ], dtype=bool)
    return _get_dyn_mask._mask


def _parse_dynamic_tokens(text_dynamic: str) -> set:
    if not text_dynamic: return set()
    return set(text_dynamic.split())


def _mating_piece_from_tokens(tokens: set) -> str:
    for tok in tokens:
        if tok.startswith("dyn:mating_piece:"):
            return tok.split(":", 2)[2].upper()
        if tok.startswith("sum:mating_piece:"):
            return tok.split(":", 2)[2].upper()
    return ""


def _drop_pieces_from_tokens(tokens: set) -> set:
    for tok in tokens:
        if tok.startswith("dyn:dropPieces:"):
            return set(tok.split("dyn:dropPieces:")[1].upper())
    return {tok.split("pv:dropPiece:")[1].upper()
            for tok in tokens if tok.startswith("pv:dropPiece:")}


def _first_action_from_tokens(tokens: set) -> str:
    for tok in tokens:
        if tok.startswith("dyn:first:"):
            return tok.split("dyn:first:")[1]
    return ""


def _pocket_multiset(pocket: dict) -> list:
    if not isinstance(pocket, dict): return []
    items = []
    for piece, cnt in pocket.items():
        items.extend([piece.upper()] * max(0, int(cnt or 0)))
    return sorted(items)


def _multiset_jaccard(a: list, b: list) -> float:
    ca, cb = Counter(a), Counter(b)
    keys = set(ca) | set(cb)
    if not keys: return 0.5
    inter = sum(min(ca[k], cb[k]) for k in keys)
    union = sum(max(ca[k], cb[k]) for k in keys)
    return inter / union if union > 0 else 0.0


def _token_jaccard(toks_a: set, toks_b: set) -> float:
    inter = len(toks_a & toks_b)
    union = len(toks_a | toks_b)
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Query encoding (static + engine-derived dynamic)
# ---------------------------------------------------------------------------

def _encode_query(board_fen: str, fen: str, pw: dict, pb: dict,
                  movetime_ms: int = 1500) -> dict:
    """
    Fully encode a query position:
      - static tokens from encode_static()
      - dynamic tokens from engine best line via encode_dynamic_v2()

    Returns a dict with:
      static_tokens, pocket_tokens, dynamic_tokens,
      text_dynamic, solution_uci, engine_used
    """
    static_tokens = encode_static(board_fen)
    pocket_tokens = encode_pockets(pw, pb)

    # Try to get the engine line
    pv = _engine_best_line(fen or board_fen, movetime_ms=movetime_ms)
    engine_used = bool(pv)

    # Build a minimal record for encode_dynamic_v2
    rec = {
        "fen":          fen or board_fen,
        "board_fen":    board_fen,
        "solution_uci": pv,
        "pockets_white": pw,
        "pockets_black": pb,
    }

    dynamic_tokens = encode_dynamic_v2(rec) if pv else []
    text_dynamic   = " ".join(dynamic_tokens)

    return {
        "static_tokens":  static_tokens,
        "pocket_tokens":  pocket_tokens,
        "dynamic_tokens": dynamic_tokens,
        "text_dynamic":   text_dynamic,
        "solution_uci":   pv,
        "engine_used":    engine_used,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/assets/<path:filename>")
def assets(filename):
    return send_from_directory(ASSETS_DIR, filename)


@app.route("/api/debug_hit")
def debug_hit():
    doc_id = request.args.get("id", "")
    if not doc_id:
        return jsonify({"error": "id required"}), 400
    doc = _lucene_doc(doc_id)
    if doc is None:
        return jsonify({"error": "not found"}), 404
    return jsonify({
        "id":            doc.get("id"),
        "turn":          doc.get("turn"),
        "pockets_white": doc.get("pockets_white"),
        "pockets_black": doc.get("pockets_black"),
        "mate_in":       doc.get("mate_in"),
        "solution_uci":  doc.get("solution_uci"),
        "text_dynamic_preview": (doc.get("text_dynamic") or "")[:500],
    })


@app.route("/api/queries")
def get_queries():
    return jsonify([])


@app.route("/api/retrieve/<path:doc_id>")
def retrieve(doc_id: str):
    field = request.args.get("field", "text_all")
    topk  = int(request.args.get("topk", 10))
    q_hit = _lucene_doc(doc_id)
    if q_hit is None:
        return jsonify({"error": "doc_id not found"}), 404
    query_tokens = q_hit.get("text_all", "") if field != "text_static" else q_hit.get("text_static", "")
    raw_hits = _lucene_search(query_tokens, field, topk, exclude_id=doc_id)
    hits = [_format_hit(h, i+1) for i, h in enumerate(raw_hits)]
    return jsonify({"query": _format_hit(q_hit, 0), "hits": hits})


@app.route("/api/search_by_fen", methods=["POST"])
def search_by_fen():
    """
    Standard BM25 search using static tokens only.
    Fast — no engine call.
    """
    data      = request.get_json()
    raw_fen   = (data.get("board_fen") or data.get("fen") or "").strip()
    board_fen = raw_fen.split(" ")[0] if raw_fen else ""
    if "[" in board_fen:
        board_fen = board_fen.split("[")[0]
    parts = board_fen.split("/")
    if len(parts) == 9:
        board_fen = "/".join(parts[:8])

    topk = int(data.get("topk", 10))
    if not board_fen:
        return jsonify({"error": "board_fen required"}), 400

    pw = data.get("pockets_white", {})
    pb = data.get("pockets_black", {})

    try:
        static_tokens = encode_static(board_fen)
        pocket_tokens = encode_pockets(pw, pb)
        query_tokens  = " ".join(static_tokens + pocket_tokens)
    except Exception as e:
        return jsonify({"error": f"encoding failed: {e}"}), 500

    raw_hits = _lucene_search(query_tokens, "text_static", topk)
    hits = [_format_hit(h, i+1) for i, h in enumerate(raw_hits)]

    return jsonify({
        "query": {
            "id":            "search_by_fen",
            "board_fen":     board_fen,
            "pockets_white": pw,
            "pockets_black": pb,
            "site":          "",
            "ply":           0,
            "lichess_url":   "",
        },
        "hits": hits,
    })


@app.route("/api/rerank_search", methods=["POST"])
def rerank_search():
    """
    Full re-rank pipeline:
      1. Run Fairy-Stockfish on the query FEN (~1-2 sec)
      2. Encode dynamic tokens from engine PV
      3. Fresh Lucene search on text_all using static + dynamic tokens
      4. Re-rank all candidates using query's own sub-scores
      5. Return top-K results

    Request body:
    {
        "fen":           "full Crazyhouse FEN with pocket notation",
        "board_fen":     "board-only FEN (no pockets)",
        "pockets_white": {"N":1, ...},
        "pockets_black": {"Q":1, ...},
        "weights":       { ... slider values ... },
        "candidate_k":   50,   # how many BM25 candidates to retrieve
        "return_k":      10,   # how many to return after re-ranking
        "movetime_ms":   1500  # engine think time
    }
    """
    data       = request.get_json()
    raw_fen    = (data.get("fen") or data.get("board_fen") or "").strip()
    board_fen  = raw_fen.split(" ")[0] if raw_fen else ""
    if "[" in board_fen:
        board_fen = board_fen.split("[")[0]
    parts = board_fen.split("/")
    if len(parts) == 9:
        board_fen = "/".join(parts[:8])

    full_fen     = raw_fen   # keep the full FEN with pockets for the engine
    pw           = data.get("pockets_white", {})
    pb           = data.get("pockets_black", {})
    w            = data.get("weights", {})
    candidate_k  = int(data.get("candidate_k", 50))
    return_k     = int(data.get("return_k", 10))
    movetime_ms  = int(data.get("movetime_ms", 1500))

    if not board_fen:
        return jsonify({"error": "fen or board_fen required"}), 400

    # ── Step 1 & 2: Engine + encode ─────────────────────────────────────
    q = _encode_query(board_fen, full_fen, pw, pb, movetime_ms=movetime_ms)

    # ── Step 3: Fresh Lucene search on text_all ──────────────────────────
    # Weight: dynamic×3 + pocket×2 + static×1  (matches corpus construction)
    query_tokens = " ".join(
        q["dynamic_tokens"] * 3 +
        q["pocket_tokens"]  * 2 +
        q["static_tokens"]
    )
    raw_hits = _lucene_search(query_tokens, "text_all", candidate_k)
    hits     = [_format_hit(h, i+1) for i, h in enumerate(raw_hits)]

    if not hits:
        return jsonify({
            "engine_used":    q["engine_used"],
            "solution_uci":   q["solution_uci"],
            "candidate_k":    candidate_k,
            "hits":           [],
        })

    # ── Step 4: Re-rank using query's own tokens as reference ───────────
    cache_ok = _load_cache()

    w_bm25     = float(w.get("bm25",        0.25))
    w_feat     = float(w.get("tactic",      0.30))
    w_move     = float(w.get("move",        0.25))
    w_matingpc = float(w.get("matingpiece", 0.10))
    w_droppcs  = float(w.get("droppcs",     0.00))
    w_pocket   = float(w.get("pocket_sim",  0.00))
    w_firstact = float(w.get("first_act",   0.05))
    w_pvlen    = float(w.get("pvlen",       0.05))
    w_rating   = float(w.get("rating",      0.00))
    total_w    = (w_bm25 + w_feat + w_move + w_matingpc + w_droppcs +
                  w_pocket + w_firstact + w_pvlen + w_rating) or 1.0

    # Query reference values (from engine line)
    ref_tokens  = _parse_dynamic_tokens(q["text_dynamic"])
    ref_uci     = q["solution_uci"]
    ref_pvlen   = len(ref_uci)
    ref_mp      = _mating_piece_from_tokens(ref_tokens)
    ref_drops   = _drop_pieces_from_tokens(ref_tokens)
    ref_first   = _first_action_from_tokens(ref_tokens)
    ref_turn    = "black" if " b " in full_fen else "white"
    ref_pocket  = _pocket_multiset(pw if ref_turn == "white" else pb)

    # Build query feature vector for cache-based Jaccard
    q_feat_vec  = None
    dyn_mask    = _get_dyn_mask()
    if cache_ok and dyn_mask is not None:
        # Encode query features from its dynamic tokens
        # We use the token set to build a binary vector matching the cache columns
        dyn_tok_set = {t for t in ref_tokens if t.startswith(("dyn:", "pv:", "sum:"))}
        # Map tokens to feature columns
        from extract_features_crazyhouse import extract_features
        q_feat_dict = extract_features({
            "text_static":  " ".join(q["static_tokens"] + q["pocket_tokens"]),
            "text_dynamic": q["text_dynamic"],
            "turn":         ref_turn,
            "pockets_white": pw,
            "pockets_black": pb,
        })
        q_feat_arr = np.array(
            [q_feat_dict.get(c, 0.0) for c in _feat_cols], dtype=np.float32
        )
        q_feat_vec = q_feat_arr[dyn_mask]

    # Normalise BM25 scores
    bm25_scores = [float(h.get("score", 0)) for h in hits]
    max_bm25    = max(bm25_scores) or 1.0
    bm25_norm   = [s / max_bm25 for s in bm25_scores]

    results = []
    for i, h in enumerate(hits):
        hid     = h.get("id", "")
        sol_uci = _parse_arr(h.get("solution_uci") or [])
        pvlen   = len(sol_uci)
        rating  = h.get("game_rating_avg") or 0
        h_turn  = h.get("turn") or "white"
        h_toks  = _parse_dynamic_tokens(h.get("text_dynamic", ""))

        # BM25
        s_bm25 = bm25_norm[i]

        # Feature Jaccard — query vector vs hit cache vector
        s_feat = 0.5  # default neutral
        if q_feat_vec is not None and _feat_idx is not None and hid in _feat_idx:
            ci = _feat_idx[hid]
            cv = _feat_mm[ci][dyn_mask].astype(np.float64)
            s_feat = _feature_jaccard_vec(q_feat_vec.astype(np.float64), cv)
        elif ref_tokens:
            # Token fallback
            ref_dyn = {t for t in ref_tokens if t.startswith(("dyn:", "pv:", "sum:"))}
            h_dyn   = {t for t in h_toks   if t.startswith(("dyn:", "pv:", "sum:"))}
            s_feat  = _token_jaccard(ref_dyn, h_dyn) if ref_dyn else 0.5

        # Move-edit similarity (query PV vs hit PV)
        s_move = _levenshtein_sim(ref_uci, sol_uci)

        # Mating piece
        h_mp = _mating_piece_from_tokens(h_toks)
        if ref_mp and h_mp:
            s_matingpc = 1.0 if ref_mp == h_mp else 0.0
        else:
            s_matingpc = 0.5

        # Drop pieces
        h_drops = _drop_pieces_from_tokens(h_toks)
        if ref_drops or h_drops:
            inter = len(ref_drops & h_drops)
            union = len(ref_drops | h_drops)
            s_droppcs = inter / union if union > 0 else 1.0
        else:
            s_droppcs = 0.5

        # Pocket similarity
        h_pocket = _pocket_multiset(
            h.get("pockets_white") if h_turn == "white" else h.get("pockets_black")
        )
        s_pocket = _multiset_jaccard(ref_pocket, h_pocket)

        # First action
        h_first = _first_action_from_tokens(h_toks)
        if ref_first and h_first:
            s_firstact = 1.0 if ref_first == h_first else 0.0
        else:
            s_firstact = 0.5

        # PV length
        diff_len = abs(ref_pvlen - pvlen)
        s_pvlen  = max(0.0, 1.0 - diff_len * 0.5)

        # Rating proximity
        if rating:
            # No reference rating (query is a position) — use corpus average as proxy
            s_rating = 0.5
        else:
            s_rating = 0.5

        combined = (
            w_bm25     * s_bm25     +
            w_feat     * s_feat     +
            w_move     * s_move     +
            w_matingpc * s_matingpc +
            w_droppcs  * s_droppcs  +
            w_pocket   * s_pocket   +
            w_firstact * s_firstact +
            w_pvlen    * s_pvlen    +
            w_rating   * s_rating
        ) / total_w

        results.append({
            **h,
            "bm25_rank": i + 1,
            "_combined": round(combined, 4),
            "_sub": {
                "bm25":        round(s_bm25,     4),
                "tactic":      round(s_feat,     4),
                "move":        round(s_move,     4),
                "matingpiece": round(s_matingpc, 4),
                "droppcs":     round(s_droppcs,  4),
                "pocket_sim":  round(s_pocket,   4),
                "first_act":   round(s_firstact, 4),
                "pvlen":       round(s_pvlen,    4),
                "rating":      round(s_rating,   4),
            },
        })

    results.sort(key=lambda x: -x["_combined"])

    return jsonify({
        "engine_used":  q["engine_used"],
        "solution_uci": q["solution_uci"],
        "solution_dyn_tokens": list(ref_tokens)[:30],  # preview for debugging
        "candidate_k":  candidate_k,
        "cache_used":   q_feat_vec is not None,
        "hits":         results[:return_k],
    })


@app.route("/api/rerank", methods=["POST"])
def rerank():
    """
    Re-rank an existing hit list against hit #1 as reference.
    Used by the re-rank tab when operating on the initial BM25 results.
    """
    data     = request.get_json()
    query_id = data.get("query_id", "")
    hits     = data.get("hits", [])
    w        = data.get("weights", {})

    w_bm25     = float(w.get("bm25",        0.25))
    w_feat     = float(w.get("tactic",      0.30))
    w_move     = float(w.get("move",        0.25))
    w_matingpc = float(w.get("matingpiece", 0.10))
    w_droppcs  = float(w.get("droppcs",     0.00))
    w_pocket   = float(w.get("pocket_sim",  0.00))
    w_firstact = float(w.get("first_act",   0.05))
    w_pvlen    = float(w.get("pvlen",       0.05))
    w_rating   = float(w.get("rating",      0.00))

    if not hits:
        return jsonify({"cache_used": False, "hits": []})

    cache_ok = _load_cache()

    bm25_scores = [float(h.get("score", 0)) for h in hits]
    max_bm25    = max(bm25_scores) or 1.0
    bm25_norm   = [s / max_bm25 for s in bm25_scores]

    ref        = hits[0]
    ref_uci    = _parse_arr(ref.get("solution_uci") or [])
    ref_pvlen  = len(ref_uci)
    ref_rating = ref.get("game_rating_avg") or 0
    ref_turn   = ref.get("turn") or "white"
    ref_tokens = _parse_dynamic_tokens(ref.get("text_dynamic", ""))
    ref_mp     = _mating_piece_from_tokens(ref_tokens)
    ref_drops  = _drop_pieces_from_tokens(ref_tokens)
    ref_first  = _first_action_from_tokens(ref_tokens)
    ref_pocket = _pocket_multiset(
        ref.get("pockets_white") if ref_turn == "white" else ref.get("pockets_black")
    )

    total_w = (w_bm25 + w_feat + w_move + w_matingpc + w_droppcs +
               w_pocket + w_firstact + w_pvlen + w_rating) or 1.0
    results = []
    dyn_mask = _get_dyn_mask()

    for i, h in enumerate(hits):
        hid     = h.get("id", "")
        sol_uci = _parse_arr(h.get("solution_uci") or [])
        pvlen   = len(sol_uci)
        rating  = h.get("game_rating_avg") or 0
        h_turn  = h.get("turn") or "white"
        h_toks  = _parse_dynamic_tokens(h.get("text_dynamic", ""))

        if i == 0:
            results.append({
                "id": hid, "bm25_rank": 1, "combined": 1.0,
                "is_reference": True,
                "sub": {k: 1.0 for k in [
                    "bm25","tactic","move","matingpiece",
                    "droppcs","pocket_sim","first_act","pvlen","rating"
                ]},
            })
            continue

        s_bm25 = bm25_norm[i]

        # Feature Jaccard via cache
        s_feat = None
        if cache_ok and dyn_mask is not None:
            qi = _feat_idx.get(query_id) if query_id else None
            ci = _feat_idx.get(hid)
            if qi is not None and ci is not None:
                qv = _feat_mm[qi][dyn_mask].astype(np.float64)
                cv = _feat_mm[ci][dyn_mask].astype(np.float64)
                s_feat = _feature_jaccard_vec(qv, cv)
            elif ci is not None:
                # No query in cache — compare ref hit vs candidate
                ref_ci = _feat_idx.get(hits[0].get("id",""))
                if ref_ci is not None:
                    qv = _feat_mm[ref_ci][dyn_mask].astype(np.float64)
                    cv = _feat_mm[ci][dyn_mask].astype(np.float64)
                    s_feat = _feature_jaccard_vec(qv, cv)

        if s_feat is None:
            ref_dyn = {t for t in ref_tokens if t.startswith(("dyn:", "pv:", "sum:"))}
            h_dyn   = {t for t in h_toks   if t.startswith(("dyn:", "pv:", "sum:"))}
            s_feat  = _token_jaccard(ref_dyn, h_dyn) if ref_dyn else 0.5

        s_move = _levenshtein_sim(ref_uci, sol_uci)

        h_mp = _mating_piece_from_tokens(h_toks)
        s_matingpc = (1.0 if ref_mp == h_mp else 0.0) if (ref_mp and h_mp) else 0.5

        h_drops = _drop_pieces_from_tokens(h_toks)
        if ref_drops or h_drops:
            inter = len(ref_drops & h_drops)
            union = len(ref_drops | h_drops)
            s_droppcs = inter / union if union > 0 else 1.0
        else:
            s_droppcs = 0.5

        h_pocket  = _pocket_multiset(
            h.get("pockets_white") if h_turn == "white" else h.get("pockets_black")
        )
        s_pocket = _multiset_jaccard(ref_pocket, h_pocket)

        h_first = _first_action_from_tokens(h_toks)
        s_firstact = (1.0 if ref_first == h_first else 0.0) if (ref_first and h_first) else 0.5

        diff_len = abs(ref_pvlen - pvlen)
        s_pvlen  = max(0.0, 1.0 - diff_len * 0.5)

        if ref_rating and rating:
            s_rating = max(0.0, 1.0 - abs(ref_rating - rating) / 500.0)
        else:
            s_rating = 0.5

        combined = (
            w_bm25     * s_bm25     +
            w_feat     * s_feat     +
            w_move     * s_move     +
            w_matingpc * s_matingpc +
            w_droppcs  * s_droppcs  +
            w_pocket   * s_pocket   +
            w_firstact * s_firstact +
            w_pvlen    * s_pvlen    +
            w_rating   * s_rating
        ) / total_w

        results.append({
            "id": hid, "bm25_rank": i+1,
            "combined": round(combined, 4),
            "sub": {
                "bm25":        round(s_bm25,     4),
                "tactic":      round(s_feat,     4),
                "move":        round(s_move,     4),
                "matingpiece": round(s_matingpc, 4),
                "droppcs":     round(s_droppcs,  4),
                "pocket_sim":  round(s_pocket,   4),
                "first_act":   round(s_firstact, 4),
                "pvlen":       round(s_pvlen,    4),
                "rating":      round(s_rating,   4),
            },
        })

    results.sort(key=lambda x: -x["combined"])
    return jsonify({"cache_used": cache_ok, "hits": results})


@app.route("/api/export", methods=["POST"])
def export_labels():
    data = request.get_json()

    # Support both payload formats:
    # 1. {all_hit_ids, similar_ids, query_fen, query_id}  ← new format (export all with 0/1 labels)
    # 2. {ids}                                             ← legacy format (all selected = label 1)
    if "ids" in data:
        ids         = data["ids"]
        query_id    = ids[0] if ids else ""
        all_hit_ids = ids[1:]
        similar_ids = set(all_hit_ids)   # legacy: everything passed in = similar
        query_fen   = data.get("query_fen", "")
    else:
        query_id    = data.get("query_id", "")
        query_fen   = data.get("query_fen", "")
        similar_ids = set(data.get("similar_ids", []))
        all_hit_ids = data.get("all_hit_ids", [])

    if not all_hit_ids:
        return jsonify({"error": "no hits provided"}), 400

    q_doc = _lucene_doc(query_id) if query_id and query_id != "search_by_fen" and query_id != "rerank_search" else None
    out   = io.StringIO()

    for rank, hit_id in enumerate(all_hit_ids, start=1):
        h_doc = _lucene_doc(hit_id) or {}
        record = {
            "query_id":               query_id,
            "candidate_id":           hit_id,
            "label":                  1 if hit_id in similar_ids else 0,
            "bm25_rank":              rank,
            # query_fen: prefer the live FEN from the frontend over the stored doc FEN
            "query_fen":              query_fen or (q_doc.get("board_fen") if q_doc else ""),
            "candidate_fen":          h_doc.get("board_fen", ""),
            "query_mate":             q_doc.get("mate_in") or q_doc.get("mate_before") if q_doc else None,
            "candidate_mate":         h_doc.get("mate_in") or h_doc.get("mate_before"),
            "query_turn":             q_doc.get("turn", "") if q_doc else "",
            "candidate_turn":         h_doc.get("turn", ""),
            "query_solution_san":     _parse_arr(q_doc.get("solution_san")) if q_doc else [],
            "candidate_solution_san": _parse_arr(h_doc.get("solution_san")),
            "query_solution_uci":     _parse_arr(q_doc.get("solution_uci")) if q_doc else [],
            "candidate_solution_uci": _parse_arr(h_doc.get("solution_uci")),
            "query_text_static":      q_doc.get("text_static", "") if q_doc else "",
            "candidate_text_static":  h_doc.get("text_static", ""),
            "query_text_dynamic":     q_doc.get("text_dynamic", "") if q_doc else "",
            "candidate_text_dynamic": h_doc.get("text_dynamic", ""),
            "query_site":             q_doc.get("site") if q_doc else "",
            "candidate_site":         h_doc.get("site", ""),
            "query_rating_avg":       q_doc.get("game_rating_avg") if q_doc else None,
            "candidate_rating_avg":   h_doc.get("game_rating_avg"),
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")

    out_bytes = out.getvalue().encode("utf-8")
    return send_file(
        io.BytesIO(out_bytes),
        mimetype="application/x-ndjson",
        as_attachment=True,
        download_name=f"ch_export_{(query_id or 'export')[:20]}_{int(time.time())}.jsonl",
    )


if __name__ == "__main__":
    print("Starting Crazyhouse Tactical Retrieval server...")
    app.run(host="0.0.0.0", port=5000, debug=False)