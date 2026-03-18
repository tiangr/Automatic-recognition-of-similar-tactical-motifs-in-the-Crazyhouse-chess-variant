"""
Crazyhouse Tactical Retrieval — Flask Backend
Run: python app.py  (from inside src/)
"""

from pathlib import Path
import json
import io
import chess
import chess.variant
import chess.pgn
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from rank_bm25 import BM25Okapi

# Encoders — try to import from src/, fall back to built-in implementations
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from encode import encode_static, encode_pockets
    from encodev2 import encode_dynamic_v2
except ImportError:
    # ── Fallback: inline implementations matching build_corpus.py logic ──────
    import chess as _chess

    def encode_static(board_fen: str) -> list[str]:
        """Piece-on-square tokens: e.g. wq@d1, bp@e5"""
        tokens = []
        try:
            board = _chess.Board(board_fen)
        except Exception:
            return tokens
        for sq in _chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                color = "w" if piece.color == _chess.WHITE else "b"
                pt    = piece.symbol().lower()
                sq_name = _chess.square_name(sq)
                tokens.append(f"{color}{pt}@{sq_name}")
        return tokens

    def encode_pockets(pockets_white: dict, pockets_black: dict) -> list[str]:
        """Pocket tokens: e.g. pocket:w:N pocket:b:P"""
        tokens = []
        for piece, count in (pockets_white or {}).items():
            for _ in range(int(count)):
                tokens.append(f"pocket:w:{piece}")
        for piece, count in (pockets_black or {}).items():
            for _ in range(int(count)):
                tokens.append(f"pocket:b:{piece}")
        return tokens

    try:
        from encodev2 import encode_dynamic_v2
    except ImportError:
        def encode_dynamic_v2(rec: dict) -> list[str]:
            return []

# src/ directory (where this file lives)
_SRC  = Path(__file__).resolve().parent
# project root (one level up from src/)
ROOT  = _SRC.parent

app = Flask(__name__, static_folder=str(_SRC / "static"), static_url_path="")
CORS(app)

# ── Paths ────────────────────────────────────────────────────────────────────
CORPUS = ROOT / "data" / "derived" / "corpus_tactical.jsonl"
AUG    = ROOT / "data" / "derived" / "tactics_1k_aug.jsonl"

# ── Load data once at startup ─────────────────────────────────────────────────
def load_aug_index():
    idx = {}
    with AUG.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            _id = r.get("event_id") or (r.get("site", "?") + "_" + str(r.get("ply", "?")))
            idx[_id] = r
    return idx

def load_docs():
    docs, texts = [], []
    with CORPUS.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            docs.append(rec)
            texts.append(rec["text_all"].split())
    return docs, texts

print("Loading corpus and index …")
aug_index = load_aug_index()
docs, tokenized = load_docs()
bm25 = BM25Okapi(tokenized)
# Pre-build a static-field BM25 index (used by search_by_fen for custom FEN queries)
static_tokenized = [d.get("text_static", "").split() for d in docs]
bm25_static = BM25Okapi(static_tokenized)
doc_id_to_idx = {d["id"]: i for i, d in enumerate(docs)}
print(f"  {len(docs)} corpus docs, {len(aug_index)} aug records")

# ── Helpers ───────────────────────────────────────────────────────────────────
def base_game_id(doc_id: str) -> str:
    return doc_id.rsplit("_", 1)[0]

def pocket_str(p):
    if not p:
        return "—"
    return " ".join(f"{k}{v}" for k, v in sorted(p.items()))

def lichess_url(site: str):
    if not site or site == "?":
        return None
    game_id = site.rstrip("/").split("/")[-1]
    if len(game_id) >= 8:
        return f"https://lichess.org/{game_id}"
    return None

def build_record(doc_id: str, score=None):
    aug = aug_index.get(doc_id)
    if aug is None:
        return None
    pv = aug.get("pv_before") or aug.get("pv_prev") or []
    bm = aug.get("bestmove_before") or aug.get("best_prev") or "?"
    site = aug.get("site", "?")
    return {
        "id":            doc_id,
        "site":          site,
        "ply":           aug.get("ply"),
        "mate_before":   aug.get("mate_before"),
        "cp_before":     aug.get("cp_before"),
        "delta":         aug.get("delta"),
        "played_move":   aug.get("played_move"),
        "best_move":     bm,
        "pv":            pv[:10],
        "pv_has_drop":   any("@" in m for m in pv),
        "pockets_white": pocket_str(aug.get("pockets_white")),
        "pockets_black": pocket_str(aug.get("pockets_black")),
        "board_fen":     aug.get("board_fen"),
        "fen":           aug.get("fen"),
        "turn":          aug.get("turn", "white"),
        "lichess_url":   lichess_url(site),
        "score":         round(score, 3) if score is not None else None,
    }

def build_pgn_for_ids(ids):
    out = io.StringIO()
    for doc_id in ids:
        aug = aug_index.get(doc_id)
        if not aug:
            continue
        board = chess.variant.CrazyhouseBoard()
        prefix = aug.get("uci_moves_prefix") or aug.get("uci_moves") or []
        ply = aug.get("ply", 0)
        if not isinstance(prefix, list):
            prefix = []
        if aug.get("uci_moves") and not aug.get("uci_moves_prefix"):
            prefix = aug["uci_moves"][:max(0, ply - 1)]

        game = chess.pgn.Game()
        game.headers["Event"]   = doc_id
        game.headers["Site"]    = aug.get("site", "?")
        game.headers["Variant"] = "Crazyhouse"

        node = game
        for u in prefix:
            try:
                mv = board.parse_uci(u)
                node = node.add_variation(mv)
                board.push(mv)
            except Exception:
                break

        pv    = aug.get("pv_before") or []
        bm    = aug.get("bestmove_before") or ""
        mate  = aug.get("mate_before")
        delta = aug.get("delta")
        node.comment = (
            f"delta={delta} mate_before={mate} "
            f"best={bm} pv={' '.join(pv[:10])}"
        )
        print(game, file=out, end="\n\n")
    return out.getvalue()

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/api/queries")
def get_queries():
    rows = []
    for doc_id in doc_id_to_idx:
        aug = aug_index.get(doc_id)
        if aug:
            rows.append({
                "id":          doc_id,
                "ply":         aug.get("ply"),
                "mate_before": aug.get("mate_before"),
                "delta":       aug.get("delta"),
            })
    rows.sort(key=lambda r: r["delta"] or 0, reverse=True)
    return jsonify(rows)

@app.route("/api/search_by_fen", methods=["POST"])
def search_by_fen():
    """
    Strategy:
    1. Extract board_fen from the posted FEN (strip pocket rank if crazyhouse).
    2. Find the corpus doc with a matching board_fen (exact match).
    3. If found, use THAT doc's pre-built tokens (with full PV/dynamic) for BM25.
    4. If no exact match, fall back to encoding on-the-fly with static+pocket tokens.
    Either way, return the matched doc as query_rec and the BM25 hits.
    """
    body  = request.get_json()
    fen   = body.get("fen", "").strip()
    field = body.get("field", "text_all")
    topk  = int(body.get("topk", 10))

    if not fen:
        return jsonify({"error": "no fen"}), 400

    # Parse FEN — strip pocket rank (9th segment) if present
    raw_parts     = fen.split(" ")
    ranks         = raw_parts[0].split("/")
    board_fen_only = "/".join(ranks[:8])

    # ── Step 1: find exact board_fen match in corpus ──────────────────────────
    best_id = None
    for doc in docs:
        if doc.get("board_fen", "") == board_fen_only and doc["id"] in aug_index:
            best_id = doc["id"]
            break

    if best_id:
        # Use the corpus doc's own tokens — full PV/dynamic signal included
        qi    = doc_id_to_idx[best_id]
        query = docs[qi]
        query_text = query.get(field, query["text_all"]).split()
        scores = bm25.get_scores(query_text)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        query_rec = build_record(best_id)

        hits = []
        for idx in ranked:
            cand_id = docs[idx]["id"]
            if cand_id == best_id:
                continue
            if base_game_id(cand_id) == base_game_id(best_id):
                continue
            rec = build_record(cand_id, scores[idx])
            if rec:
                hits.append(rec)
            if len(hits) >= topk:
                break

        return jsonify({"query": query_rec, "hits": hits, "field": field})

    # ── Step 2: no exact match — encode on the fly ────────────────────────────
    pocket_str_raw = ranks[8] if len(ranks) >= 9 else ""
    pockets_w: dict = {}
    pockets_b: dict = {}
    for ch in pocket_str_raw:
        if ch.isupper() and ch in "QRBNP":
            pockets_w[ch] = pockets_w.get(ch, 0) + 1
        elif ch.islower() and ch.upper() in "QRBNP":
            pockets_b[ch.upper()] = pockets_b.get(ch.upper(), 0) + 1

    try:
        static_tokens = encode_static(board_fen_only)
        pocket_tokens = encode_pockets(pockets_w, pockets_b)
    except Exception as e:
        return jsonify({"error": f"encoding failed: {e}"}), 500

    # Without PV we only have static+pocket signal
    query_text = static_tokens + pocket_tokens * 2
    scores = bm25_static.get_scores(query_text)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    # Use top hit as the "query representative"
    best_id = next((docs[i]["id"] for i in ranked if docs[i]["id"] in aug_index), None)
    query_rec = build_record(best_id) if best_id else {
        "id": "custom_fen", "site": "?", "ply": None,
        "mate_before": None, "cp_before": None, "delta": None,
        "played_move": None, "best_move": None, "pv": [], "pv_has_drop": False,
        "pockets_white": pocket_str(pockets_w), "pockets_black": pocket_str(pockets_b),
        "board_fen": board_fen_only, "fen": fen,
        "turn": "white" if (len(raw_parts) > 1 and raw_parts[1] == "w") else "black",
        "lichess_url": None, "score": None,
    }

    hits = []
    seen = {base_game_id(best_id)} if best_id else set()
    for idx in ranked:
        cand_id = docs[idx]["id"]
        if cand_id == best_id:
            continue
        g = base_game_id(cand_id)
        if g in seen:
            continue
        rec = build_record(cand_id, scores[idx])
        if rec:
            hits.append(rec)
            seen.add(g)
        if len(hits) >= topk:
            break

    return jsonify({"query": query_rec, "hits": hits, "field": field})


@app.route("/api/retrieve/<path:doc_id>")
def retrieve(doc_id: str):
    field = request.args.get("field", "text_all")
    topk  = int(request.args.get("topk", 10))

    if doc_id not in doc_id_to_idx:
        return jsonify({"error": "doc_id not in corpus"}), 404

    qi    = doc_id_to_idx[doc_id]
    query = docs[qi]
    text  = query.get(field, query["text_all"])

    scores = bm25.get_scores(text.split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    q_base    = base_game_id(doc_id)
    query_rec = build_record(doc_id, score=None)

    hits = []
    for idx in ranked:
        cand_id = docs[idx]["id"]
        if cand_id == doc_id:
            continue
        if base_game_id(cand_id) == q_base:
            continue
        rec = build_record(cand_id, scores[idx])
        if rec:
            hits.append(rec)
        if len(hits) >= topk:
            break

    return jsonify({"query": query_rec, "hits": hits, "field": field})

@app.route("/api/export", methods=["POST"])
def export_pgn():
    body = request.get_json()
    ids  = body.get("ids", [])
    if not ids:
        return jsonify({"error": "no ids"}), 400

    pgn_text = build_pgn_for_ids(ids)
    buf = io.BytesIO(pgn_text.encode("utf-8"))
    buf.seek(0)
    return send_file(
        buf,
        mimetype="application/x-chess-pgn",
        as_attachment=True,
        download_name="crazyhouse_export.pgn",
    )

if __name__ == "__main__":
    app.run(debug=True, port=5000)