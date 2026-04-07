"""
app.py — Crazyhouse Tactical Retrieval Web App
Updated to use corpus_mates.jsonl + mates_meta.jsonl (full-scale mate database)
"""

from pathlib import Path
import json
import io
import chess
import chess.variant
import chess.pgn
from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS
from rank_bm25 import BM25Okapi

from encode import encode_static, encode_pockets

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR.parent / "data" / "derived"
STATIC_DIR  = BASE_DIR / "static"

CORPUS_PATH = DATA_DIR / "corpus_mates.jsonl"
META_PATH   = DATA_DIR / "mates_meta.jsonl"

# Set to None to load everything (for production on Pi)
# Set to e.g. 200_000 for local testing
MAX_DOCS = 200_000

# ---------------------------------------------------------------------------
# Load data at startup
# ---------------------------------------------------------------------------
print(f"Loading corpus (limit={MAX_DOCS if MAX_DOCS else 'all'})...")
docs = []
tokenized_all    = []
tokenized_static = []

with CORPUS_PATH.open("r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        if MAX_DOCS and len(docs) >= MAX_DOCS:
            break
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        docs.append(rec)
        tokenized_all.append(rec.get("text_all", "").split())
        tokenized_static.append(rec.get("text_static", "").split())
        if (i + 1) % 50_000 == 0:
            print(f"  loaded {len(docs):,} corpus docs...")

print(f"Corpus loaded: {len(docs):,} docs")

print("Building BM25 indexes...")
bm25        = BM25Okapi(tokenized_all)
bm25_static = BM25Okapi(tokenized_static)
print("BM25 indexes ready.")

# Build corpus lookup by id and by board_fen
doc_by_id  = {d["id"]: i for i, d in enumerate(docs)}
doc_by_fen = {}
for i, d in enumerate(docs):
    fen = d.get("board_fen", "")
    if fen and fen not in doc_by_fen:
        doc_by_fen[fen] = i

print("Loading metadata...")
meta_index = {}
with META_PATH.open("r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        eid = rec.get("event_id") or rec.get("id")
        if eid:
            meta_index[eid] = rec

print(f"Metadata loaded: {len(meta_index):,} records")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=str(STATIC_DIR))
CORS(app)


def base_game_id(doc_id: str) -> str:
    return doc_id.rsplit("_", 1)[0]


def get_meta(doc_id: str) -> dict:
    """Return metadata for a doc_id, falling back to corpus doc itself."""
    if doc_id in meta_index:
        return meta_index[doc_id]
    # fallback: build minimal meta from corpus doc
    idx = doc_by_id.get(doc_id)
    if idx is not None:
        d = docs[idx]
        return {
            "event_id":      d.get("id"),
            "site":          d.get("site", ""),
            "ply":           d.get("ply", 0),
            "board_fen":     d.get("board_fen", ""),
            "pockets_white": d.get("pockets_white", {}),
            "pockets_black": d.get("pockets_black", {}),
        }
    return {}


def format_hit(doc_id: str, score: float, rank: int) -> dict:
    """Format a retrieval hit for the frontend."""
    meta = get_meta(doc_id)
    idx  = doc_by_id.get(doc_id)
    d    = docs[idx] if idx is not None else {}

    site = meta.get("site", "") or d.get("site", "")
    ply  = meta.get("ply", 0)  or d.get("ply", 0)

    # Build Lichess URL with move number
    lichess_url = ""
    if site and "lichess.org" in site:
        game_id = site.split("/")[-1]
        move_num = (int(ply) + 1) // 2 if ply else 0
        color = "black" if int(ply) % 2 == 0 else "white"
        lichess_url = f"https://lichess.org/{game_id}#{ply}"

    return {
        "id":            doc_id,
        "rank":          rank,
        "score":         round(score, 3),
        "site":          site,
        "ply":           ply,
        "lichess_url":   lichess_url,
        "board_fen":     meta.get("board_fen") or d.get("board_fen", ""),
        "pockets_white": meta.get("pockets_white") or d.get("pockets_white", {}),
        "pockets_black": meta.get("pockets_black") or d.get("pockets_black", {}),
        # Fields available only if full aug was loaded
        "mate_before":   meta.get("mate_before"),
        "delta":         meta.get("delta"),
        "bestmove":      meta.get("bestmove_before"),
        "pv":            meta.get("pv_before", []),
        "played_move":   meta.get("played_move"),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/queries")
def get_queries():
    """Return a sample of corpus positions for the query list."""
    limit = int(request.args.get("limit", 200))
    results = []
    for i, d in enumerate(docs[:limit * 10]):
        meta = get_meta(d["id"])
        delta = meta.get("delta")
        results.append({
            "id":    d["id"],
            "site":  d.get("site", ""),
            "ply":   d.get("ply", 0),
            "delta": delta,
        })
    # Sort by delta descending if available
    results.sort(key=lambda x: x["delta"] or 0, reverse=True)
    return jsonify(results[:limit])


@app.route("/api/retrieve/<path:doc_id>")
def retrieve(doc_id: str):
    """BM25 retrieval for a given corpus doc ID."""
    field = request.args.get("field", "text_all")
    topk  = int(request.args.get("topk", 10))

    if doc_id not in doc_by_id:
        return jsonify({"error": "doc_id not found"}), 404

    qi    = doc_by_id[doc_id]
    query = docs[qi]
    q_base = base_game_id(doc_id)

    if field == "text_static":
        scores = bm25_static.get_scores(query.get("text_static", "").split())
    else:
        scores = bm25.get_scores(query.get("text_all", "").split())

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    hits = []
    rank = 1
    for idx in ranked:
        cand = docs[idx]
        cid  = cand["id"]
        if cid == doc_id:
            continue
        if base_game_id(cid) == q_base:
            continue
        hits.append(format_hit(cid, scores[idx], rank))
        rank += 1
        if rank > topk:
            break

    query_info = format_hit(doc_id, 1.0, 0)
    return jsonify({"query": query_info, "hits": hits})


@app.route("/api/search_by_fen", methods=["POST"])
def search_by_fen():
    """Find corpus doc matching a FEN and return BM25 hits."""
    data      = request.get_json()
    # Accept both "fen" (sent by frontend) and "board_fen" (internal)
    raw_fen   = (data.get("board_fen") or data.get("fen") or "").strip()
    # Extract board placement only (strip turn, castling, pockets etc.)
    board_fen = raw_fen.split(" ")[0] if raw_fen else ""
    # Strip bracket pocket notation e.g. r3k2r/.../R2Q1R2[BBNNNb]
    if "[" in board_fen:
        board_fen = board_fen.split("[")[0]
    # Strip 9-rank crazyhouse FEN (pocket as extra rank) — keep only first 8 ranks
    parts = board_fen.split("/")
    if len(parts) == 9:
        board_fen = "/".join(parts[:8])
    field     = data.get("field", "text_all")
    topk      = int(data.get("topk", 10))

    if not board_fen:
        return jsonify({"error": "board_fen required"}), 400

    # Try exact match first
    if board_fen in doc_by_fen:
        qi     = doc_by_fen[board_fen]
        doc_id = docs[qi]["id"]
        return retrieve(doc_id)

    # Fallback: encode on the fly using static features only
    pw = data.get("pockets_white", {})
    pb = data.get("pockets_black", {})
    try:
        static_tokens  = encode_static(board_fen)
        pocket_tokens  = encode_pockets(pw, pb)
        query_tokens   = static_tokens + pocket_tokens
    except Exception as e:
        return jsonify({"error": f"encoding failed: {e}"}), 500

    scores = bm25_static.get_scores(query_tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    hits = []
    rank = 1
    for idx in ranked:
        cid = docs[idx]["id"]
        hits.append(format_hit(cid, scores[idx], rank))
        rank += 1
        if rank > topk:
            break

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
        "hits": hits
    })


@app.route("/api/export", methods=["POST"])
def export_pgn():
    """Export selected positions as a PGN file."""
    data       = request.get_json()
    query_id   = data.get("query_id", "")
    hit_ids    = data.get("hit_ids", [])
    all_ids    = ([query_id] if query_id else []) + hit_ids

    pgn_out = io.StringIO()

    for pos_id in all_ids:
        idx  = doc_by_id.get(pos_id)
        if idx is None:
            continue
        d    = docs[idx]
        meta = get_meta(pos_id)

        board = chess.variant.CrazyhouseBoard()
        game  = chess.pgn.Game()
        game.headers["Event"]   = pos_id
        game.headers["Site"]    = meta.get("site") or d.get("site", "?")
        game.headers["Variant"] = "Crazyhouse"

        prefix = meta.get("uci_moves_prefix") or []
        node   = game
        try:
            for u in prefix:
                mv   = board.parse_uci(u)
                node = node.add_variation(mv)
                board.push(mv)
        except Exception:
            pass

        # Add comment with available metadata
        parts = []
        if meta.get("mate_before"):
            parts.append(f"mate={meta['mate_before']}")
        if meta.get("delta"):
            parts.append(f"delta={meta['delta']:.0f}")
        if meta.get("bestmove_before"):
            parts.append(f"best={meta['bestmove_before']}")
        if meta.get("pv_before"):
            parts.append(f"pv={' '.join(meta['pv_before'][:8])}")
        if parts:
            node.comment = " ".join(parts)

        pgn_out.write(str(game) + "\n\n")

    pgn_bytes = pgn_out.getvalue().encode("utf-8")
    return send_file(
        io.BytesIO(pgn_bytes),
        mimetype="application/x-chess-pgn",
        as_attachment=True,
        download_name="crazyhouse_positions.pgn"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting Crazyhouse Tactical Retrieval server...")
    app.run(host="0.0.0.0", port=5000, debug=False)