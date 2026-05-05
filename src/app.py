"""
app.py — Crazyhouse Tactical Retrieval Web App
Lucene BM25 backend. No classifier, no re-ranking.

Start order:
  1. java -Xmx2g -cp ".;../lucene/*" CrazyhouseLuceneServer
  2. python app.py
"""

from pathlib import Path
import json
import io
import time
import urllib.parse
import urllib.request

from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS

from encode import encode_static, encode_pockets

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR   = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
LUCENE_URL = "http://localhost:8983"

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
    print("  Start it with: java -Xmx2g -cp '.;../lucene/*' CrazyhouseLuceneServer")

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=str(STATIC_DIR))
CORS(app)


def _base_game_id(doc_id: str) -> str:
    idx = doc_id.rsplit("_", 1)
    return idx[0] if len(idx) > 1 else doc_id


def _format_hit(h: dict, rank: int) -> dict:
    site = h.get("site") or ""
    ply  = h.get("ply") or 0
    lichess_url = ""
    if site and "lichess.org" in site:
        lichess_url = f"https://lichess.org/{site.split('/')[-1]}#{ply}"
    pw = h.get("pockets_white", {})
    pb = h.get("pockets_black", {})
    if isinstance(pw, str):
        try: pw = json.loads(pw)
        except: pw = {}
    if isinstance(pb, str):
        try: pb = json.loads(pb)
        except: pb = {}
    return {
        "id":            h.get("id", ""),
        "rank":          rank,
        "score":         round(float(h.get("score", 0)), 3),
        "site":          site,
        "ply":           ply,
        "lichess_url":   lichess_url,
        "board_fen":     h.get("board_fen", ""),
        "pockets_white": pw,
        "pockets_black": pb,
        "mate_before":   h.get("mate_before"),
        "turn":          h.get("turn", "white"),
        "delta":         None,
        "bestmove":      None,
        "pv":            [],
        "played_move":   None,
    }

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


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
    hits = [_format_hit(h, i + 1) for i, h in enumerate(raw_hits)]

    return jsonify({
        "query": _format_hit(q_hit, 0),
        "hits":  hits,
    })


@app.route("/api/search_by_fen", methods=["POST"])
def search_by_fen():
    data      = request.get_json()
    raw_fen   = (data.get("board_fen") or data.get("fen") or "").strip()
    board_fen = raw_fen.split(" ")[0] if raw_fen else ""
    if "[" in board_fen:
        board_fen = board_fen.split("[")[0]
    parts = board_fen.split("/")
    if len(parts) == 9:
        board_fen = "/".join(parts[:8])

    field = data.get("field", "text_all")
    topk  = int(data.get("topk", 10))

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
    hits = [_format_hit(h, i + 1) for i, h in enumerate(raw_hits)]

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


@app.route("/api/export", methods=["POST"])
def export_labels():
    data = request.get_json()

    # ── Accept both calling conventions ─────────────────────────────────────
    # Frontend sends:  { ids: [queryId, hit1, hit2, ...] }
    if "ids" in data:
        ids         = data["ids"]
        query_id    = ids[0] if ids else ""
        all_hit_ids = ids[1:]
        similar_ids = set(all_hit_ids)   # all exported hits are "similar"
        query_fen   = data.get("query_fen", "")
    else:
        query_id    = data.get("query_id", "")
        query_fen   = data.get("query_fen", "")
        similar_ids = set(data.get("similar_ids", []))
        all_hit_ids = data.get("all_hit_ids", [])

    if not all_hit_ids:
        return jsonify({"error": "no hits provided"}), 400

    q_doc = _lucene_doc(query_id) if query_id else None

    out = io.StringIO()
    for rank, hit_id in enumerate(all_hit_ids, start=1):
        h_doc = _lucene_doc(hit_id) or {}
        record = {
            "query_id":               query_id,
            "candidate_id":           hit_id,
            "label":                  1 if hit_id in similar_ids else 0,
            "bm25_rank":              rank,
            "query_fen":              (q_doc.get("board_fen") if q_doc else "") or query_fen,
            "candidate_fen":          h_doc.get("board_fen", ""),
            "query_mate":             q_doc.get("mate_before") if q_doc else None,
            "candidate_mate":         h_doc.get("mate_before"),
            "query_turn":             q_doc.get("turn", "") if q_doc else "",
            "candidate_turn":         h_doc.get("turn", ""),
            "query_text_static":      q_doc.get("text_static", "") if q_doc else "",
            "candidate_text_static":  h_doc.get("text_static", ""),
            "query_text_dynamic":     q_doc.get("text_dynamic", "") if q_doc else "",
            "candidate_text_dynamic": h_doc.get("text_dynamic", ""),
            # Metadata fields — used by train_similarity.py as meta_ features
            # Engine-derived
            "query_meta_length":           q_doc.get("meta_length") if q_doc else None,
            "query_meta_delta":            q_doc.get("meta_delta") if q_doc else None,
            "query_meta_cp_before":        q_doc.get("meta_cp_before") if q_doc else None,
            "query_meta_mate_in":          q_doc.get("meta_mate_in") if q_doc else None,
            "candidate_meta_length":       h_doc.get("meta_length"),
            "candidate_meta_delta":        h_doc.get("meta_delta"),
            "candidate_meta_cp_before":    h_doc.get("meta_cp_before"),
            "candidate_meta_mate_in":      h_doc.get("meta_mate_in"),
            # Lichess-API-derived (present after enrich_with_ratings.py)
            "query_meta_avg_rating":       q_doc.get("meta_avg_rating") if q_doc else None,
            "query_meta_solver_rating":    q_doc.get("meta_solver_rating") if q_doc else None,
            "query_meta_estimated_time":   q_doc.get("meta_estimated_time") if q_doc else None,
            "query_meta_speed_ord":        q_doc.get("meta_speed_ord") if q_doc else None,
            "candidate_meta_avg_rating":   h_doc.get("meta_avg_rating"),
            "candidate_meta_solver_rating":h_doc.get("meta_solver_rating"),
            "candidate_meta_estimated_time":h_doc.get("meta_estimated_time"),
            "candidate_meta_speed_ord":    h_doc.get("meta_speed_ord"),
        }
        out.write(json.dumps(record, ensure_ascii=False) + "\n")

    out_bytes = out.getvalue().encode("utf-8")
    return send_file(
        io.BytesIO(out_bytes),
        mimetype="application/x-ndjson",
        as_attachment=True,
        download_name=f"labels_{(query_id or 'export')[:20]}_{int(time.time())}.jsonl",
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Starting Crazyhouse Tactical Retrieval server...")
    app.run(host="0.0.0.0", port=5000, debug=False)