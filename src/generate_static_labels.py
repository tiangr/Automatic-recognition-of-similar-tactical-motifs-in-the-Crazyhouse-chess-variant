"""
generate_static_labels.py
--------------------------
Automatically generates similarity labels for Crazyhouse tactical positions
based on STATIC features only (board position, piece counts, king position,
pawn structure, pockets). No engine required.

Pipeline:
  1. Load corpus_checkmates5.jsonl
  2. For each puzzle, query Lucene BM25 for top-K candidates
  3. Compute static similarity score between query and each candidate
  4. Assign label=1 (similar) if score >= threshold, else label=0
  5. Save as dataset_static_labels.jsonl (compatible with train_similarity.py)

Usage:
    python generate_static_labels.py

Configuration (edit below):
    CORPUS_PATH  — path to corpus_checkmates5.jsonl
    LUCENE_URL   — Lucene server URL (must be running)
    OUT_PATH     — output JSONL file
    N_QUERIES    — how many puzzles to use as queries (None = all)
    TOPK         — how many BM25 candidates to retrieve per query
    THRESHOLD    — static similarity score cutoff for label=1

Feature weights summary:
    same_turn         0.20  — both sides to move match
    piece_counts      0.25  — per-piece-type count similarity
    total_material    0.10  — overall piece count difference
    att_pocket        0.10  — attacker's pocket (mating side)
    def_pocket        0.05  — defender's pocket
    king_zone         0.10  — kingside / queenside / centre
    king_rank         0.10  — rank proximity of defending king
    pawn_structure    0.05  — pawn file distribution
    castling_status   0.05  — whether defending king has castled
"""

from pathlib import Path
from collections import Counter
import json
import urllib.parse
import urllib.request
import sys
import random

try:
    import chess
except ImportError:
    print("ERROR: python-chess not installed. Run: pip install chess")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parent.parent
CORPUS_PATH  = ROOT / "data" / "derived" / "corpus_checkmates5.jsonl"
LUCENE_URL   = "http://localhost:8983"
OUT_PATH     = ROOT / "data" / "derived" / "dataset_static_labels.jsonl"

N_QUERIES    = 500    # number of puzzles to use as queries (None = all)
TOPK         = 10     # BM25 candidates per query
THRESHOLD    = 0.63   # static similarity threshold for label=1

# ---------------------------------------------------------------------------
# Lucene helpers
# ---------------------------------------------------------------------------

def lucene_search(query_tokens: str, topk: int = 10,
                  exclude_id: str = "") -> list[dict]:
    params = urllib.parse.urlencode({
        "q":          query_tokens,
        "field":      "text_static",
        "topk":       topk,
        "exclude_id": exclude_id,
    })
    try:
        with urllib.request.urlopen(
            f"{LUCENE_URL}/search?{params}", timeout=15
        ) as r:
            return json.loads(r.read().decode())
    except Exception as e:
        print(f"  Lucene error: {e}")
        return []


def check_lucene() -> bool:
    try:
        with urllib.request.urlopen(
            f"{LUCENE_URL}/status", timeout=5
        ) as r:
            status = json.loads(r.read().decode())
            if status.get("status") == "ok":
                print(f"Lucene connected: {status.get('docs', '?'):,} docs")
                return True
    except Exception:
        pass
    print("ERROR: Lucene server not reachable at", LUCENE_URL)
    print("  Start it: java -Xmx2g -cp '.;../lucene/*' CrazyhouseLuceneServer")
    return False

# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def parse_board(board_fen: str) -> chess.Board | None:
    """Parse board FEN into chess.Board. Returns None on failure."""
    for fen in [board_fen + " w - - 0 1", board_fen]:
        try:
            return chess.Board(fen=fen)
        except Exception:
            continue
    return None


def piece_counts(board_fen: str) -> Counter:
    """
    Count pieces by color+type, e.g. Counter({'wp':6,'bn':2,...}).
    Coordinate-free — only cares about how many of each piece type exist.
    """
    counts = Counter()
    board = parse_board(board_fen)
    if board is None:
        return counts
    for sq, piece in board.piece_map().items():
        color = "w" if piece.color == chess.WHITE else "b"
        counts[color + piece.symbol().lower()] += 1
    return counts


def count_similarity(a: Counter, b: Counter) -> float:
    """
    How similar are two piece-count profiles?
    1.0 = identical counts, 0.0 = completely different.
    Formula: 1 - (sum of abs differences) / (sum of maxima)
    """
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    total_diff = sum(abs(a.get(k, 0) - b.get(k, 0)) for k in keys)
    total_max  = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    return 1.0 - total_diff / total_max if total_max > 0 else 1.0


def king_zone(board_fen: str, color: str) -> str:
    """
    Returns where the king is: 'kingside' (files f-h),
    'queenside' (files a-c), 'centre' (files d-e), or 'unknown'.
    """
    board = parse_board(board_fen)
    if board is None:
        return "unknown"
    king_sq = board.king(chess.WHITE if color == "white" else chess.BLACK)
    if king_sq is None:
        return "unknown"
    file = chess.square_file(king_sq)
    if file >= 5:
        return "kingside"
    if file <= 2:
        return "queenside"
    return "centre"


def king_rank(board_fen: str, color: str) -> int:
    """
    Returns rank of the king (1-8).
    Starting position: rank 1 (white) or 8 (black).
    Mid-game centre king: rank 4-5.
    Returns -1 on failure.
    """
    board = parse_board(board_fen)
    if board is None:
        return -1
    king_sq = board.king(chess.WHITE if color == "white" else chess.BLACK)
    if king_sq is None:
        return -1
    return chess.square_rank(king_sq) + 1   # 1-indexed


def king_castled(board_fen: str, color: str) -> bool:
    """
    Heuristic: is the king castled?
    White castled kingside: king on g1. Queenside: king on c1.
    Black castled kingside: king on g8. Queenside: king on c8.
    """
    board = parse_board(board_fen)
    if board is None:
        return False
    king_sq = board.king(chess.WHITE if color == "white" else chess.BLACK)
    if king_sq is None:
        return False
    sq_name = chess.square_name(king_sq)
    if color == "white":
        return sq_name in ("g1", "c1")
    else:
        return sq_name in ("g8", "c8")


def pawn_file_profile(board_fen: str) -> dict:
    """
    Returns {'w': Counter(file→count), 'b': Counter(file→count)}.
    Captures pawn structure as file distribution.
    """
    profile = {"w": Counter(), "b": Counter()}
    board = parse_board(board_fen)
    if board is None:
        return profile
    for sq, piece in board.piece_map().items():
        if piece.piece_type == chess.PAWN:
            color = "w" if piece.color == chess.WHITE else "b"
            profile[color][chess.square_file(sq)] += 1
    return profile


def pawn_similarity(fen_a: str, fen_b: str) -> float:
    """Average pawn-file-profile similarity across both colours."""
    pa = pawn_file_profile(fen_a)
    pb = pawn_file_profile(fen_b)
    return sum(count_similarity(pa[c], pb[c]) for c in ("w", "b")) / 2


def pawn_rank_similarity(fen_a: str, fen_b: str) -> float:
    """
    Compare pawn RANK distribution — the strongest discriminator between
    starting positions (all pawns on ranks 2/7) and mid-game positions
    (pawns advanced/exchanged across all ranks).
    """
    def rank_profile(fen: str) -> dict:
        profile = {"w": Counter(), "b": Counter()}
        board = parse_board(fen)
        if board is None:
            return profile
        for sq, piece in board.piece_map().items():
            if piece.piece_type == chess.PAWN:
                color = "w" if piece.color == chess.WHITE else "b"
                profile[color][chess.square_rank(sq)] += 1
        return profile

    pa = rank_profile(fen_a)
    pb = rank_profile(fen_b)
    return sum(count_similarity(pa[c], pb[c]) for c in ("w", "b")) / 2


def pocket_multiset(pocket: dict) -> list:
    """{'N':2,'Q':1} → ['N','N','Q']  (sorted)"""
    if not isinstance(pocket, dict):
        return []
    items = []
    for piece, cnt in pocket.items():
        items.extend([piece.upper()] * max(0, int(cnt or 0)))
    return sorted(items)


def multiset_jaccard(a: list, b: list) -> float:
    """Jaccard similarity of two multisets. 0.5 when both empty (neutral)."""
    ca, cb = Counter(a), Counter(b)
    keys = set(ca) | set(cb)
    if not keys:
        return 0.5   # both empty → neutral
    inter = sum(min(ca[k], cb[k]) for k in keys)
    union = sum(max(ca[k], cb[k]) for k in keys)
    return inter / union if union > 0 else 0.0


def parse_pocket_field(val) -> dict:
    """Parse pocket field that may be a dict, JSON string, or None."""
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return {}
    return {}

# ---------------------------------------------------------------------------
# Static similarity — main function
# ---------------------------------------------------------------------------

def static_similarity(query: dict, candidate: dict) -> float:
    """
    Compute static similarity score in [0.0, 1.0].

    Feature               Weight  Description
    ─────────────────────────────────────────────────────────────────────
    same_turn             0.08   1.0 same, 0.7 different (soft penalty)
    piece_counts          0.30   Per-type piece count similarity
    total_material        0.08   Overall piece count closeness
    attacker_pocket       0.09   Pocket of the mating side (Jaccard)
    defender_pocket       0.05   Pocket of the defending side (Jaccard)
    king_zone             0.10   Defending king: kingside/queenside/centre
    king_rank             0.08   Defending king rank proximity
    castling_status       0.05   Both defending kings castled or both not
    pawn_file_structure   0.07   Pawn file distribution similarity
    pawn_rank_structure   0.10   Pawn rank distribution — key discriminator
                                 (starting pos: all on ranks 2/7 vs mid-game)

    Suggested threshold for label=1: score >= 0.63
    """
    q_fen  = query.get("board_fen") or query.get("query_fen") or ""
    c_fen  = candidate.get("board_fen") or candidate.get("candidate_fen") or ""

    def norm_turn(t):
        t = (t or "white").strip().lower()
        return "black" if t.startswith("b") else "white"

    q_turn = norm_turn(query.get("turn") or query.get("query_turn"))
    c_turn = norm_turn(candidate.get("turn") or candidate.get("candidate_turn"))

    score        = 0.0
    total_weight = 0.0

    def add(s: float, w: float):
        nonlocal score, total_weight
        score        += s * w
        total_weight += w

    # ── 1. Same side to move (weight 0.08) ───────────────────────────────
    # Soft penalty: different turn = 0.7 (not 0.0) because the board
    # structure is still similar even if who moves differs.
    add(1.0 if q_turn == c_turn else 0.7, 0.08)

    # ── 2. Piece count similarity per type (weight 0.30) ─────────────────
    q_cnt = piece_counts(q_fen)
    c_cnt = piece_counts(c_fen)
    add(count_similarity(q_cnt, c_cnt), 0.30)

    # ── 3. Total material (weight 0.08) ──────────────────────────────────
    diff = abs(sum(q_cnt.values()) - sum(c_cnt.values()))
    add(max(0.0, 1.0 - diff / 8.0), 0.08)

    # ── 4 & 5. Pockets (weight 0.09 + 0.05) ─────────────────────────────
    q_pw = parse_pocket_field(query.get("pockets_white") or {})
    q_pb = parse_pocket_field(query.get("pockets_black") or {})
    c_pw = parse_pocket_field(candidate.get("pockets_white") or {})
    c_pb = parse_pocket_field(candidate.get("pockets_black") or {})

    q_att = pocket_multiset(q_pb if q_turn == "black" else q_pw)
    c_att = pocket_multiset(c_pb if c_turn == "black" else c_pw)
    add(multiset_jaccard(q_att, c_att), 0.09)

    q_def_pkt = pocket_multiset(q_pw if q_turn == "black" else q_pb)
    c_def_pkt = pocket_multiset(c_pw if c_turn == "black" else c_pb)
    add(multiset_jaccard(q_def_pkt, c_def_pkt), 0.05)

    # ── 6, 7, 8. Defending king position (weight 0.10 + 0.08 + 0.05) ────
    q_def_color = "white" if q_turn == "black" else "black"
    c_def_color = "white" if c_turn == "black" else "black"

    q_kzone = king_zone(q_fen, q_def_color)
    c_kzone = king_zone(c_fen, c_def_color)
    add(1.0 if q_kzone == c_kzone and q_kzone != "unknown"
        else 0.5 if q_kzone == "unknown" else 0.0, 0.10)

    q_krank = king_rank(q_fen, q_def_color)
    c_krank = king_rank(c_fen, c_def_color)
    add(max(0.0, 1.0 - abs(q_krank - c_krank) / 4.0)
        if q_krank > 0 and c_krank > 0 else 0.5, 0.08)

    add(1.0 if king_castled(q_fen, q_def_color) == king_castled(c_fen, c_def_color)
        else 0.0, 0.05)

    # ── 9. Pawn file distribution (weight 0.07) ───────────────────────────
    add(pawn_similarity(q_fen, c_fen), 0.07)

    # ── 10. Pawn rank distribution (weight 0.10) ──────────────────────────
    # Starting position: all pawns on ranks 2/7.
    # Mid-game: pawns advanced/exchanged. Strong discriminator.
    add(pawn_rank_similarity(q_fen, c_fen), 0.10)

    return score / total_weight if total_weight > 0 else 0.0

# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def run_tests():
    """Sanity-check the similarity function with known cases."""
    print("Running self-tests...\n")

    ref = {
        "board_fen":     "r3kbnr/1pp2ppp/p1p2q2/4p3/4P3/2NP1B2/PPP2PRp/R1BQK3",
        "turn":          "black",
        "pockets_white": {"P": 1, "N": 1, "B": 1},
        "pockets_black": {},
    }

    tests = [
        ("1. Identical positions",      ref, ref,                                                                               1, "~0.95"),
        ("2. Same board, diff turn",    ref, {**ref, "turn": "white"},                                                          1, "~0.75"),
        ("3. Starting vs mid-game",     ref, {"board_fen":"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR",
                                              "turn":"white","pockets_white":{},"pockets_black":{}},                            0, "~0.50"),
        ("4. Very similar mid-game",    ref, {"board_fen":"r3kbnr/1pp2ppp/p1p2q2/4p3/4P3/2NP1B2/PPP2P1p/R1BQK2R",
                                              "turn":"black","pockets_white":{"P":1,"N":1},"pockets_black":{}},                 1, "~0.93"),
        ("5. Same turn, diff material", ref, {"board_fen":"6k1/5ppp/8/8/8/8/5PPP/6K1",
                                              "turn":"black","pockets_white":{},"pockets_black":{}},                            0, "~0.35"),
    ]

    all_ok = True
    print(f"{'Test':<32} {'Score':>6}  {'Label':>6}  {'Expected':>8}  OK?")
    print("─" * 68)
    for name, q, c, expected_lbl, expected_score in tests:
        s   = static_similarity(q, c)
        lbl = 1 if s >= THRESHOLD else 0
        ok  = "✓" if lbl == expected_lbl else "✗"
        if lbl != expected_lbl:
            all_ok = False
        print(f"{name:<32} {s:>6.3f}  {lbl:>6}  {expected_score:>8}  {ok}")

    print(f"\nThreshold: {THRESHOLD}")
    print(f"All tests passed: {all_ok}")
    print("─" * 68)

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("generate_static_labels.py — Static Similarity Labeler")
    print("=" * 60)

    if not check_lucene():
        sys.exit(1)

    if not CORPUS_PATH.exists():
        print(f"ERROR: Corpus not found at {CORPUS_PATH}")
        sys.exit(1)

    # Load corpus
    print(f"\nLoading corpus from {CORPUS_PATH} ...")
    corpus = []
    with open(CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                corpus.append(json.loads(line))
            except Exception:
                continue
    print(f"Loaded {len(corpus):,} puzzles")

    # Sample queries
    if N_QUERIES is not None and N_QUERIES < len(corpus):
        random.seed(42)
        queries = random.sample(corpus, N_QUERIES)
        print(f"Sampled {N_QUERIES} queries (seed=42 for reproducibility)")
    else:
        queries = corpus
        print(f"Using all {len(queries)} puzzles as queries")

    # Process
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    total_pairs = 0
    total_pos   = 0
    total_neg   = 0
    n_done      = 0

    print(f"\nSettings: TOPK={TOPK}, THRESHOLD={THRESHOLD}")
    print(f"Output: {OUT_PATH}\n")

    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
        for query in queries:
            query_id  = query.get("id", "")
            board_fen = query.get("board_fen", "")

            if not board_fen:
                n_done += 1
                continue

            # Use text_static tokens for BM25 retrieval
            query_tokens = query.get("text_static", "")
            if not query_tokens:
                n_done += 1
                continue

            hits = lucene_search(query_tokens, topk=TOPK, exclude_id=query_id)

            for rank, hit in enumerate(hits, start=1):
                hit_id = hit.get("id", "")
                if not hit_id or hit_id == query_id:
                    continue

                hit_rec = {
                    "board_fen":     hit.get("board_fen", ""),
                    "turn":          hit.get("turn", "white"),
                    "pockets_white": parse_pocket_field(hit.get("pockets_white") or {}),
                    "pockets_black": parse_pocket_field(hit.get("pockets_black") or {}),
                }

                sim_score = static_similarity(query, hit_rec)
                label     = 1 if sim_score >= THRESHOLD else 0

                record = {
                    # Required by train_similarity.py
                    "query_id":               query_id,
                    "candidate_id":           hit_id,
                    "label":                  label,
                    "bm25_rank":              rank,
                    "query_fen":              board_fen,
                    "candidate_fen":          hit_rec["board_fen"],
                    "query_mate":             query.get("mate_in") or query.get("mate_before"),
                    "candidate_mate":         hit.get("mate_in") or hit.get("mate_before"),
                    "query_turn":             query.get("turn", "white"),
                    "candidate_turn":         hit.get("turn", "white"),
                    "query_text_static":      query.get("text_static", ""),
                    "candidate_text_static":  hit.get("text_static", ""),
                    "query_text_dynamic":     query.get("text_dynamic", ""),
                    "candidate_text_dynamic": hit.get("text_dynamic", ""),
                    "query_solution_san":     query.get("solution_san", []),
                    "candidate_solution_san": hit.get("solution_san", []),
                    "query_solution_uci":     query.get("solution_uci", []),
                    "candidate_solution_uci": hit.get("solution_uci", []),
                    "query_rating_avg":       query.get("game_rating_avg"),
                    "candidate_rating_avg":   hit.get("game_rating_avg"),
                    # Extra analysis column
                    "static_sim_score":       round(sim_score, 4),
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_pairs += 1
                if label == 1:
                    total_pos += 1
                else:
                    total_neg += 1

            n_done += 1
            if n_done % 50 == 0 or n_done == len(queries):
                pct = 100 * total_pos / max(total_pairs, 1)
                print(f"  {n_done:>4}/{len(queries)} queries | "
                      f"{total_pairs:,} pairs | "
                      f"pos={total_pos:,} ({pct:.0f}%) "
                      f"neg={total_neg:,}")

    print()
    print("=" * 60)
    print("DONE")
    print(f"  Queries processed : {n_done:,}")
    print(f"  Total pairs       : {total_pairs:,}")
    print(f"  Label=1 (similar) : {total_pos:,}  "
          f"({100*total_pos/max(total_pairs,1):.1f}%)")
    print(f"  Label=0 (differ.) : {total_neg:,}  "
          f"({100*total_neg/max(total_pairs,1):.1f}%)")
    print(f"  Output            : {OUT_PATH}")
    print()
    print("Next step:")
    print(f"  python train_similarity.py "
          f"--data {OUT_PATH.name} --out output/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_tests()
    print()
    ans = input(
        f"Run full label generation? "
        f"({N_QUERIES or 'all'} queries × top-{TOPK}) [y/N]: "
    )
    if ans.strip().lower() == "y":
        main()
    else:
        print("Aborted. Edit N_QUERIES/TOPK/THRESHOLD at the top and rerun.")