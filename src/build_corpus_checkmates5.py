"""
build_corpus_full.py
--------------------
Builds the BM25 corpus from checkmates5_verified.jsonl.

Input is already filtered (engine-verified forced mate-in-3, Crazyhouse
relevant) so no additional filtering is needed here.

Output corpus records contain five text fields:
  text_dynamic_general   — whole-solution flags (?#, ?px, !#q, !SN, ...)
  text_dynamic_solution  — per-move events (!-R, !xq, !Q>k, !@N, ...)
  text_dynamic           — gen + sol combined (for single-field BM25)
  text_static            — board piece placement + pocket tokens
  text_all               — weighted: gen×3 + sol×2 + pocket×2 + static×1
"""

from pathlib import Path
import json

from encode   import encode_static, encode_pockets, encode_pawn_structure
from encodev2 import encode_corpus_fields

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MIN_PLY = 6   # safety filter — skip extremely short games
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
INP  = ROOT / "data" / "derived" / "checkmates5_600k.jsonl"
OUT  = ROOT / "data" / "derived" / "corpus_checkmates5.jsonl"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    n_read = n_skipped = n_written = 0

    print(f"Input  : {INP}")
    print(f"Output : {OUT}")
    print(f"Min ply: {MIN_PLY}")
    print()

    with INP.open("r", encoding="utf-8", errors="replace") as f, \
         OUT.open("w", encoding="utf-8") as out:

        for line in f:
            if not line.strip():
                continue
            n_read += 1

            rec = json.loads(line)

            # Skip positions too early in the game
            if rec.get("ply", 0) < MIN_PLY:
                n_skipped += 1
                continue

            board_fen = rec.get("board_fen")
            if not board_fen:
                n_skipped += 1
                continue

            pw = rec.get("pockets_white", {})
            pb = rec.get("pockets_black", {})

            static_toks      = encode_static(board_fen)
            pawn_toks        = encode_pawn_structure(board_fen)
            pocket_toks      = encode_pockets(pw, pb)
            static_toks_full = static_toks + pawn_toks
            fields           = encode_corpus_fields(rec, static_toks_full, pocket_toks)

            # Build doc id — prefer event_id, fall back to site+ply
            doc_id = (
                rec.get("event_id")
                or f"{rec.get('site', '?')}_{rec.get('ply', 0)}"
            )

            rec_out = {
                "id":          doc_id,
                "site":        rec.get("site"),
                "ply":         rec.get("ply"),
                "board_fen":   board_fen,
                "turn":        rec.get("turn", "white"),
                "pockets_white": pw,
                "pockets_black": pb,

                # Puzzle metadata
                "mate_in":          rec.get("mate_in"),
                "solution_uci":     rec.get("solution_uci", []),
                "solution_san":     rec.get("solution_san", []),
                "engine_verified":  rec.get("engine_verified", False),

                # Game metadata (for display / filtering in webapp)
                "white":            rec.get("white"),
                "black":            rec.get("black"),
                "white_elo":        rec.get("white_elo"),
                "black_elo":        rec.get("black_elo"),
                "game_rating_avg":  rec.get("game_rating_avg"),
                "time_control":     rec.get("time_control"),
                "utc_date":         rec.get("utc_date"),
                "result":           rec.get("result"),
                "source_pgn":       rec.get("source_pgn"),

                # Meta fields used by retrieval / re-ranking
                # mate_before is gone — use mate_in as the mate depth
                "mate_before":      rec.get("mate_in"),   # kept for backwards compat with app.py
                "meta_mate_in":     rec.get("mate_in"),
                "meta_length":      len(rec.get("solution_uci") or []),
                "meta_avg_rating":  rec.get("game_rating_avg"),
                "meta_white_rating": rec.get("white_elo"),
                "meta_black_rating": rec.get("black_elo"),

                # Fields not available in new data — set to None for compatibility
                "meta_delta":       None,
                "meta_cp_before":   None,
                "meta_solver_rating": None,
                "meta_estimated_time": None,
                "meta_clock_initial": None,
                "meta_clock_inc":   None,
                "meta_speed":       None,

                **fields,
            }

            out.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            n_written += 1

            if n_written % 10_000 == 0:
                print(f"  written: {n_written:,}  (read: {n_read:,}, skipped: {n_skipped:,}) …")

    print()
    print("DONE")
    print(f"  read    : {n_read:,}")
    print(f"  skipped : {n_skipped:,}  ({n_skipped / max(n_read, 1) * 100:.1f}%)")
    print(f"  written : {n_written:,}")
    print(f"  output  : {OUT}")


if __name__ == "__main__":
    main()