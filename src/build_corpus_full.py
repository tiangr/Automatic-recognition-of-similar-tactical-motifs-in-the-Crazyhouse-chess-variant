"""
build_corpus_full.py
--------------------
Builds the BM25 corpus from tactics_full_tactical.jsonl.

Filter options (set FILTER_MODE at the top):
  "all"        — keep every Crazyhouse-relevant event (drops/pockets)
  "mate"       — keep only positions where engine found a forced mate
  "mate_short" — keep only mate-in-2 or mate-in-3 positions

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
FILTER_MODE = "mate"   # "all" | "mate" | "mate_short"
MIN_PLY     = 6
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
INP  = ROOT / "data" / "derived" / "tactics_full_tactical.jsonl"

_OUT_MAP = {
    "all":        ROOT / "data" / "derived" / "corpus_full.jsonl",
    "mate":       ROOT / "data" / "derived" / "corpus_mates.jsonl",
    "mate_short": ROOT / "data" / "derived" / "corpus_mates_short.jsonl",
}
OUT = _OUT_MAP[FILTER_MODE]


def _keep(rec: dict) -> bool:
    if rec.get("ply", 0) < MIN_PLY:
        return False
    if FILTER_MODE == "all":
        return True
    mate = rec.get("mate_before")
    if FILTER_MODE == "mate":
        return mate is not None
    if FILTER_MODE == "mate_short":
        return mate in (2, 3)
    return True


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    n_read = n_skipped = n_written = 0

    print(f"Filter mode : {FILTER_MODE}")
    print(f"Min ply     : {MIN_PLY}")
    print(f"Input       : {INP}")
    print(f"Output      : {OUT}")
    print()

    with INP.open("r", encoding="utf-8", errors="replace") as f, \
         OUT.open("w", encoding="utf-8") as out:

        for line in f:
            if not line.strip():
                continue
            n_read += 1

            rec = json.loads(line)

            if not _keep(rec):
                n_skipped += 1
                continue

            board_fen = rec.get("board_fen")
            if not board_fen:
                n_skipped += 1
                continue

            pw = rec.get("pockets_white", {})
            pb = rec.get("pockets_black", {})

            static_toks = encode_static(board_fen)
            pawn_toks   = encode_pawn_structure(board_fen)   # v2: pawn structure tokens
            pocket_toks = encode_pockets(pw, pb)
            # Merge pawn structure into static tokens so text_static contains them
            static_toks_full = static_toks + pawn_toks
            fields      = encode_corpus_fields(rec, static_toks_full, pocket_toks)

            rec_out = {
                "id":          rec.get("event_id") or (rec.get("site") + "_" + str(rec.get("ply"))),
                "site":        rec.get("site"),
                "ply":         rec.get("ply"),
                "mate_before": rec.get("mate_before"),
                "board_fen":   board_fen,
                "turn":        rec.get("turn", "white"),
                "pockets_white": pw,
                "pockets_black": pb,
                # Metadata fields (section 4.3 of feature_descriptions_summary_2_.md)
                # Engine-derived (always present):
                #   meta_length    = PV length (solution length in half-moves) ~ MD "length"
                #   meta_delta     = centipawn loss at the tactic (difficulty proxy)
                #   meta_cp_before = engine eval before tactic
                #   meta_mate_in   = forced mate depth (None = not a forced mate)
                # Lichess-API-derived (present after running enrich_with_ratings.py):
                #   meta_avg_rating      ~ MD "rating"      (average player Elo)
                #   meta_solver_rating   ~ MD "rating"      (the solver's own Elo)
                #   meta_estimated_time  ~ MD "popularity"  (time control seriousness proxy)
                #   meta_white_rating, meta_black_rating
                #   meta_clock_initial, meta_clock_inc, meta_speed
                "meta_length":          len((rec.get("pv_before") or rec.get("pv_prev") or [])),
                "meta_delta":           rec.get("delta"),
                "meta_cp_before":       rec.get("cp_before"),
                "meta_mate_in":         rec.get("mate_before"),
                "meta_avg_rating":      rec.get("meta_avg_rating"),
                "meta_solver_rating":   rec.get("meta_solver_rating"),
                "meta_white_rating":    rec.get("meta_white_rating"),
                "meta_black_rating":    rec.get("meta_black_rating"),
                "meta_estimated_time":  rec.get("meta_estimated_time"),
                "meta_clock_initial":   rec.get("meta_clock_initial"),
                "meta_clock_inc":       rec.get("meta_clock_inc"),
                "meta_speed":           rec.get("meta_speed"),
                **fields,
            }
            out.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            n_written += 1

            if n_written % 100_000 == 0:
                print(f"  written: {n_written:,}  (read: {n_read:,}, skipped: {n_skipped:,}) …")

    print()
    print("DONE")
    print(f"  read    : {n_read:,}")
    print(f"  skipped : {n_skipped:,}  ({n_skipped/max(n_read,1)*100:.1f}%)")
    print(f"  written : {n_written:,}")
    print(f"  output  : {OUT}")


if __name__ == "__main__":
    main()