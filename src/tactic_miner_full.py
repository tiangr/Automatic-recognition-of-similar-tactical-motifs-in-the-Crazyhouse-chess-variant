"""
tactic_miner_full.py
--------------------
Mine tactical events from ALL monthly PGN files in the db/ folder.

Key differences from tactic_miner.py:
  - Iterates over every lichess_db_crazyhouse_rated_*.pgn in db/
  - No hard cap on number of tactics (mines everything)
  - Resume support: skips games whose site ID is already in the output file
  - Progress is written line-by-line (safe to Ctrl-C and restart)
  - Writes to tactics_full.jsonl (separate from the old tactics_1k.jsonl)
  - Prints a per-file summary and a grand total at the end

Usage:
    python tactic_miner_full.py

Tunable constants (top of file):
    MAX_PLIES          - ignore moves beyond this ply (avoids very long games)
    MOVETIME_MS        - engine think time per position in milliseconds
    DELTA_CP_THRESHOLD - minimum centipawn swing to record a tactical event
"""

from pathlib import Path
import chess.pgn
import chess.variant
import orjson
from tqdm import tqdm

from engine_wrapper import FairyStockfish

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[1]
DB_DIR      = ROOT.parents[0] / "db"           # magistrska/db  (monthly PGNs live here)
OUT_JSONL   = ROOT / "data" / "derived" / "tactics_full.jsonl"
ENGINE_PATH = next((ROOT / "engines").glob("fairy-stockfish-largeboard_x86-64*"))

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------
MAX_PLIES          = 80    # only analyse the first N plies of each game
MOVETIME_MS        = 80    # ms per engine call (keep low for large-scale mining)
DELTA_CP_THRESHOLD = 500   # centipawn loss threshold to flag a tactical event

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def score_to_cp(cp, mate) -> float | None:
    """Normalise engine output to a single numeric scale (side-to-move perspective)."""
    if mate is not None:
        return 100_000.0 if mate > 0 else -100_000.0
    if cp is None:
        return None
    return float(cp)


def load_seen_sites(path: Path) -> set[str]:
    """
    Read already-written output and collect every site ID that was recorded.
    This allows safe resume: run the script again after a crash/stop and it
    will skip all games that produced at least one event in a previous run.

    NOTE: A game that produced ZERO events will be re-analysed on resume
    (we have no record of it).  That is acceptable — it is just a small amount
    of redundant engine work.
    """
    seen = set()
    if not path.exists():
        return seen
    with path.open("rb") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = orjson.loads(raw)
                seen.add(rec["site"])
            except Exception:
                pass
    return seen


def iter_pgn_files(db_dir: Path):
    """Yield PGN files sorted chronologically (they are named by month)."""
    files = sorted(db_dir.glob("lichess_db_crazyhouse_rated_*.pgn"))
    if not files:
        raise FileNotFoundError(f"No crazyhouse PGN files found in {db_dir}")
    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    pgn_files = iter_pgn_files(DB_DIR)
    print(f"Found {len(pgn_files)} PGN file(s) in {DB_DIR}:")
    for p in pgn_files:
        print(f"  {p.name}")

    # Resume: collect site IDs already present in the output file
    seen_sites = load_seen_sites(OUT_JSONL)
    print(f"\nResume: {len(seen_sites)} site(s) already processed — will skip them.\n")

    eng = FairyStockfish(str(ENGINE_PATH))
    eng.new_game()

    total_games   = 0
    total_skipped = 0
    total_found   = 0

    # Open output in append mode so resume works correctly
    with OUT_JSONL.open("ab") as out:

        for pgn_path in pgn_files:
            file_games   = 0
            file_skipped = 0
            file_found   = 0

            print(f"\n{'='*60}")
            print(f"Processing: {pgn_path.name}")
            print(f"{'='*60}")

            with pgn_path.open("r", encoding="utf-8", errors="replace") as f:
                pbar = tqdm(desc=f"  Games ({pgn_path.stem[-7:]})", unit="game")

                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break

                    site  = game.headers.get("Site", "?")
                    white = game.headers.get("White")
                    black = game.headers.get("Black")

                    file_games += 1
                    pbar.update(1)

                    # Resume: skip if we already have events from this game
                    if site in seen_sites:
                        file_skipped += 1
                        continue

                    board      = chess.variant.CrazyhouseBoard()
                    uci_prefix = []
                    game_found = 0

                    for ply_idx, move in enumerate(game.mainline_moves(), start=1):
                        if ply_idx > MAX_PLIES:
                            break
                        if board.is_game_over():
                            break

                        prefix_before = uci_prefix.copy()

                        cp_before, mate_before, best_before, pv_before = eng.analyze_moves(
                            prefix_before,
                            movetime_ms=MOVETIME_MS,
                        )
                        s_before = score_to_cp(cp_before, mate_before)

                        if s_before is None:
                            try:
                                board.push(move)
                                uci_prefix.append(move.uci())
                            except Exception:
                                break
                            continue

                        played_uci = move.uci()

                        try:
                            board.push(move)
                            uci_prefix.append(played_uci)
                        except Exception:
                            break

                        if board.is_game_over():
                            continue

                        cp_after, mate_after, best_after, pv_after = eng.analyze_moves(
                            uci_prefix,
                            movetime_ms=MOVETIME_MS,
                        )
                        s_after = score_to_cp(cp_after, mate_after)
                        if s_after is None:
                            continue

                        after_for_mover  = -s_after
                        loss_for_mover   = s_before - after_for_mover

                        if loss_for_mover >= DELTA_CP_THRESHOLD:
                            rec = {
                                "event_id":            f"{site}_{ply_idx}",
                                "site":                site,
                                "white":               white,
                                "black":               black,
                                "ply":                 ply_idx,
                                "uci_moves":           prefix_before,
                                "played_move":         played_uci,
                                "bestmove_before":     best_before,
                                "pv_before":           (pv_before or [])[:12],
                                "cp_before":           cp_before,
                                "mate_before":         mate_before,
                                "cp_after":            cp_after,
                                "mate_after":          mate_after,
                                "score_before_stm":    s_before,
                                "score_after_stm":     s_after,
                                "score_after_for_mover": after_for_mover,
                                "delta":               loss_for_mover,
                                # track which source file this came from
                                "source_pgn":          pgn_path.name,
                            }
                            out.write(orjson.dumps(rec) + b"\n")
                            out.flush()   # ensure data is on disk after each record
                            game_found += 1
                            file_found += 1
                            total_found += 1

                    # Mark site as seen so we won't re-process it on resume
                    seen_sites.add(site)
                    file_skipped += (game_found == 0)  # no-event games are "implicitly" processed

                pbar.close()

            total_games   += file_games
            total_skipped += file_skipped

            print(f"  Games in file : {file_games}")
            print(f"  Skipped (resume): {file_skipped}")
            print(f"  Tactics found : {file_found}")

    eng.close()

    print(f"\n{'='*60}")
    print("GRAND TOTAL")
    print(f"  PGN files processed : {len(pgn_files)}")
    print(f"  Games seen          : {total_games}")
    print(f"  Tactics written     : {total_found}")
    print(f"  Output              : {OUT_JSONL}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
