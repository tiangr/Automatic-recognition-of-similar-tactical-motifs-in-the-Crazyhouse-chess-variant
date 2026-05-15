"""
mine_checkmates.py
------------------
Fast parallel pipeline: mine + augment + filter in one pass.
Zero engine calls — pure board logic only.

How it works
------------
1. Termination filter  — skip time forfeits/resignations instantly.
                         Only process Termination = "Normal" games.
2. Checkmate filter    — replay full game, confirm final position is
                         actual checkmate via board.is_checkmate().
3. Solution            — last MATE_IN_PLIES moves actually played.
                         Guaranteed to end in checkmate. No engine needed.
4. Crazyhouse filter   — keep only positions where Crazyhouse mechanics
                         are in play (drop in solution or pocket non-empty).
5. Deduplication       — hash table of seen site URLs, persists on resume.
6. Multiprocessing     — one worker per CPU core, no engine overhead at all.

Usage
-----
    python mine_checkmates.py
    WORKERS=4 python mine_checkmates.py

Output
------
    data/derived/checkmates5.jsonl   — one JSON line per qualifying puzzle

Schema
------
  event_id          "{site}_mate5"
  site / event / white / black / result
  utc_date / utc_time
  white_elo / black_elo / game_rating_avg
  white_rating_diff / black_rating_diff
  white_title / black_title
  time_control / termination / variant
  puzzle_url        None (not in monthly PGN dumps)
  ply               1-indexed ply of puzzle start position
  turn              "white" or "black" (who delivers mate)
  uci_moves         all moves before the puzzle starts
  solution_uci      last 5 UCI moves of the game (ends in checkmate)
  solution_san      same in SAN notation
  board_fen         board FEN at puzzle start
  fen               full Crazyhouse FEN at puzzle start
  pockets_white / pockets_black
  mate_in           5 (always)
  bestmove_before   first move of the solution
  source_pgn        source PGN filename
"""

import io
import os
import queue
import multiprocessing as mp
from pathlib import Path
from typing import Optional

import chess
import chess.pgn
import chess.variant
import orjson

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[1]
DB_DIR      = ROOT.parents[0] / "db"
OUT_JSONL   = ROOT / "data" / "derived" / "checkmates5.jsonl"

# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------
MATE_IN_PLIES = 5
WORKERS       = int(os.environ.get("WORKERS", max(1, mp.cpu_count() - 1)))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PIECE_LETTERS = {
    chess.PAWN:   "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK:   "R",
    chess.QUEEN:  "Q",
}


# ===========================================================================
# Helpers
# ===========================================================================

def _pocket_counts(pocket) -> dict:
    d = {}
    for pt, letter in PIECE_LETTERS.items():
        n = pocket.count(pt)
        if n:
            d[letter] = int(n)
    return d


def _safe_int(s) -> Optional[int]:
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


def _extract_headers(game: chess.pgn.Game) -> dict:
    h  = game.headers
    we = _safe_int(h.get("WhiteElo"))
    be = _safe_int(h.get("BlackElo"))
    return {
        "event":              h.get("Event"),
        "site":               h.get("Site", "?"),
        "white":              h.get("White"),
        "black":              h.get("Black"),
        "result":             h.get("Result"),
        "utc_date":           h.get("UTCDate"),
        "utc_time":           h.get("UTCTime"),
        "white_elo":          we,
        "black_elo":          be,
        "white_rating_diff":  _safe_int(h.get("WhiteRatingDiff")),
        "black_rating_diff":  _safe_int(h.get("BlackRatingDiff")),
        "white_title":        h.get("WhiteTitle"),
        "black_title":        h.get("BlackTitle"),
        "time_control":       h.get("TimeControl"),
        "termination":        h.get("Termination"),
        "variant":            h.get("Variant", "Crazyhouse"),
        "game_rating_avg":    (we + be) // 2 if (we and be) else None,
        "puzzle_url":         None,
    }


def _uci_to_san(board_before: chess.variant.CrazyhouseBoard, uci_moves: list) -> list:
    b, san = board_before.copy(), []
    for uci in uci_moves:
        try:
            mv = b.parse_uci(uci)
            san.append(b.san(mv))
            b.push(mv)
        except Exception:
            san.append(uci)
    return san


def _pockets_nonempty(p: dict) -> bool:
    return bool(p) and any(v > 0 for v in p.values())


def _is_crazyhouse_relevant(sol_uci: list, pw: dict, pb: dict) -> bool:
    return (
        any("@" in m for m in sol_uci)
        or _pockets_nonempty(pw)
        or _pockets_nonempty(pb)
    )


# ===========================================================================
# Worker — no engine, pure board logic
# ===========================================================================

def _worker(task_q: mp.Queue, result_q: mp.Queue):
    while True:
        item = task_q.get()
        if item is None:
            break

        pgn_text, pgn_name = item
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            result_q.put(None)
            continue

        headers = _extract_headers(game)

        # Filter 1: only games that ended normally
        if headers.get("termination") != "Normal":
            result_q.put(None)
            continue

        all_moves = list(game.mainline_moves())

        # Filter 2: enough moves for a puzzle
        if len(all_moves) < MATE_IN_PLIES + 1:
            result_q.put(None)
            continue

        # Filter 3: replay and confirm actual checkmate
        board   = chess.variant.CrazyhouseBoard()
        uci_all = []
        ok      = True
        for mv in all_moves:
            try:
                board.push(mv)
                uci_all.append(mv.uci())
            except Exception:
                ok = False
                break

        if not ok or not board.is_checkmate():
            result_q.put(None)
            continue

        # Solution = last MATE_IN_PLIES moves actually played
        candidate_uci = uci_all[:-MATE_IN_PLIES]
        solution_uci  = uci_all[-MATE_IN_PLIES:]
        ply_idx       = len(candidate_uci) + 1

        # Reconstruct board at puzzle start position
        board_snap = chess.variant.CrazyhouseBoard()
        for u in candidate_uci:
            try:
                board_snap.push(board_snap.parse_uci(u))
            except Exception:
                ok = False
                break

        if not ok:
            result_q.put(None)
            continue

        pw = _pocket_counts(board_snap.pockets[chess.WHITE])
        pb = _pocket_counts(board_snap.pockets[chess.BLACK])

        # Filter 4: must involve Crazyhouse mechanics
        if not _is_crazyhouse_relevant(solution_uci, pw, pb):
            result_q.put(None)
            continue

        rec = {
            "event_id":        f"{headers['site']}_mate{MATE_IN_PLIES}",
            "mate_in":         MATE_IN_PLIES,
            "source_pgn":      pgn_name,
            **headers,
            "ply":             ply_idx,
            "turn":            "white" if board_snap.turn == chess.WHITE else "black",
            "uci_moves":       candidate_uci,
            "solution_uci":    solution_uci,
            "solution_san":    _uci_to_san(board_snap, solution_uci),
            "board_fen":       board_snap.board_fen(),
            "fen":             board_snap.fen(),
            "pockets_white":   pw,
            "pockets_black":   pb,
            "bestmove_before": solution_uci[0],
        }

        result_q.put(rec)


# ===========================================================================
# Main
# ===========================================================================

def _load_seen(path: Path) -> set:
    seen = set()
    if not path.exists():
        return seen
    with path.open("rb") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                seen.add(orjson.loads(raw)["site"])
            except Exception:
                pass
    return seen


def main():
    from tqdm import tqdm

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    pgn_files = sorted(DB_DIR.glob("lichess_db_crazyhouse_rated_*.pgn"))
    if not pgn_files:
        raise FileNotFoundError(f"No PGN files found in {DB_DIR}")

    print(f"Found {len(pgn_files)} PGN file(s)  |  workers: {WORKERS}")
    print(f"Mate-in: {MATE_IN_PLIES}  |  no engine — pure board logic\n")

    seen_sites = _load_seen(OUT_JSONL)
    print(f"Resume: {len(seen_sites)} sites already done.\n")

    task_q   = mp.Queue(maxsize=WORKERS * 8)
    result_q = mp.Queue()

    procs = []
    for _ in range(WORKERS):
        p = mp.Process(target=_worker, args=(task_q, result_q), daemon=True)
        p.start()
        procs.append(p)

    total_games = total_written = total_skipped = 0
    pending = 0

    overall = tqdm(
        total=len(pgn_files),
        desc="PGN files",
        unit="file",
        position=0,
        colour="cyan",
    )

    def drain(block: bool, pbar):
        nonlocal total_written, pending
        while pending > 0:
            try:
                rec = result_q.get(timeout=60 if block else 0.005)
                pending -= 1
                if rec is not None:
                    out.write(orjson.dumps(rec) + b"\n")
                    out.flush()
                    total_written += 1
                    pbar.set_postfix({"puzzles": total_written}, refresh=False)
            except queue.Empty:
                if not block:
                    break

    with OUT_JSONL.open("ab") as out:
        for pgn_path in pgn_files:
            file_written_start = total_written

            pbar = tqdm(
                desc=pgn_path.stem[-10:],
                unit="game",
                position=1,
                leave=False,
                colour="green",
                postfix={"puzzles": total_written},
            )

            with pgn_path.open("r", encoding="utf-8", errors="replace") as f:
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break

                    total_games += 1
                    pbar.update(1)

                    site = game.headers.get("Site", "?")
                    if site in seen_sites:
                        total_skipped += 1
                        continue
                    seen_sites.add(site)

                    buf = io.StringIO()
                    game.accept(chess.pgn.FileExporter(buf))

                    drain(block=False, pbar=pbar)
                    task_q.put((buf.getvalue(), pgn_path.name))
                    pending += 1

            drain(block=True, pbar=pbar)

            file_written = total_written - file_written_start
            pbar.set_description(f"{pgn_path.stem[-10:]} ✓ {file_written} puzzles")
            pbar.close()
            overall.update(1)
            overall.set_postfix({"total_puzzles": total_written})

    overall.close()

    for _ in procs:
        task_q.put(None)
    for p in procs:
        p.join()

    print(f"\n{'='*60}")
    print("GRAND TOTAL")
    print(f"  PGN files : {len(pgn_files)}")
    print(f"  Games     : {total_games}")
    print(f"  Skipped   : {total_skipped}")
    print(f"  Written   : {total_written}")
    print(f"  Output    : {OUT_JSONL}")
    print(f"{'='*60}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()