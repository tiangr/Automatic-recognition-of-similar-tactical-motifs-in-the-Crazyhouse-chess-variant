"""
verify_checkmates.py
--------------------
Reads checkmates5.jsonl, verifies each puzzle with Fairy-Stockfish + NNUE,
replaces solution with engine's optimal mating line.
 
Logic
-----
1. Board check  — replay uci_moves + solution_uci on CrazyhouseBoard.
                  Discard instantly if final position is not checkmate.
                  Fast, no engine needed.
2. Engine check — `go movetime MOVETIME_MS` with NNUE.
                  Engine must find a forced mate (any length).
                  PV is replayed move by move until is_checkmate() — this
                  gives us the exact mating sequence, no guessing on length.
3. Replace      — solution_uci/san replaced with engine's optimal line.
                  mate_in updated to actual engine-found value.
                  All _before fields removed (not needed per mentor).
 
Setup
-----
Place crazyhouse-*.nnue in engines/ folder before running.
 
Usage
-----
    python verify_checkmates.py
    WORKERS=4 python verify_checkmates.py
 
    Resume from a specific game: change START_FROM_GAME at the top.
"""
 
import os
import queue
import subprocess
import time
import multiprocessing as mp
from pathlib import Path
 
import chess
import chess.variant
import orjson
 
# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[1]
IN_JSONL    = ROOT / "data" / "derived" / "checkmates5.jsonl"
OUT_JSONL   = ROOT / "data" / "derived" / "temp.jsonl"
ENGINE_PATH = str(next((ROOT / "engines").glob("fairy-stockfish-largeboard_x86-64*")))
NNUE_PATH   = str(next((ROOT / "engines").glob("crazyhouse-*.nnue")))
 
# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
MATE_IN_PLIES   = 5      # only keep puzzles with exactly 5-ply mating sequence
MOVETIME_MS     = 5000   # 5s per position — faster test run
START_FROM_GAME = 6073472   # change to resume from a specific game number... temp.jsonl of 156266 excluding naprej = ko bos mergov fajle
WORKERS         = int(os.environ.get("WORKERS", max(1, mp.cpu_count() - 1)))
 
# Fields to remove from output — leftovers from old pipeline
FIELDS_TO_REMOVE = {"mate_before", "cp_before", "bestmove_before", "pv_before",
                    "cp_after", "mate_after", "score_before_stm", "score_after_stm",
                    "score_after_for_mover", "delta", "played_move", "uci_moves_prefix",
                    "mate_in_plies"}
 
 
# ===========================================================================
# Engine
# ===========================================================================
 
class _Engine:
    def __init__(self, engine_path: str, nnue_path: str):
        self._p = subprocess.Popen(
            [engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=1,
            universal_newlines=True,
        )
        self._handshake(nnue_path)
 
    def _send(self, cmd: str):
        self._p.stdin.write(cmd + "\n")
        self._p.stdin.flush()
 
    def _read_until(self, token: str, timeout: float = 15.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if token in self._p.stdout.readline().strip():
                return
        raise TimeoutError(f"Engine never saw '{token}'")
 
    def _handshake(self, nnue_path: str):
        self._send("uci")
        self._read_until("uciok")
        self._send("setoption name UCI_Variant value crazyhouse")
        self._send("setoption name Use NNUE value true")
        self._send(f"setoption name EvalFile value {nnue_path}")
        self._send("isready")
        self._read_until("readyok")
 
    def new_game(self):
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok")
 
    def find_mate(self, uci_moves: list):
        """
        Search for MOVETIME_MS ms. Returns (mate_score, pv) or (None, []).
        mate_score is positive int = side to move delivers mate.
        """
        self._send("position startpos moves " + " ".join(uci_moves))
        self._send(f"go movetime {MOVETIME_MS}")
 
        mate     = None
        pv       = []
        deadline = time.time() + (MOVETIME_MS / 1000) + 15.0
 
        while time.time() < deadline:
            line = self._p.stdout.readline().strip()
            if not line:
                continue
            if line.startswith("info "):
                parts = line.split()
                if "score" in parts:
                    i = parts.index("score")
                    if i + 2 < len(parts) and parts[i + 1] == "mate":
                        try:
                            m = int(parts[i + 2])
                            if m > 0:
                                mate = m
                        except ValueError:
                            pass
                if "pv" in parts:
                    pv = parts[parts.index("pv") + 1:]
            if line.startswith("bestmove"):
                break
 
        return mate, pv
 
    def close(self):
        try:
            self._send("quit")
        except Exception:
            pass
        if self._p.poll() is None:
            self._p.kill()
 
 
# ===========================================================================
# Helpers
# ===========================================================================
 
def _uci_to_san(board: chess.variant.CrazyhouseBoard, uci_moves: list) -> list:
    """Convert UCI moves to SAN, correctly handling drop moves (P@g2 etc)."""
    b, san = board.copy(), []
    for uci in uci_moves:
        try:
            mv = b.parse_uci(uci)
            san.append(b.san(mv))
            b.push(mv)
        except Exception:
            san.append(uci)
    return san
 
 
def _extract_solution(board_snap: chess.variant.CrazyhouseBoard,
                      pv: list) -> tuple[list, list] | tuple[None, None]:
    """
    Replay PV on board_snap move by move, stop at first checkmate.
    Returns (solution_uci, solution_san) or (None, None) if PV
    never reaches checkmate.
    """
    b            = board_snap.copy()
    solution_uci = []
    solution_san = []
 
    for uci in pv:
        try:
            mv = b.parse_uci(uci)
            san = b.san(mv)
            b.push(mv)
            solution_uci.append(uci)
            solution_san.append(san)
        except Exception:
            break
 
        if b.is_checkmate():
            return solution_uci, solution_san
 
    return None, None
 
 
def _clean(rec: dict) -> dict:
    """Remove legacy _before/_after fields."""
    return {k: v for k, v in rec.items() if k not in FIELDS_TO_REMOVE}
 
 
# ===========================================================================
# Worker
# ===========================================================================
 
def _worker(task_q: mp.Queue, result_q: mp.Queue, engine_path: str, nnue_path: str):
    def make_engine():
        e = _Engine(engine_path, nnue_path)
        e.new_game()
        return e
 
    eng = make_engine()
 
    while True:
        item = task_q.get()
        if item is None:
            break
 
        # Restart engine if it crashed
        if eng._p.poll() is not None:
            try:
                eng.close()
            except Exception:
                pass
            eng = make_engine()
 
        rec          = item
        uci_moves    = rec.get("uci_moves", [])
        solution_uci = rec.get("solution_uci", [])
 
        # Step 1: board check — replay played solution, confirm checkmate
        board = chess.variant.CrazyhouseBoard()
        ok    = True
        for u in uci_moves:
            try:
                board.push(board.parse_uci(u))
            except Exception:
                ok = False
                break
 
        if not ok:
            result_q.put(None)
            continue
 
        board_snap = board.copy()  # puzzle start position
 
        for u in solution_uci:
            try:
                board.push(board.parse_uci(u))
            except Exception:
                ok = False
                break
 
        if not ok or not board.is_checkmate():
            result_q.put(None)
            continue
 
        # Step 2: engine finds optimal mating line with NNUE
        try:
            mate, pv = eng.find_mate(uci_moves)
        except Exception:
            try:
                eng.close()
            except Exception:
                pass
            eng = make_engine()
            result_q.put(None)
            continue
 
        if not mate or mate <= 0 or not pv:
            result_q.put(None)
            continue
 
        # Step 3: replay PV until checkmate to get exact solution
        opt_uci, opt_san = _extract_solution(board_snap, pv)
        if opt_uci is None:
            result_q.put(None)
            continue
 
        # Must be exactly MATE_IN_PLIES plies — discard shorter or longer
        if len(opt_uci) != MATE_IN_PLIES:
            result_q.put(None)
            continue
 
        # Build clean output record
        out_rec = _clean(rec)
        out_rec.update({
            "mate_in":         len(opt_uci) // 2 + 1,
            "solution_uci":    opt_uci,
            "solution_san":    opt_san,
            "engine_verified": True,
        })
 
        result_q.put(out_rec)
 
    eng.close()
 
 
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
 
    if not IN_JSONL.exists():
        raise FileNotFoundError(f"Input not found: {IN_JSONL}")
 
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
 
    total_records = sum(1 for line in IN_JSONL.open("rb") if line.strip())
 
    print(f"Input      : {IN_JSONL}  ({total_records:,} records)")
    print(f"Output     : {OUT_JSONL}")
    print(f"Engine     : {ENGINE_PATH}")
    print(f"NNUE       : {NNUE_PATH}")
    print(f"Movetime   : {MOVETIME_MS // 1000}s per position")
    print(f"Start from : game {START_FROM_GAME}")
    print(f"Workers    : {WORKERS}\n")
 
    seen_sites = _load_seen(OUT_JSONL)
    print(f"Resume: {len(seen_sites):,} already verified.\n")
 
    task_q   = mp.Queue(maxsize=WORKERS * 8)
    result_q = mp.Queue()
 
    procs = []
    for _ in range(WORKERS):
        p = mp.Process(
            target=_worker,
            args=(task_q, result_q, ENGINE_PATH, NNUE_PATH),
            daemon=True,
        )
        p.start()
        procs.append(p)
 
    total_read = total_written = total_skipped = 0
    pending = 0
 
    pbar = tqdm(
        total=total_records,
        desc="Verifying",
        unit="puzzle",
        colour="cyan",
        postfix={"verified": 0, "rejected": 0},
    )
 
    def drain(block: bool):
        nonlocal total_written, pending
        while pending > 0:
            try:
                rec = result_q.get(
                    timeout=(MOVETIME_MS / 1000 + 30) if block else 0.005
                )
                pending -= 1
                if rec is not None:
                    out.write(orjson.dumps(rec) + b"\n")
                    out.flush()
                    total_written += 1
                pbar.set_postfix(
                    {"verified": total_written,
                     "rejected": total_read - total_skipped - total_written},
                    refresh=False,
                )
            except queue.Empty:
                if not block:
                    break
 
    with OUT_JSONL.open("ab") as out:
        with IN_JSONL.open("rb") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = orjson.loads(raw)
                except Exception:
                    continue
 
                total_read += 1
 
                if total_read < START_FROM_GAME:
                    pbar.update(1)
                    continue
 
                pbar.update(1)
 
                site = rec.get("site", "")
                if site in seen_sites:
                    total_skipped += 1
                    continue
                seen_sites.add(site)
 
                drain(block=False)
                task_q.put(rec)
                pending += 1
 
        drain(block=True)
 
    pbar.close()
 
    for _ in procs:
        task_q.put(None)
    for p in procs:
        p.join()
 
    print(f"\n{'='*60}")
    print("GRAND TOTAL")
    print(f"  Records read     : {total_read:,}")
    print(f"  Skipped (resume) : {total_skipped:,}")
    print(f"  Verified & kept  : {total_written:,}")
    print(f"  Rejected         : {total_read - total_skipped - total_written:,}")
    print(f"  Output           : {OUT_JSONL}")
    print(f"{'='*60}")
 
 
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
