import subprocess
import time
from typing import List, Optional, Tuple

class FairyStockfish:
    def __init__(self, engine_path: str):
        self.p = subprocess.Popen(
            [engine_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
        )
        self._uci_handshake()

    def _send(self, cmd: str) -> None:
        assert self.p.stdin is not None
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()

    def _readline(self, timeout: float = 10.0) -> str:
        assert self.p.stdout is not None
        start = time.time()
        while time.time() - start < timeout:
            line = self.p.stdout.readline()
            if line:
                return line.strip()
        raise TimeoutError("Engine read timeout")

    def _read_until(self, token: str, timeout: float = 10.0) -> List[str]:
        start = time.time()
        lines = []
        while time.time() - start < timeout:
            line = self._readline(timeout=timeout)
            lines.append(line)
            if token in line:
                return lines
        raise TimeoutError(f"Did not see '{token}'. Last lines: {lines[-10:]}")

    def _uci_handshake(self) -> None:
        self._send("uci")
        self._read_until("uciok", timeout=10.0)
        self._send("isready")
        self._read_until("readyok", timeout=10.0)

        # Set variant
        self._send("setoption name UCI_Variant value crazyhouse")
        self._send("isready")
        self._read_until("readyok", timeout=10.0)

    def new_game(self) -> None:
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok", timeout=10.0)

    def analyze_moves(
        self,
        uci_moves: List[str],
        movetime_ms: int = 200
    ) -> Tuple[Optional[float], Optional[int], str, List[str]]:
        """
        Returns:
          (cp_score, mate_score, bestmove, pv_moves)
        - cp_score: centipawns from side-to-move perspective if available
        - mate_score: mate in N (positive means side to move mates) if available
        """
        self._send("position startpos moves " + " ".join(uci_moves))
        self._send(f"go movetime {movetime_ms}")

        bestmove = ""
        last_score_cp = None
        last_score_mate = None
        last_pv = []

        while True:
            line = self._readline(timeout=30.0)

            # Parse info lines
            if line.startswith("info "):
                # Example: info depth 10 score cp 34 pv e2e4 ...
                parts = line.split()
                if "score" in parts:
                    i = parts.index("score")
                    if i + 2 < len(parts):
                        stype = parts[i + 1]
                        sval = parts[i + 2]
                        if stype == "cp":
                            try:
                                last_score_cp = float(sval)
                                last_score_mate = None
                            except ValueError:
                                pass
                        elif stype == "mate":
                            try:
                                last_score_mate = int(sval)
                                last_score_cp = None
                            except ValueError:
                                pass
                if "pv" in parts:
                    j = parts.index("pv")
                    last_pv = parts[j + 1 :]

            if line.startswith("bestmove"):
                # bestmove e2e4
                bestmove = line.split()[1] if len(line.split()) > 1 else ""
                break

        return last_score_cp, last_score_mate, bestmove, last_pv

    def close(self) -> None:
        try:
            self._send("quit")
        except Exception:
            pass
        if self.p.poll() is None:
            self.p.kill()
