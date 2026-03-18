"""
augment_tactics_full.py
-----------------------
Augments tactics_full.jsonl (mined from all monthly PGNs) with
reconstructed board state, pockets, and FEN.

Identical logic to augment_tactics.py — only the input/output paths differ.
"""

from pathlib import Path
import json
import chess
import chess.variant

ROOT = Path(__file__).resolve().parents[1]
INP  = ROOT / "data" / "derived" / "tactics_full.jsonl"
OUT  = ROOT / "data" / "derived" / "tactics_full_aug.jsonl"


PIECE_LETTERS = {
    chess.PAWN:   "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK:   "R",
    chess.QUEEN:  "Q",
}


def pocket_counts(pocket) -> dict:
    d = {}
    for pt, letter in PIECE_LETTERS.items():
        n = pocket.count(pt)
        if n:
            d[letter] = int(n)
    return d


def reconstruct_event_board(rec: dict):
    uci_moves = rec.get("uci_moves") or rec.get("uci_moves_prefix") or []
    ply       = rec.get("ply")

    board = chess.variant.CrazyhouseBoard()

    if not isinstance(uci_moves, list):
        raise ValueError("uci_moves is not a list")

    prefix = uci_moves[: max(0, ply - 1)] if isinstance(ply, int) and ply > 0 else uci_moves

    for u in prefix:
        mv = board.parse_uci(u)
        board.push(mv)

    return board, prefix


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_ok = n_bad = n_terminal = 0

    with INP.open("r", encoding="utf-8", errors="replace") as f, \
         OUT.open("w", encoding="utf-8") as out:

        for line in f:
            line = line.strip()
            if not line:
                continue
            n_in += 1

            try:
                rec = json.loads(line)
            except Exception:
                n_bad += 1
                continue

            try:
                board, prefix = reconstruct_event_board(rec)
            except Exception:
                n_bad += 1
                continue

            if board.is_game_over():
                n_terminal += 1
                continue

            rec2 = dict(rec)
            rec2["event_id"]       = rec.get("event_id") or f"{rec.get('site', '?')}_{rec.get('ply', '?')}"
            rec2["uci_moves_prefix"] = prefix
            rec2["board_fen"]      = board.board_fen()
            rec2["turn"]           = "white" if board.turn == chess.WHITE else "black"
            rec2["pockets_white"]  = pocket_counts(board.pockets[chess.WHITE])
            rec2["pockets_black"]  = pocket_counts(board.pockets[chess.BLACK])
            try:
                rec2["fen"] = board.fen()
            except Exception:
                rec2["fen"] = None

            out.write(json.dumps(rec2, ensure_ascii=False) + "\n")
            n_ok += 1

            if n_ok % 5000 == 0:
                print(f"  augmented: {n_ok} …")

    print("DONE")
    print(f"  in:               {n_in}")
    print(f"  augmented (ok):   {n_ok}")
    print(f"  bad/parse errors: {n_bad}")
    print(f"  terminal skipped: {n_terminal}")
    print(f"  out:              {OUT}")


if __name__ == "__main__":
    main()
