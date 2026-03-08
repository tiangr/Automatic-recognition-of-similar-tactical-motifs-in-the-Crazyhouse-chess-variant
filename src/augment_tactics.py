from pathlib import Path
import json
import chess
import chess.variant

ROOT = Path(__file__).resolve().parents[1]
INP = ROOT / "data" / "derived" / "tactics_1k.jsonl"
OUT = ROOT / "data" / "derived" / "tactics_1k_aug.jsonl"


PIECE_LETTERS = {
    chess.PAWN: "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK: "R",
    chess.QUEEN: "Q",
}


def pocket_counts(pocket) -> dict:
    d = {}
    for pt, letter in PIECE_LETTERS.items():
        n = pocket.count(pt)
        if n:
            d[letter] = int(n)
    return d


def reconstruct_event_board(rec: dict):
    """
    Reconstruct the event position = position BEFORE played_move.

    Supports both conventions:
    1) uci_moves is the full game move list + ply tells us where the event is
    2) uci_moves is already the event prefix (length == ply-1)
    """
    uci_moves = rec.get("uci_moves") or rec.get("uci_moves_prefix") or []
    ply = rec.get("ply")

    board = chess.variant.CrazyhouseBoard()

    if not isinstance(uci_moves, list):
        raise ValueError("uci_moves is not a list")

    # If ply is available, reconstruct "position before played move" via [:ply-1]
    if isinstance(ply, int) and ply > 0:
        prefix = uci_moves[: max(0, ply - 1)]
    else:
        # fallback: assume uci_moves already is the event prefix
        prefix = uci_moves

    for u in prefix:
        mv = board.parse_uci(u)
        board.push(mv)

    return board, prefix


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_ok = 0
    n_bad = 0
    n_terminal = 0

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

            # useful normalization / debug
            rec2["event_id"] = rec.get("event_id") or f"{rec.get('site', '?')}_{rec.get('ply', '?')}"
            rec2["uci_moves_prefix"] = prefix

            # required by later pipeline
            rec2["board_fen"] = board.board_fen()
            rec2["turn"] = "white" if board.turn == chess.WHITE else "black"
            rec2["pockets_white"] = pocket_counts(board.pockets[chess.WHITE])
            rec2["pockets_black"] = pocket_counts(board.pockets[chess.BLACK])

            #FEN
            try:
                rec2["fen"] = board.fen()
            except Exception:
                rec2["fen"] = None

            out.write(json.dumps(rec2, ensure_ascii=False) + "\n")
            n_ok += 1

    print("DONE")
    print("in:", n_in)
    print("aug:", n_ok)
    print("bad:", n_bad)
    print("terminal skipped:", n_terminal)
    print("out:", OUT)


if __name__ == "__main__":
    main()