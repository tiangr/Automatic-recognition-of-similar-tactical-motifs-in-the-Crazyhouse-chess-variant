import chess.pgn
import chess.variant
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # <- magistrska
PGN_PATH = ROOT / "db" / "lichess_db_crazyhouse_rated_2016-01.pgn"

def main():
    with open(PGN_PATH, "r", encoding="utf-8") as f:
        game = chess.pgn.read_game(f)

    # Force Crazyhouse board (important!)
    board = chess.variant.CrazyhouseBoard()

    uci_moves = []
    for move in game.mainline_moves():
        uci_moves.append(move.uci())   # normal moves: e2e4 ; drops: N@d4 etc.
        board.push(move)

    print("UCI MOVES:")
    print(" ".join(uci_moves))

    print("\nUCI COMMAND TO PASTE INTO ENGINE:")
    print("position startpos moves " + " ".join(uci_moves))

if __name__ == "__main__":
    main()
