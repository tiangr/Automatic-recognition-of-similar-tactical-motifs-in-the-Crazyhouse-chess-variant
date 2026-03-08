from pathlib import Path
import chess.pgn

ROOT = Path(__file__).resolve().parents[1]
PGN_IN = ROOT.parents[0] / "db" / "lichess_db_crazyhouse_rated_2016-01.pgn"  # magistrska/db
PGN_OUT = ROOT / "data" / "derived" / "sample_500_games.pgn"

def main(n_games: int = 500):
    PGN_OUT.parent.mkdir(parents=True, exist_ok=True)

    with PGN_IN.open("r", encoding="utf-8", errors="replace") as f_in, \
         PGN_OUT.open("w", encoding="utf-8") as f_out:
        for i in range(n_games):
            game = chess.pgn.read_game(f_in)
            if game is None:
                break
            print(game, file=f_out, end="\n\n")
            if (i + 1) % 50 == 0:
                print("Wrote", i + 1, "games")

    print("DONE:", PGN_OUT)

if __name__ == "__main__":
    main()
