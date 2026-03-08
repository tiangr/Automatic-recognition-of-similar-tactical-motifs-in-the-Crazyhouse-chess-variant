from pathlib import Path
from engine_wrapper import FairyStockfish

ROOT = Path(__file__).resolve().parents[1]
ENGINES_DIR = ROOT / "engines"

ENGINE = next(ENGINES_DIR.glob("fairy-stockfish-largeboard_x86-64*"))
ENGINE = str(ENGINE)

print("ENGINE PATH:", ENGINE)

MOVES = "e2e4 e7e5 g1f3 d7d6 b1c3 c8g4 f1e2 b8c6 e1g1 g4f3 e2f3 c6d4 d2d3 g8f6 c1g5 f8e7 g5f6 d4f3 d1f3 N@d4 f3d1 e7f6 c3d5 B@c6 d5f6 d8f6 c2c3 N@f3 g2f3 d4f3 g1h1 B@h3 B@g2 h3g2 h1g2 P@g4 f1h1 B@h3 g2g3 c6e4 N@g2 P@h4 g2h4".split()

def main():
    eng = FairyStockfish(ENGINE)
    eng.new_game()
    cp, mate, best, pv = eng.analyze_moves(MOVES, movetime_ms=200)
    print("cp:", cp, "mate:", mate)
    print("bestmove:", best)
    print("pv:", " ".join(pv[:12]))
    eng.close()

if __name__ == "__main__":
    main()
