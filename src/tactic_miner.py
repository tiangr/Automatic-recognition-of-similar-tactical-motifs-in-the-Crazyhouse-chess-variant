from pathlib import Path
import chess.pgn
import chess.variant
import orjson
from tqdm import tqdm

from engine_wrapper import FairyStockfish

ROOT = Path(__file__).resolve().parents[1]
ENGINE_PATH = next((ROOT / "engines").glob("fairy-stockfish-largeboard_x86-64*"))
SAMPLE_PGN = ROOT / "data" / "derived" / "sample_500_games.pgn"
OUT_JSONL = ROOT / "data" / "derived" / "tactics_1k.jsonl"


MAX_PLIES = 80
MOVETIME_MS = 80
DELTA_CP_THRESHOLD = 500


def score_to_cp(cp, mate):
    """
    Normalize engine output into one numeric scale.
    Positive means good for side to move.
    """
    # If there is mate, we see that as game ending blunder
    if mate is not None:
        return 100000 if mate > 0 else -100000
    if cp is None:
        return None
    return float(cp)


def main():
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    eng = FairyStockfish(str(ENGINE_PATH))
    eng.new_game()

    found = 0
    with SAMPLE_PGN.open("r", encoding="utf-8", errors="replace") as f, \
         OUT_JSONL.open("wb") as out:

        pbar = tqdm(total=500, desc="Games")

        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            site = game.headers.get("Site", "?")
            white = game.headers.get("White")
            black = game.headers.get("Black")

            board = chess.variant.CrazyhouseBoard()
            uci_prefix = []

            for ply_idx, move in enumerate(game.mainline_moves(), start=1):
                if ply_idx > MAX_PLIES:
                    break

                if board.is_game_over():
                    break

                # Position BEFORE played move
                prefix_before = uci_prefix.copy()

                cp_before, mate_before, best_before, pv_before = eng.analyze_moves(
                    prefix_before,
                    movetime_ms=MOVETIME_MS,
                )
                s_before = score_to_cp(cp_before, mate_before)
                if s_before is None:
                    # still push the move so the game replay continues
                    try:
                        board.push(move)
                        uci_prefix.append(move.uci())
                    except Exception:
                        break
                    continue

                played_uci = move.uci()

                #Play the actual move
                try:
                    board.push(move)
                    uci_prefix.append(played_uci)
                except Exception:
                    break

                if board.is_game_over():
                    continue

                # Position AFTER played move
                cp_after, mate_after, best_after, pv_after = eng.analyze_moves(
                    uci_prefix,
                    movetime_ms=MOVETIME_MS,
                )
                s_after = score_to_cp(cp_after, mate_after)
                if s_after is None:
                    continue

                # IMPORTANT:
                # before = score for mover
                # after  = score for opponent (because side-to-move changed)
                # so convert after-score back into mover's perspective
                after_for_mover = -s_after
                loss_for_mover = s_before - after_for_mover

                if loss_for_mover >= DELTA_CP_THRESHOLD:
                    rec = {
                        "event_id": f"{site}_{ply_idx}",
                        "site": site,
                        "white": white,
                        "black": black,
                        "ply": ply_idx,

                        # store position BEFORE the played move
                        "uci_moves": prefix_before,

                        # actual played move that caused the swing
                        "played_move": played_uci,

                        # engine recommendation BEFORE the move
                        "bestmove_before": best_before,
                        "pv_before": (pv_before or [])[:12],

                        # raw engine info
                        "cp_before": cp_before,
                        "mate_before": mate_before,
                        "cp_after": cp_after,
                        "mate_after": mate_after,

                        # normalized scores
                        "score_before_stm": s_before,
                        "score_after_stm": s_after,
                        "score_after_for_mover": after_for_mover,

                        #tactical swing
                        "delta": loss_for_mover,
                    }

                    out.write(orjson.dumps(rec))
                    out.write(b"\n")
                    found += 1

                    if found % 50 == 0:
                        print("Found tactics:", found)

                    if found >= 1000:
                        break

            pbar.update(1)
            if found >= 1000:
                break

        pbar.close()

    eng.close()
    print("DONE. tactics:", found, "->", OUT_JSONL)


if __name__ == "__main__":
    main()