from __future__ import annotations

from pathlib import Path
import json
import argparse

import chess.pgn
import chess.variant

from engine_wrapper import FairyStockfish
from encode import encode_static, encode_pockets
from encodev2 import encode_dynamic_v2


PIECE_LETTERS = {
    chess.PAWN: "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK: "R",
    chess.QUEEN: "Q",
    chess.KING: "K",
}

def pockets_to_dict(board: chess.variant.CrazyhouseBoard):
    pw, pb = {}, {}
    for color, out in [(chess.WHITE, pw), (chess.BLACK, pb)]:
        pocket = board.pockets[color]
        for pt, letter in PIECE_LETTERS.items():
            if letter == "K":
                continue
            c = pocket.count(pt)
            if c:
                out[letter] = int(c)
    return pw, pb

def parse_query_id(query_id: str):
    # expects: "<site>_<ply>"
    base, ply_str = query_id.rsplit("_", 1)
    return base, int(ply_str)

def find_game_by_site(pgn_path: Path, site_prefix: str):
    """
    Finds the first game whose [Site "..."] header starts with site_prefix.
    Example site_prefix: "https://lichess.org/N7w3FJpI"
    """
    with pgn_path.open("r", encoding="utf-8", errors="replace") as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                break
            site = g.headers.get("Site", "")
            if site.startswith(site_prefix):
                return g
    return None

def game_to_uci_moves(game: chess.pgn.Game):
    board = chess.variant.CrazyhouseBoard()
    moves = []
    for mv in game.mainline_moves():
        moves.append(mv.uci())
        board.push(mv)
    return moves

def tag_for_offset(off: int) -> str:
    if off < 0:
        return f"m{abs(off)}"
    if off > 0:
        return f"p{off}"
    return "p0"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus_in", type=Path, default=Path("data/derived/corpus_tactical.jsonl"))
    ap.add_argument("--corpus_out", type=Path, default=Path("data/derived/corpus_tactical_with_queryply.jsonl"))
    ap.add_argument("--pgn_in", type=Path, default=Path("data/raw/sample_500_games.pgn"))
    ap.add_argument("--engine_path", type=str, required=True)
    ap.add_argument("--query_id", type=str, required=True)  # e.g. https://lichess.org/N7w3FJpI_14
    ap.add_argument("--radius", type=int, default=10)        # +-10
    ap.add_argument("--movetime_ms", type=int, default=200)
    args = ap.parse_args()

    base_site, query_ply = parse_query_id(args.query_id)

    # Load original corpus lines
    corpus_docs = []
    with args.corpus_in.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                corpus_docs.append(json.loads(line))

    # Find the query game in PGN
    game = find_game_by_site(args.pgn_in, base_site)
    if not game:
        raise SystemExit(f"Could not find a game in {args.pgn_in} with Site starting '{base_site}'")

    full_uci_moves = game_to_uci_moves(game)
    max_ply_available = len(full_uci_moves) + 1  # ply refers to "position before move ply" => ply can be len+1

    engine = FairyStockfish(args.engine_path)
    engine.new_game()

    synth_docs = []

    for off in range(-args.radius, args.radius + 1):
        target_ply = query_ply + off
        if target_ply < 1 or target_ply > max_ply_available:
            continue

        # Position "before played move at ply" => push moves up to ply-1
        prefix = full_uci_moves[:max(0, target_ply - 1)]

        # reconstruct board for static + pockets
        board = chess.variant.CrazyhouseBoard()
        for u in prefix:
            board.push(board.parse_uci(u))

        board_fen = board.board_fen()
        pw, pb = pockets_to_dict(board)

        # engine analysis to get bestmove/pv for dynamic encoding
        cp, mate, bestmove, pv = engine.analyze_moves(prefix, movetime_ms=args.movetime_ms)

        # build a rec compatible with encode_dynamic_v2 (needs uci_moves + ply + pv_before + bestmove_before)
        rec_for_dyn = {
            "uci_moves": full_uci_moves,
            "ply": target_ply,
            "pv_before": pv,
            "bestmove_before": bestmove,
            "board_fen": board_fen,
            "pockets_white": pw,
            "pockets_black": pb,
        }

        static = encode_static(board_fen)
        pocket = encode_pockets(pw, pb)
        dynamic = encode_dynamic_v2(rec_for_dyn)

        tag = tag_for_offset(off)

        # IMPORTANT: make it a "separate game" by changing the base id/site
        synth_site = f"{base_site}_SYN{tag}"
        synth_id = f"{synth_site}_{target_ply}"

        rec_out = {
            "id": synth_id,
            "site": synth_site,
            "ply": target_ply,
            "board_fen": board_fen,
            "pockets_white": pw,
            "pockets_black": pb,
            "text_static": " ".join(static + pocket),
            "text_dynamic": " ".join(dynamic),
            "text_all": " ".join(dynamic + dynamic + dynamic + pocket + pocket + static),
            # optional debug:
            "synth_from": args.query_id,
            "offset_from_query": off,
            "engine_bestmove": bestmove,
            "engine_pv": pv[:10],
            "engine_cp": cp,
            "engine_mate": mate,
        }

        synth_docs.append(rec_out)

    engine.close()

    # Write new corpus: original + synth appended
    args.corpus_out.parent.mkdir(parents=True, exist_ok=True)
    with args.corpus_out.open("w", encoding="utf-8") as f:
        for d in corpus_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
        for d in synth_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("WROTE:", args.corpus_out)
    print("Injected synth docs:", len(synth_docs))
    for d in synth_docs:
        print(" ", d["id"])

if __name__ == "__main__":
    main()


