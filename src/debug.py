import json
import chess.variant

r = json.loads(open("data/derived/tactics_1k_tactical.jsonl", encoding="utf-8").readline())

board = chess.variant.CrazyhouseBoard()
for u in r["uci_moves"][:r["ply"]-1]:
    board.push(board.parse_uci(u))

print("turn:", "white" if board.turn else "black")
print("legal c7c6?", board.is_legal(board.parse_uci("c7c6")))
print("fen now:", board.fen())
print("played:", r["played_move"], "best:", r["bestmove_before"])
