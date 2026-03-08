import chess

def encode_static(board_fen: str) -> list[str]:
    # board_fen = only pieces on board (no turn/castling/etc)
    tokens = []
    board = chess.Board(fen=board_fen + " w - - 0 1")  # hack: use normal board for parsing squares
    # Note: this ignores crazyhouse legality; we only want piece placement tokens.
    for sq, piece in board.piece_map().items():
        color = "w" if piece.color == chess.WHITE else "b"
        p = piece.symbol().lower()  # p,n,b,r,q,k
        tokens.append(f"{color}{p}@{chess.square_name(sq)}")
    return tokens

def encode_pockets(pw: dict, pb: dict) -> list[str]:
    # pw/pb like {"P":2,"N":1}
    tokens = []
    for side, pocket in [("w", pw), ("b", pb)]:
        for piece, cnt in pocket.items():
            for _ in range(int(cnt)):
                tokens.append(f"{side}pocket:{piece}")
    return tokens

def encode_dynamic(rec: dict) -> list[str]:
    # Use pv_before or pv_prev if present
    pv = rec.get("pv_before") or rec.get("pv_prev") or []
    tokens = []
    # drop / capture / checkmate approximations via UCI strings
    for mv in pv[:6]:
        if "@" in mv:
            tokens.append("has:drop")
            # piece letter before @ if like N@d4
            tokens.append(f"drop:{mv.split('@')[0].upper()}")
        if len(mv) >= 4 and mv[1].isdigit() and mv[3].isdigit():
            # normal move like e2e4, can't detect capture w/out board
            tokens.append("has:move")
    # include bestmove token if available
    bm = rec.get("bestmove_before") or rec.get("best_prev")
    if bm and bm != "(none)":
        tokens.append(f"best:{bm}")
        if "@" in bm:
            tokens.append("best:drop")
    # include delta bucket
    d = rec.get("delta")
    if isinstance(d, (int, float)):
        if d >= 10000:
            tokens.append("delta:mate")
        elif d >= 800:
            tokens.append("delta:800+")
        elif d >= 600:
            tokens.append("delta:600+")
        elif d >= 400:
            tokens.append("delta:400+")
        else:
            tokens.append("delta:lt400")
    return tokens
