import chess
import chess.variant

PIECE_LETTERS = {
    chess.PAWN: "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK: "R",
    chess.QUEEN: "Q",
    chess.KING: "K",
}


def normalize_uci(u: str) -> str | None:
    if not u:
        return None
    u = u.strip()
    u = u.replace("+", "").replace("#", "")

    if u in ("--", "..."):
        return None

    return u


def _piece_letter(piece_type: int) -> str:
    return PIECE_LETTERS.get(piece_type, "?")


def _reconstruct_event_board(rec: dict):
    """
    Reconstruct the event position = position BEFORE played_move.

    Preferred sources:
    1) full crazyhouse fen
    2) explicit uci_moves_prefix
    3) uci_moves interpreted safely
    4) board_fen as a last-resort board-only fallback
    """
    board = chess.variant.CrazyhouseBoard()

    # Best: full crazyhouse FEN already contains board, turn, pockets, etc.
    fen = rec.get("fen")
    if fen:
        try:
            board.set_fen(fen)
            return board
        except Exception:
            pass

    # Next best: explicit normalized prefix from augment_tactics.py
    prefix = rec.get("uci_moves_prefix")
    if isinstance(prefix, list):
        try:
            for u in prefix:
                board.push(board.parse_uci(u))
            return board
        except Exception as e:
            raise ValueError(f"prefix_fail:{type(e).__name__}")

    # Fallback: infer from uci_moves
    uci_moves = rec.get("uci_moves")
    ply = rec.get("ply")

    if isinstance(uci_moves, list):
        try:
            # If uci_moves is already an event prefix, use it as-is
            if isinstance(ply, int) and ply > 0 and len(uci_moves) == ply - 1:
                prefix = uci_moves
            # If uci_moves looks like a longer game move list, trim to event prefix
            elif isinstance(ply, int) and ply > 0 and len(uci_moves) >= ply - 1:
                prefix = uci_moves[:ply - 1]
            else:
                prefix = uci_moves

            for u in prefix:
                board.push(board.parse_uci(u))
            return board
        except Exception as e:
            raise ValueError(f"uci_replay_fail:{type(e).__name__}")

    # Last resort: board only, without pockets/history
    board_fen = rec.get("board_fen")
    if board_fen:
        try:
            board.set_board_fen(board_fen)
            return board
        except Exception as e:
            raise ValueError(f"board_fen_fail:{type(e).__name__}")

    raise ValueError("missing_startpos")


def encode_dynamic_v2(rec: dict) -> list[str]:
    """
    Dynamic tokens by replaying PV on CrazyhouseBoard starting from the event position.
    """
    try:
        board = _reconstruct_event_board(rec)
    except Exception as e:
        return [f"dyn:start_fail:{str(e)}"]

    pv = rec.get("pv_before") or rec.get("pv_prev") or []
    bm = rec.get("bestmove_before") or rec.get("best_prev") or ""

    pv_line = list(pv)
    if bm and bm != "(none)":
        if not pv_line or pv_line[0] != bm:
            pv_line = [bm] + pv_line

    tokens = []

    has_drop = False
    has_capture = False
    has_check = False
    has_mate = False
    has_drop_check = False

    drop_count = 0
    cap_count = 0
    move_count = 0
    drop_piece_set = set()

    pocket_gain = {"P": 0, "N": 0, "B": 0, "R": 0, "Q": 0}
    pocket_spend = {"P": 0, "N": 0, "B": 0, "R": 0, "Q": 0}

    first_action = None

    N = 10
    for raw in pv_line[:N]:
        uci = normalize_uci(raw)
        if not uci:
            tokens.append("dyn:pv_skip")
            continue

        try:
            move = board.parse_uci(uci)
        except Exception:
            tokens.append(f"dyn:pv_parse_fail:{uci}")
            break

        cap = board.is_capture(move)
        cap_piece = None
        if cap:
            has_capture = True
            cap_count += 1
            if board.is_en_passant(move):
                cap_piece = "P"
            else:
                captured = board.piece_at(move.to_square)
                if captured:
                    cap_piece = _piece_letter(captured.piece_type)

        is_drop = "@" in uci
        drop_piece = None
        if is_drop:
            has_drop = True
            drop_count += 1
            drop_piece = uci.split("@")[0].upper()
            drop_piece_set.add(drop_piece)
            pocket_spend[drop_piece] = pocket_spend.get(drop_piece, 0) + 1

        board.push(move)
        move_count += 1

        if cap and cap_piece and cap_piece != "K":
            pocket_gain[cap_piece] = pocket_gain.get(cap_piece, 0) + 1

        if board.is_check():
            has_check = True
        if board.is_checkmate():
            has_mate = True

        if first_action is None:
            if is_drop and board.is_check():
                first_action = "dropCheck"
            elif is_drop:
                first_action = "drop"
            elif cap and board.is_check():
                first_action = "capCheck"
            elif cap:
                first_action = "capture"
            elif board.is_checkmate():
                first_action = "mate"
            elif board.is_check():
                first_action = "check"

        if is_drop and drop_piece:
            tokens.append("pv:drop")
            tokens.append(f"pv:dropPiece:{drop_piece}")
            sq = uci.split("@")[1]
            tokens.append(f"pv:dropSq:{sq}")
            if board.is_check():
                tokens.append("pv:dropCheck")
                has_drop_check = True

        if cap:
            tokens.append("pv:capture")
            if cap_piece:
                tokens.append(f"pv:capturePiece:{cap_piece}")
            if board.is_check():
                tokens.append("pv:captureCheck")

        if board.is_checkmate():
            tokens.append("pv:mate")
            break

    tokens.append("dyn:hasDrop" if has_drop else "dyn:noDrop")
    tokens.append("dyn:hasCapture" if has_capture else "dyn:noCapture")
    if has_drop_check:
        tokens.append("dyn:hasDropCheck")
    if has_check:
        tokens.append("dyn:hasCheck")
    if has_mate:
        tokens.append("dyn:hasMate")

    if first_action:
        tokens.append(f"dyn:first:{first_action}")

    def bucket(n: int) -> str:
        if n == 0:
            return "0"
        if n == 1:
            return "1"
        if n == 2:
            return "2"
        return "3plus"

    tokens.append(f"dyn:drops:{bucket(drop_count)}")
    tokens.append(f"dyn:captures:{bucket(cap_count)}")

    if move_count <= 3:
        tokens.append("dyn:pvLen:short")
    elif move_count <= 7:
        tokens.append("dyn:pvLen:med")
    else:
        tokens.append("dyn:pvLen:long")

    if drop_piece_set:
        tokens.append("dyn:dropPieces:" + "".join(sorted(drop_piece_set)))

    if has_drop:
        tokens.append("sum:hasDrop")
    if has_capture:
        tokens.append("sum:hasCapture")
    if has_check:
        tokens.append("sum:hasCheck")
    if has_mate:
        tokens.append("sum:hasMate")
    if first_action:
        tokens.append(f"sum:first:{first_action}")

    for p, c in pocket_gain.items():
        if c:
            tokens.append(f"sum:pocketGain:{p}{c}")
    for p, c in pocket_spend.items():
        if c:
            tokens.append(f"sum:pocketSpend:{p}{c}")

    return tokens