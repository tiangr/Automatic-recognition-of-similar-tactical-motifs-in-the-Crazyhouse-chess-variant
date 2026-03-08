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

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


# ----------------------------
# helpers
# ----------------------------

def _piece_letter(piece_type: int) -> str:
    return PIECE_LETTERS.get(piece_type, "?")

def _piece_color(piece: chess.Piece) -> str:
    return "w" if piece.color == chess.WHITE else "b"

def _bucket(n: int) -> str:
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    return "3plus"

def normalize_uci(u: str) -> str | None:
    if not u:
        return None
    u = u.strip().replace("+", "").replace("#", "")
    if u in ("--", "..."):
        return None
    return u

def _is_board_only_fen(s: str) -> bool:
    return isinstance(s, str) and " " not in s and "/" in s

def _material_balance_for_color(board: chess.Board, color: bool) -> int:
    """
    Positive if 'color' has more material on board, negative otherwise.
    """
    score = 0
    for _, piece in board.piece_map().items():
        val = PIECE_VALUES[piece.piece_type]
        if piece.color == color:
            score += val
        else:
            score -= val
    return score


# ----------------------------
# static features
# ----------------------------

def encode_static(board_fen: str) -> list[str]:
    """
    Static encoding for the board position only.

    Input:
      board_fen = board layout only, e.g. 'rnbqkbnr/pppppppp/8/...'

    Output token groups:
      - exact piece positions
      - reachable squares
      - attack/defense connectivity
      - simple pawn structure
    """
    tokens = []

    # board_fen is board-only placement, so add minimal FEN suffix
    board = chess.Board(fen=board_fen + " w - - 0 1")

    piece_map = board.piece_map()

    # 1) exact piece positions
    for sq, piece in piece_map.items():
        c = _piece_color(piece)
        p = _piece_letter(piece.piece_type)
        sqn = chess.square_name(sq)
        tokens.append(f"pos:{c}{p}@{sqn}")

    # 2) reachable squares / attacks from current placement
    for sq, piece in piece_map.items():
        c = _piece_color(piece)
        p = _piece_letter(piece.piece_type)
        from_sq = chess.square_name(sq)

        for to_sq in board.attacks(sq):
            to_name = chess.square_name(to_sq)
            tokens.append(f"reach:{c}{p}>{to_name}")
            tokens.append(f"reachFrom:{c}{p}@{from_sq}>{to_name}")

    # 3) piece connectivity: attacks / defenses
    for from_sq, attacker in piece_map.items():
        ac = _piece_color(attacker)
        ap = _piece_letter(attacker.piece_type)
        from_name = chess.square_name(from_sq)

        for to_sq in board.attacks(from_sq):
            target = piece_map.get(to_sq)
            if target is None:
                continue

            tc = _piece_color(target)
            tp = _piece_letter(target.piece_type)
            to_name = chess.square_name(to_sq)

            if attacker.color == target.color:
                tokens.append(f"def:{ac}{ap}>{tp}")
                tokens.append(f"defSq:{ac}{ap}@{from_name}>{tp}@{to_name}")
            else:
                tokens.append(f"att:{ac}{ap}>{tp}")
                tokens.append(f"attSq:{ac}{ap}@{from_name}>{tp}@{to_name}")

    # 4) simple pawn structure
    pawns_w = board.pieces(chess.PAWN, chess.WHITE)
    pawns_b = board.pieces(chess.PAWN, chess.BLACK)

    tokens.extend(_encode_pawn_structure(pawns_w, "w"))
    tokens.extend(_encode_pawn_structure(pawns_b, "b"))

    return tokens


def _encode_pawn_structure(pawn_squares, color_name: str) -> list[str]:
    out = []
    files = {}

    for sq in pawn_squares:
        f = chess.square_file(sq)
        files.setdefault(f, []).append(sq)

    # doubled pawns
    for f, sqs in files.items():
        if len(sqs) >= 2:
            out.append(f"pawn:{color_name}:doubled:file{f}")

    for sq in pawn_squares:
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        sqn = chess.square_name(sq)

        # isolated pawn
        has_left = (f - 1) in files
        has_right = (f + 1) in files
        if not has_left and not has_right:
            out.append(f"pawn:{color_name}:isolated")
            out.append(f"pawnSq:{color_name}:isolated@{sqn}")

        # protected pawn / chain
        supporters = []
        if color_name == "w":
            if f > 0 and r > 0:
                supporters.append(chess.square(f - 1, r - 1))
            if f < 7 and r > 0:
                supporters.append(chess.square(f + 1, r - 1))
        else:
            if f > 0 and r < 7:
                supporters.append(chess.square(f - 1, r + 1))
            if f < 7 and r < 7:
                supporters.append(chess.square(f + 1, r + 1))

        supported = any(s in pawn_squares for s in supporters)
        if supported:
            out.append(f"pawn:{color_name}:chain")
            out.append(f"pawnSq:{color_name}:chain@{sqn}")

    return out


def encode_pockets(pw: dict, pb: dict) -> list[str]:
    """
    Crazyhouse pocket encoding.
    """
    tokens = []
    for side, pocket in [("w", pw or {}), ("b", pb or {})]:
        for piece, cnt in sorted(pocket.items()):
            cnt = int(cnt)
            for _ in range(cnt):
                tokens.append(f"{side}pocket:{piece}")
            if cnt > 0:
                tokens.append(f"{side}pocketCount:{piece}{cnt}")
    return tokens


# ----------------------------
# dynamic features
# ----------------------------

def encode_dynamic(rec: dict) -> list[str]:
    """
    Dynamic encoding by replaying the PV on a CrazyhouseBoard.

    Keeps:
      - pvpat2 / pvpat3
      - mateBy / mateByDrop / mateByMove
      - mateZone
      - firstReply

    But without over-boosting them.
    """
    board = _load_crazyhouse_start_position(rec)
    if isinstance(board, list):
        return board

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
    has_sacrifice = False
    has_drop_check = False

    drop_count = 0
    cap_count = 0
    move_count = 0

    drop_piece_set = set()
    moved_piece_types = []
    captured_piece_types = []
    interaction_types = []
    mate_piece_types = set()
    sacrifice_piece_types = set()

    pocket_gain = {"P": 0, "N": 0, "B": 0, "R": 0, "Q": 0}
    pocket_spend = {"P": 0, "N": 0, "B": 0, "R": 0, "Q": 0}

    first_action = None
    prev_move_type = None
    move_labels = []
    first_reply = None

    N = 10
    for idx, raw in enumerate(pv_line[:N]):
        uci = normalize_uci(raw)
        if not uci:
            continue

        try:
            move = board.parse_uci(uci)
        except Exception:
            tokens.append(f"dyn:pv_parse_fail:{uci}")
            break

        is_drop = "@" in uci
        mover_type = None
        cap_piece_type = None
        move_to_square = None
        drop_square_name = None

        if is_drop:
            drop_piece = uci.split("@")[0].upper()
            drop_square_name = uci.split("@")[1]
            mover_type = drop_piece
            has_drop = True
            drop_count += 1
            drop_piece_set.add(drop_piece)
            pocket_spend[drop_piece] = pocket_spend.get(drop_piece, 0) + 1
            move_type = "drop"
        else:
            mover_piece = board.piece_at(move.from_square)
            if mover_piece:
                mover_type = _piece_letter(mover_piece.piece_type)
            move_to_square = move.to_square
            move_type = "move"

        cap = board.is_capture(move)
        if cap:
            has_capture = True
            cap_count += 1
            if board.is_en_passant(move):
                cap_piece_type = "P"
            else:
                captured = board.piece_at(move.to_square)
                if captured:
                    cap_piece_type = _piece_letter(captured.piece_type)
            move_type = "capture" if not is_drop else "dropCapture"

        mover_color = board.turn
        before_balance = _material_balance_for_color(board, mover_color)
        board.push(move)
        after_balance = _material_balance_for_color(board, mover_color)
        move_count += 1

        if cap and cap_piece_type and cap_piece_type != "K":
            pocket_gain[cap_piece_type] = pocket_gain.get(cap_piece_type, 0) + 1

        in_check = board.is_check()
        is_mate = board.is_checkmate()

        if in_check:
            has_check = True
            move_type = move_type + "Check"
            tokens.append("dyn:general:check")

        if is_drop and in_check:
            has_drop_check = True

        if is_mate:
            has_mate = True
            tokens.append("dyn:general:mate")

        if first_action is None:
            first_action = move_type

        if mover_type:
            moved_piece_types.append(mover_type)
            tokens.append(f"dyn:move:{mover_type}")

        if cap_piece_type:
            captured_piece_types.append(cap_piece_type)
            tokens.append(f"dyn:capture:{cap_piece_type}")

        if mover_type and cap_piece_type:
            interaction_types.append((mover_type, cap_piece_type))
            tokens.append(f"dyn:interact:{mover_type}>{cap_piece_type}")

        if not cap and after_balance < before_balance:
            has_sacrifice = True
            if mover_type:
                sacrifice_piece_types.add(mover_type)
                tokens.append(f"dyn:sac:{mover_type}")

        if prev_move_type is not None:
            tokens.append(f"dyn:seq:{prev_move_type}>{move_type}")
        prev_move_type = move_type

        label = "drop" if is_drop else ("capture" if cap else "move")
        if in_check:
            label += "Check"
        if is_mate:
            label += "Mate"
        move_labels.append(label)

        if idx == 1:
            first_reply = label
            tokens.append(f"dyn:firstReply:{label}")

        if is_mate and mover_type:
            tokens.append(f"dyn:mateBy:{mover_type}")
            if is_drop:
                tokens.append(f"dyn:mateByDrop:{mover_type}")
            else:
                tokens.append(f"dyn:mateByMove:{mover_type}")

            mate_zone = None
            if is_drop and drop_square_name:
                file_char = drop_square_name[0]
                if file_char in "abc":
                    mate_zone = "queenside"
                elif file_char in "fgh":
                    mate_zone = "kingside"
                else:
                    mate_zone = "center"
            elif move_to_square is not None:
                file_idx = chess.square_file(move_to_square)
                if file_idx <= 2:
                    mate_zone = "queenside"
                elif file_idx >= 5:
                    mate_zone = "kingside"
                else:
                    mate_zone = "center"

            if mate_zone:
                tokens.append(f"dyn:mateZone:{mate_zone}")

        if is_mate:
            for p in _mate_contributing_piece_types(board):
                mate_piece_types.add(p)
                tokens.append(f"dyn:matePiece:{p}")
            break

    if len(move_labels) >= 2:
        tokens.append(f"pvpat2:{move_labels[0]}>{move_labels[1]}")

    if len(move_labels) >= 3:
        tokens.append(f"pvpat3:{move_labels[0]}>{move_labels[1]}>{move_labels[2]}")

    tokens.append("dyn:hasDrop" if has_drop else "dyn:noDrop")
    tokens.append("dyn:hasCapture" if has_capture else "dyn:noCapture")
    if has_drop_check:
        tokens.append("dyn:hasDropCheck")
    if has_check:
        tokens.append("dyn:hasCheck")
    if has_mate:
        tokens.append("dyn:hasMate")
    if has_sacrifice:
        tokens.append("dyn:hasSac")

    if first_action:
        tokens.append(f"dyn:first:{first_action}")

    tokens.append(f"dyn:drops:{_bucket(drop_count)}")
    tokens.append(f"dyn:captures:{_bucket(cap_count)}")

    if move_count <= 3:
        tokens.append("dyn:pvLen:short")
    elif move_count <= 7:
        tokens.append("dyn:pvLen:med")
    else:
        tokens.append("dyn:pvLen:long")

    if drop_piece_set:
        tokens.append("dyn:dropPieces:" + "".join(sorted(drop_piece_set)))

    for p in sorted(set(moved_piece_types)):
        tokens.append(f"sum:move:{p}")
    for p in sorted(set(captured_piece_types)):
        tokens.append(f"sum:capture:{p}")
    for a, b in sorted(set(interaction_types)):
        tokens.append(f"sum:interact:{a}>{b}")
    for p in sorted(sacrifice_piece_types):
        tokens.append(f"sum:sac:{p}")
    for p in sorted(mate_piece_types):
        tokens.append(f"sum:matePiece:{p}")

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
    if first_reply:
        tokens.append(f"sum:firstReply:{first_reply}")

    for p, c in pocket_gain.items():
        if c:
            tokens.append(f"sum:pocketGain:{p}{c}")
    for p, c in pocket_spend.items():
        if c:
            tokens.append(f"sum:pocketSpend:{p}{c}")

    return tokens


def _load_crazyhouse_start_position(rec: dict):
    """
    Returns CrazyhouseBoard or list[str] of error tokens.
    """
    # Best option for your dataset: reconstruct from move prefix
    uci_moves = rec.get("uci_moves") or rec.get("uci_moves_prefix")
    ply = rec.get("ply")

    if isinstance(uci_moves, list) and isinstance(ply, int) and ply > 0:
        board = chess.variant.CrazyhouseBoard()
        try:
            for u in uci_moves[:ply - 1]:
                board.push(board.parse_uci(u))
            return board
        except Exception as e:
            return [f"dyn:prefix_fail:{type(e).__name__}"]

    # Fallback: full crazyhouse FEN if present
    fen = rec.get("fen")
    if isinstance(fen, str) and " " in fen:
        board = chess.variant.CrazyhouseBoard()
        try:
            board.set_fen(fen)
            return board
        except Exception as e:
            return [f"dyn:fen_fail:{type(e).__name__}"]

    # board_fen in your corpus is usually board-only, so cannot be used
    board_fen = rec.get("board_fen")
    if _is_board_only_fen(board_fen):
        return ["dyn:missing_prefix_for_board_only_fen"]

    return ["dyn:missing_startpos"]


def _mate_contributing_piece_types(board: chess.Board) -> set[str]:
    """
    After mate, collect attacker piece types attacking the king square
    or adjacent king-zone squares.
    """
    out = set()
    defender = board.turn
    attacker = not defender
    ksq = board.king(defender)
    if ksq is None:
        return out

    zone = {ksq}
    kf = chess.square_file(ksq)
    kr = chess.square_rank(ksq)

    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            nf, nr = kf + df, kr + dr
            if 0 <= nf <= 7 and 0 <= nr <= 7:
                zone.add(chess.square(nf, nr))

    for sq, piece in board.piece_map().items():
        if piece.color != attacker:
            continue
        attacked = board.attacks(sq)
        if any(z in attacked for z in zone):
            out.add(_piece_letter(piece.piece_type))

    return out


# ----------------------------
# optional convenience wrapper
# ----------------------------

def encode_record(rec: dict) -> dict:
    """
    Convenience wrapper if you want one call per record.
    """
    board_fen = rec.get("board_fen")
    pw = rec.get("pockets_white", {})
    pb = rec.get("pockets_black", {})

    static = encode_static(board_fen) if board_fen else []
    pockets = encode_pockets(pw, pb)
    dynamic = encode_dynamic(rec)

    return {
        "static_tokens": static,
        "pocket_tokens": pockets,
        "dynamic_tokens": dynamic,
        "text_static": " ".join(static + pockets),
        "text_dynamic": " ".join(dynamic),
        "text_all": " ".join(dynamic + dynamic + dynamic + dynamic + pockets + static),
    }