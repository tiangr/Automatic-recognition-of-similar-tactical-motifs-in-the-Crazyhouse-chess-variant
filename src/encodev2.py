"""
encodev2.py
-----------
Dynamic token encoder for Crazyhouse tactical positions.

v2 → v3 enrichments (mentor-informed, April 2026):
  - Mating piece identity  (dyn:mating_piece:X)
  - Drop-mate variant      (dyn:mateDrop:X)
  - Consecutive capture pairs (dyn:capPair:XY) — Tier-3 importance
  - Per-piece capture totals  (dyn:capPieceTotal:XN)
  - Sacrifice detection    (dyn:hasSacrifice, pv:sacrifice, pv:dropSacrifice)
  - Opponent-captures flag (dyn:oppCaptures / dyn:noOppCaptures)
  - Drop proximity to enemy king (dyn:dropNearKing / dyn:dropFarKing)
  - sum:mating_piece:X in summary block

v3 → v4 enrichments (feature_descriptions_2_.md alignment, April 2026):
  - Pawn promotion detection  (dyn:hasPromotion, pv:promotion, pv:promotionPiece:X)
    → maps to DG_!F_p+ (Tier-4, CART Depth-2, permutation rank 3)
  - Piece-moved tokens        (pv:pieceMoved:X) for every move
    → maps to DS_!piece_moved_X (permutation rank 17)
  - Attack-relation tokens    (pv:attack:X>Y) for new attacks per move
    → maps to DS_!attack_X>Y (permutation rank 14)
  - Mating piece PAIR tokens  (dyn:matingPiecePair:XY)
    → maps to DG_!mating_piece_XY (LogReg top-10, e.g. double-bishop mate)
  - NAMING FIX: matingPiece → mating_piece (underscore, matches MD column names)

Feature importance tiers (from mentor .md files):
  Tier 1: checkmate agreement         (dyn:hasMate → DG_!F_checkmate)
  Tier 2: move-sequence similarity    (computed at pair-level in training)
  Tier 3: captured-piece agreement    (dyn:capPair, dyn:capPieceTotal)
  Tier 4: promotion, sacrifice, mating-piece agreement
"""

import chess
import chess.variant
from collections import Counter

PIECE_LETTERS = {
    chess.PAWN:   "P",
    chess.KNIGHT: "N",
    chess.BISHOP: "B",
    chess.ROOK:   "R",
    chess.QUEEN:  "Q",
    chess.KING:   "K",
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


def _chebyshev(sq1: int, sq2: int) -> int:
    """Chebyshev (king-move) distance between two squares."""
    r1, c1 = divmod(sq1, 8)
    r2, c2 = divmod(sq2, 8)
    return max(abs(r1 - r2), abs(c1 - c2))


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

    fen = rec.get("fen")
    if fen:
        try:
            board.set_fen(fen)
            return board
        except Exception:
            pass

    prefix = rec.get("uci_moves_prefix")
    if isinstance(prefix, list):
        try:
            for u in prefix:
                board.push(board.parse_uci(u))
            return board
        except Exception as e:
            raise ValueError(f"prefix_fail:{type(e).__name__}")

    uci_moves = rec.get("uci_moves")
    ply = rec.get("ply")

    if isinstance(uci_moves, list):
        try:
            if isinstance(ply, int) and ply > 0 and len(uci_moves) == ply - 1:
                prefix = uci_moves
            elif isinstance(ply, int) and ply > 0 and len(uci_moves) >= ply - 1:
                prefix = uci_moves[:ply - 1]
            else:
                prefix = uci_moves

            for u in prefix:
                board.push(board.parse_uci(u))
            return board
        except Exception as e:
            raise ValueError(f"uci_replay_fail:{type(e).__name__}")

    board_fen = rec.get("board_fen")
    if board_fen:
        try:
            board.set_board_fen(board_fen)
            return board
        except Exception as e:
            raise ValueError(f"board_fen_fail:{type(e).__name__}")

    raise ValueError("missing_startpos")


def _get_new_attacks(board_before: chess.Board, board_after: chess.Board) -> list[str]:
    """
    Return pv:attack:X>Y tokens for attack relations that are NEW after a move
    (present after but not before). Coordinate-independent: piece types only.

    Matches the paper's DS_!attack_X>Y encoding (e.g. DS_!attack_Q>r).
    Only attacker vs attacked colour combinations where attacker attacks attacked.
    """
    def _attack_pairs(b: chess.Board) -> set[tuple[str, str]]:
        pairs = set()
        for sq, piece in b.piece_map().items():
            for att_sq in b.attackers(not piece.color, sq):
                att_piece = b.piece_at(att_sq)
                if att_piece:
                    pairs.add((_piece_letter(att_piece.piece_type), _piece_letter(piece.piece_type)))
        return pairs

    before_pairs = _attack_pairs(board_before)
    after_pairs  = _attack_pairs(board_after)
    new_pairs    = after_pairs - before_pairs

    return [f"pv:attack:{a}>{b}" for a, b in new_pairs]


def encode_dynamic_v2(rec: dict) -> list[str]:
    """
    Dynamic tokens by replaying PV on CrazyhouseBoard starting from the event position.

    Full feature set (v4):
      - Mating piece identity + pairs (dyn:mating_piece:X, dyn:matingPiecePair:XY)
      - Drop-mate variant (dyn:mateDrop:X)
      - Consecutive capture pairs (dyn:capPair:XY)
      - Per-piece capture totals (dyn:capPieceTotal:XN)
      - Sacrifice detection
      - Opponent-captures flag
      - Drop proximity to enemy king
      - Pawn promotion (dyn:hasPromotion, pv:promotion, pv:promotionPiece:X) [NEW v4]
      - Piece-moved tokens (pv:pieceMoved:X) [NEW v4]
      - Attack-relation tokens (pv:attack:X>Y) [NEW v4]
      - Mating piece pair tokens (dyn:matingPiecePair:XY) [NEW v4]
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

    # ── Aggregate flags ──────────────────────────────────────────────────
    has_drop       = False
    has_capture    = False
    has_check      = False
    has_mate       = False
    has_drop_check = False
    has_sacrifice  = False
    has_promotion  = False   # NEW v4
    opp_captures   = False

    drop_count  = 0
    cap_count   = 0
    move_count  = 0
    drop_piece_set = set()

    pocket_gain  = {"P": 0, "N": 0, "B": 0, "R": 0, "Q": 0}
    pocket_spend = {"P": 0, "N": 0, "B": 0, "R": 0, "Q": 0}

    first_action = None

    # For consecutive-capture pairs (Tier-3)
    cap_sequence: list[str] = []
    # For mating piece pairs (LogReg top-10, e.g. !mating_piece_bb)
    mating_pieces: list[str] = []

    solver_color = board.turn   # color to move at the start of the PV

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

        mover_is_solver = (board.turn == solver_color)

        # ── Capture detection ────────────────────────────────────────────
        cap = board.is_capture(move)
        cap_piece = None
        if cap:
            has_capture = True
            if mover_is_solver:
                cap_count += 1
            else:
                opp_captures = True

            if board.is_en_passant(move):
                cap_piece = "P"
            else:
                captured = board.piece_at(move.to_square)
                if captured:
                    cap_piece = _piece_letter(captured.piece_type)

            if cap_piece:
                cap_sequence.append(cap_piece)

        # ── Drop detection ───────────────────────────────────────────────
        is_drop = "@" in uci
        drop_piece = None
        if is_drop:
            has_drop = True
            drop_count += 1
            drop_piece = uci.split("@")[0].upper()
            drop_piece_set.add(drop_piece)
            pocket_spend[drop_piece] = pocket_spend.get(drop_piece, 0) + 1

        # ── Sacrifice detection: solver moves to square attacked by opponent ──
        is_sacrifice = False
        if mover_is_solver:
            attacked_by_opp = board.is_attacked_by(not solver_color, move.to_square)
            if attacked_by_opp and not cap:
                is_sacrifice = True
                has_sacrifice = True
            elif is_drop and attacked_by_opp:
                is_sacrifice = True
                has_sacrifice = True

        # ── Drop proximity to enemy king ─────────────────────────────────
        if is_drop and mover_is_solver:
            enemy_king_sq = board.king(not solver_color)
            if enemy_king_sq is not None:
                dist = _chebyshev(move.to_square, enemy_king_sq)
                if dist <= 2:
                    tokens.append("dyn:dropNearKing")
                else:
                    tokens.append("dyn:dropFarKing")

        # ── Piece-moved token (NEW v4) ───────────────────────────────────
        # Records which piece type is moving — maps to DS_!piece_moved_X
        if not is_drop:
            moving_piece_obj = board.piece_at(move.from_square)
            if moving_piece_obj:
                tokens.append(f"pv:pieceMoved:{_piece_letter(moving_piece_obj.piece_type)}")
        else:
            tokens.append(f"pv:pieceMoved:{drop_piece}")

        # ── Promotion detection (NEW v4) ─────────────────────────────────
        # Maps to DG_!F_p+ (Tier-4, CART depth-2, permutation rank 3)
        is_promotion = (move.promotion is not None)
        promo_piece = None
        if is_promotion:
            has_promotion = True
            promo_piece = _piece_letter(move.promotion)

        # ── Save board state before push for attack diff ─────────────────
        board_before_push = board.copy()

        # ── Push the move ────────────────────────────────────────────────
        board.push(move)
        move_count += 1

        # ── Attack-relation tokens (NEW v4) ──────────────────────────────
        # Records new X>Y attacks arising after this move — maps to DS_!attack_X>Y
        tokens.extend(_get_new_attacks(board_before_push, board))

        if cap and cap_piece and cap_piece != "K" and mover_is_solver:
            pocket_gain[cap_piece] = pocket_gain.get(cap_piece, 0) + 1

        is_check     = board.is_check()
        is_checkmate = board.is_checkmate()

        if is_check:
            has_check = True
        if is_checkmate:
            has_mate = True

        # ── Mating piece identity (naming aligned to MD: mating_piece) ───
        if is_checkmate:
            if is_drop and drop_piece:
                mating_pieces.append(drop_piece)
                tokens.append(f"dyn:mating_piece:{drop_piece}")
                tokens.append(f"dyn:mateDrop:{drop_piece}")
            else:
                moved_piece = board.piece_at(move.to_square)
                if moved_piece:
                    mp = _piece_letter(moved_piece.piece_type)
                    mating_pieces.append(mp)
                    tokens.append(f"dyn:mating_piece:{mp}")

        if first_action is None:
            if is_drop and is_check:
                first_action = "dropCheck"
            elif is_drop:
                first_action = "drop"
            elif cap and is_check:
                first_action = "capCheck"
            elif cap:
                first_action = "capture"
            elif is_checkmate:
                first_action = "mate"
            elif is_check:
                first_action = "check"

        # ── Per-move tokens ──────────────────────────────────────────────
        if is_drop and drop_piece:
            tokens.append("pv:drop")
            tokens.append(f"pv:dropPiece:{drop_piece}")
            sq = uci.split("@")[1]
            tokens.append(f"pv:dropSq:{sq}")
            if is_check:
                tokens.append("pv:dropCheck")
                has_drop_check = True
            if is_sacrifice:
                tokens.append("pv:dropSacrifice")

        if cap:
            tokens.append("pv:capture")
            if cap_piece:
                tokens.append(f"pv:capturePiece:{cap_piece}")
            if is_check:
                tokens.append("pv:captureCheck")

        if is_sacrifice and not is_drop:
            tokens.append("pv:sacrifice")
            moved_piece = board.piece_at(move.to_square)
            if moved_piece:
                tokens.append(f"pv:sacrificePiece:{_piece_letter(moved_piece.piece_type)}")

        # ── Promotion per-move tokens (NEW v4) ───────────────────────────
        if is_promotion and promo_piece:
            tokens.append("pv:promotion")
            tokens.append(f"pv:promotionPiece:{promo_piece}")

        if is_checkmate:
            tokens.append("pv:mate")
            break

    # ── Consecutive capture-pair tokens (Tier-3: ordering of captures) ───
    for i in range(len(cap_sequence) - 1):
        pair = cap_sequence[i] + cap_sequence[i + 1]
        tokens.append(f"dyn:capPair:{pair}")

    # ── Per-piece capture counts ─────────────────────────────────────────
    for piece, cnt in Counter(cap_sequence).items():
        tokens.append(f"dyn:capPieceTotal:{piece}{cnt}")

    # ── Mating piece PAIR tokens (NEW v4) ────────────────────────────────
    # e.g. dyn:matingPiecePair:bb = two bishops cooperate in checkmate
    # maps to DG_!mating_piece_XY (LogReg top-10 rank 9)
    for i in range(len(mating_pieces) - 1):
        pair = mating_pieces[i] + mating_pieces[i + 1]
        tokens.append(f"dyn:matingPiecePair:{pair}")

    # ── Summary flags ────────────────────────────────────────────────────
    tokens.append("dyn:hasDrop"    if has_drop    else "dyn:noDrop")
    tokens.append("dyn:hasCapture" if has_capture else "dyn:noCapture")
    if has_drop_check:
        tokens.append("dyn:hasDropCheck")
    if has_check:
        tokens.append("dyn:hasCheck")
    if has_mate:
        tokens.append("dyn:hasMate")
    if has_sacrifice:
        tokens.append("dyn:hasSacrifice")
    if has_promotion:                        # NEW v4
        tokens.append("dyn:hasPromotion")
    if opp_captures:
        tokens.append("dyn:oppCaptures")
    else:
        tokens.append("dyn:noOppCaptures")

    if first_action:
        tokens.append(f"dyn:first:{first_action}")

    def bucket(n: int) -> str:
        if n == 0: return "0"
        if n == 1: return "1"
        if n == 2: return "2"
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

    # ── Sum/summary tokens (repeated in text_all for higher weight) ───────
    if has_drop:
        tokens.append("sum:hasDrop")
    if has_capture:
        tokens.append("sum:hasCapture")
    if has_check:
        tokens.append("sum:hasCheck")
    if has_mate:
        tokens.append("sum:hasMate")
    if has_sacrifice:
        tokens.append("sum:hasSacrifice")
    if has_promotion:                        # NEW v4
        tokens.append("sum:hasPromotion")
    if opp_captures:
        tokens.append("sum:oppCaptures")
    if first_action:
        tokens.append(f"sum:first:{first_action}")
    for mp in mating_pieces:
        tokens.append(f"sum:mating_piece:{mp}")   # FIXED: underscore naming

    for p, c in pocket_gain.items():
        if c:
            tokens.append(f"sum:pocketGain:{p}{c}")
    for p, c in pocket_spend.items():
        if c:
            tokens.append(f"sum:pocketSpend:{p}{c}")

    return tokens


# ---------------------------------------------------------------------------
# encode_corpus_fields — called by build_corpus_full.py
# ---------------------------------------------------------------------------
def encode_corpus_fields(rec: dict, static_toks: list[str], pocket_toks: list[str]) -> dict:
    """
    Build the five text fields stored in the corpus JSONL.

    text_dynamic_general  — whole-solution flags (sum: + dyn: tokens)
    text_dynamic_solution — per-move events (pv: tokens)
    text_dynamic          — gen + sol combined
    text_static           — board piece placement + pocket tokens
    text_all              — weighted: gen×3 + sol×2 + pocket×2 + static×1
    """
    dyn_tokens = encode_dynamic_v2(rec)

    general_toks  = [t for t in dyn_tokens if t.startswith(("dyn:", "sum:"))]
    solution_toks = [t for t in dyn_tokens if t.startswith("pv:")]

    text_dyn_gen = " ".join(general_toks)
    text_dyn_sol = " ".join(solution_toks)
    text_dyn     = " ".join(dyn_tokens)
    text_static  = " ".join(static_toks + pocket_toks)

    # Weighted text_all: repeat high-signal tokens
    text_all = " ".join(
        general_toks * 3 +
        solution_toks * 2 +
        pocket_toks * 2 +
        static_toks
    )

    return {
        "text_dynamic_general":  text_dyn_gen,
        "text_dynamic_solution": text_dyn_sol,
        "text_dynamic":          text_dyn,
        "text_static":           text_static,
        "text_all":              text_all,
    }