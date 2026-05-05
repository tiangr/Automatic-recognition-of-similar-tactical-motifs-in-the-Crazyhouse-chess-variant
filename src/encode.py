"""
encode.py
---------
Static and pocket encoders for Crazyhouse tactical positions.

v1: piece placement tokens only (wQ@e1, etc.)

v2 enrichments (feature_descriptions_summary_2_.md alignment, April 2026):
  - Connectivity tokens: B>pg7 (attack), R<Kg1 (defense), Q=pe6 (x-ray)
    → maps to white_attacks, white_defends, white_xrays (and black equivalents)
    → static feature category, section 3.3 of the MD spec
  - Pawn structure tokens (static_pawns field):
    I/i=isolated, F/f=passed, 'F/'f=protected passed, L/l=backward,
    S{}/s{}=doubled, W[]/w[]=pawn chain, P(n)/p(n)=pawn islands
    → maps to white_isolated, white_passed, etc. (section 3.4 of MD spec)

NOTE: encode_static() now returns BOTH placement tokens AND connectivity/pawn
tokens for the text_static field used by BM25 and by extract_features_crazyhouse.
The dynamic encoder (encodev2.py) is unchanged.
"""

import chess
import chess.variant


# ---------------------------------------------------------------------------
# Static encoder — piece placement + connectivity + pawn structure
# ---------------------------------------------------------------------------

def encode_static(board_fen: str, include_connectivity: bool = True) -> list[str]:
    """
    Encode the static board position into tokens.

    Tokens produced:
      Placement:    w{p}@{sq}  / b{p}@{sq}   (e.g. wQ@e1, bp@e5)
      Connectivity: {P}>{p}{sq}               (attack, e.g. B>pg7)
                    {P}<{P}{sq}               (defense, e.g. R<Kg1)
                    {P}={p}{sq}               (x-ray, e.g. Q=pe6)
      Pawn structure: from encode_pawn_structure() — included via text_static
                      when build_corpus calls encode_pawn_structure() separately.

    Connectivity uses the same notation as the Bizjak (2020) paper and the MD spec.
    Uppercase = White piece, lowercase = Black piece.
    """
    tokens = []

    # Parse using standard board (crazyhouse legality ignored for static tokens)
    try:
        board = chess.Board(fen=board_fen + " w - - 0 1")
    except Exception:
        # Fallback: try parsing as-is
        try:
            board = chess.Board(fen=board_fen)
        except Exception:
            return tokens

    piece_map = board.piece_map()

    # ── Piece placement tokens ────────────────────────────────────────────
    for sq, piece in piece_map.items():
        color = "w" if piece.color == chess.WHITE else "b"
        p = piece.symbol().lower()  # p, n, b, r, q, k
        tokens.append(f"{color}{p}@{chess.square_name(sq)}")

    if not include_connectivity:
        return tokens

    # ── Connectivity tokens ───────────────────────────────────────────────
    # For each piece, record its attack/defense/x-ray relations.
    # Attack (>): piece attacks an enemy piece
    # Defense (<): piece defends a friendly piece
    # X-ray (=): piece x-rays through another piece to attack/defend behind it
    #
    # Notation: the acting piece's symbol (case = color) + relation + target piece symbol + square
    # e.g. B>pg7 = White Bishop attacks black pawn on g7
    #      R<Kg1 = White Rook defends White King on g1
    #      Q=pe6 = White Queen x-rays black pawn on e6

    for sq, piece in piece_map.items():
        acting_sym = piece.symbol()  # uppercase=White, lowercase=Black
        acting_color = piece.color

        # Squares this piece attacks
        attacked_squares = board.attacks(sq)

        for target_sq in attacked_squares:
            target_piece = piece_map.get(target_sq)
            if target_piece is None:
                continue

            target_sym = target_piece.symbol()  # uppercase=White, lowercase=Black
            sq_name = chess.square_name(target_sq)

            if target_piece.color != acting_color:
                # Attack relation: acting piece attacks enemy piece
                tokens.append(f"{acting_sym}>{target_sym}{sq_name}")
            else:
                # Defense relation: acting piece defends friendly piece
                tokens.append(f"{acting_sym}<{target_sym}{sq_name}")

    # ── X-ray tokens ─────────────────────────────────────────────────────
    # X-ray: a sliding piece attacks a square through an intervening piece.
    # We detect this by checking if a sliding piece would attack a square
    # if the intervening pieces were removed.
    sliding_types = {chess.QUEEN, chess.ROOK, chess.BISHOP}

    for sq, piece in piece_map.items():
        if piece.piece_type not in sliding_types:
            continue

        acting_sym   = piece.symbol()
        acting_color = piece.color

        # Get the ray directions for this piece type
        directions = []
        if piece.piece_type in (chess.ROOK, chess.QUEEN):
            directions += [(1, 0), (-1, 0), (0, 1), (0, -1)]  # horizontal/vertical
        if piece.piece_type in (chess.BISHOP, chess.QUEEN):
            directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]  # diagonal

        rank, file_ = divmod(sq, 8)

        for dr, df in directions:
            r, f = rank + dr, file_ + df
            found_first = False   # have we passed through at least one piece?

            while 0 <= r < 8 and 0 <= f < 8:
                target_sq = r * 8 + f
                target_piece = piece_map.get(target_sq)

                if target_piece is not None:
                    if found_first:
                        # This is the x-rayed piece (behind the first blocker)
                        target_sym = target_piece.symbol()
                        sq_name = chess.square_name(target_sq)
                        tokens.append(f"{acting_sym}={target_sym}{sq_name}")
                        break
                    else:
                        found_first = True
                        # Continue past the first piece to find the x-ray target

                r += dr
                f += df

    return tokens


# ---------------------------------------------------------------------------
# Pawn structure encoder — produces static_pawns-style tokens
# ---------------------------------------------------------------------------

def encode_pawn_structure(board_fen: str) -> list[str]:
    """
    Encode pawn structure into tokens matching the static_pawns field format
    described in the MD spec (section 3.4, Bizjak 2020 notation):

      I{sq} / i{sq}      = isolated pawn (White/Black)
      F{sq} / f{sq}      = passed pawn
      'F{sq} / 'f{sq}    = protected passed pawn
      L{sq} / l{sq}      = backward pawn
      S{sq} / s{sq}      = doubled pawn
      W[sq/sq/...] / w[] = pawn chain (diagonally linked)
      P(n) / p(n)        = pawn island count

    These are extracted by extract_features_crazyhouse.py to produce
    white_isolated, white_passed, white_protected_passed, white_backward,
    white_doubled, white_pawn_chain, white_pawn_islands (and black equiv.).
    """
    tokens = []

    try:
        board = chess.Board(fen=board_fen + " w - - 0 1")
    except Exception:
        try:
            board = chess.Board(fen=board_fen)
        except Exception:
            return tokens

    piece_map = board.piece_map()

    white_pawns = {}   # sq -> chess.square_name(sq)
    black_pawns = {}

    for sq, piece in piece_map.items():
        if piece.piece_type == chess.PAWN:
            if piece.color == chess.WHITE:
                white_pawns[sq] = chess.square_name(sq)
            else:
                black_pawns[sq] = chess.square_name(sq)

    def _file(sq): return chess.square_file(sq)
    def _rank(sq): return chess.square_rank(sq)

    def _pawn_structure_tokens(pawns: dict, opp_pawns: dict,
                                sym_isolated, sym_passed, sym_prot_passed,
                                sym_backward, sym_doubled, sym_chain, sym_island,
                                white_side: bool) -> list[str]:
        toks = []
        if not pawns:
            toks.append(f"{sym_island}(0)")
            return toks

        files = {_file(sq) for sq in pawns}
        opp_files = {_file(sq) for sq in opp_pawns}
        opp_by_file = {}
        for sq in opp_pawns:
            opp_by_file.setdefault(_file(sq), []).append(_rank(sq))

        pawn_by_file = {}
        for sq in pawns:
            pawn_by_file.setdefault(_file(sq), []).append(sq)

        for sq, sq_name in sorted(pawns.items(), key=lambda x: chess.square_name(x[0])):
            f = _file(sq)
            r = _rank(sq)

            # Isolated: no friendly pawn on adjacent files
            adj_files = {f - 1, f + 1}
            isolated = not (adj_files & files)
            if isolated:
                toks.append(f"{sym_isolated}{sq_name}")

            # Passed: no enemy pawn on same or adjacent files ahead of this pawn
            ahead_ranks = range(r + 1, 8) if white_side else range(0, r)
            blocked = False
            for af in [f - 1, f, f + 1]:
                if af in opp_by_file:
                    for opp_r in opp_by_file[af]:
                        if (white_side and opp_r > r) or (not white_side and opp_r < r):
                            blocked = True
                            break
                if blocked:
                    break

            if not blocked:
                # Check if protected passed (defended by another friendly pawn)
                protected = False
                if white_side:
                    defenders = [(f - 1, r - 1), (f + 1, r - 1)]
                else:
                    defenders = [(f - 1, r + 1), (f + 1, r + 1)]
                for df, dr in defenders:
                    if 0 <= df < 8 and 0 <= dr < 8:
                        def_sq = chess.square(df, dr)
                        if def_sq in pawns:
                            protected = True
                            break
                if protected:
                    toks.append(f"'{sym_passed}{sq_name}")
                else:
                    toks.append(f"{sym_passed}{sq_name}")

            # Backward pawn: cannot advance safely, no friendly pawn supports it
            if white_side:
                support_sqs = [(f - 1, r - 1), (f + 1, r - 1)]
                advance_sq  = (f, r + 1)
            else:
                support_sqs = [(f - 1, r + 1), (f + 1, r + 1)]
                advance_sq  = (f, r - 1)
            has_support = any(
                0 <= df < 8 and 0 <= dr < 8 and chess.square(df, dr) in pawns
                for df, dr in support_sqs
            )
            if not has_support and not isolated:
                toks.append(f"{sym_backward}{sq_name}")

        # Doubled pawns: more than one pawn on same file
        for f_idx, sqs in pawn_by_file.items():
            if len(sqs) > 1:
                for sq in sqs:
                    toks.append(f"{sym_doubled}{chess.square_name(sq)}")

        # Pawn chains: diagonally adjacent friendly pawns
        pawn_set = set(pawns.keys())
        chain_members = set()
        for sq in pawn_set:
            f, r = _file(sq), _rank(sq)
            if white_side:
                # White pawn chains go forward-right or forward-left
                candidates = [chess.square(f + 1, r + 1) if f < 7 and r < 7 else None,
                              chess.square(f - 1, r + 1) if f > 0 and r < 7 else None]
            else:
                candidates = [chess.square(f + 1, r - 1) if f < 7 and r > 0 else None,
                              chess.square(f - 1, r - 1) if f > 0 and r > 0 else None]
            for cand in candidates:
                if cand is not None and cand in pawn_set:
                    chain_members.add(sq)
                    chain_members.add(cand)

        if chain_members:
            chain_names = "/".join(chess.square_name(sq) for sq in sorted(chain_members))
            toks.append(f"{sym_chain}[{chain_names}]")

        # Pawn islands: groups of pawns separated by empty files
        sorted_files = sorted(files)
        islands = 1
        for i in range(1, len(sorted_files)):
            if sorted_files[i] - sorted_files[i - 1] > 1:
                islands += 1
        toks.append(f"{sym_island}({islands})")

        return toks

    tokens.extend(_pawn_structure_tokens(
        white_pawns, black_pawns,
        "I", "F", "F", "L", "S", "W", "P", white_side=True
    ))
    tokens.extend(_pawn_structure_tokens(
        black_pawns, white_pawns,
        "i", "f", "f", "l", "s", "w", "p", white_side=False
    ))

    return tokens


# ---------------------------------------------------------------------------
# Pocket encoder (unchanged)
# ---------------------------------------------------------------------------

def encode_pockets(pw: dict, pb: dict) -> list[str]:
    """Encode pocket pieces. pw/pb like {"P": 2, "N": 1}"""
    tokens = []
    for side, pocket in [("w", pw), ("b", pb)]:
        for piece, cnt in pocket.items():
            for _ in range(int(cnt)):
                tokens.append(f"{side}pocket:{piece}")
    return tokens


# ---------------------------------------------------------------------------
# Legacy dynamic encoder (kept for compatibility; use encodev2.py instead)
# ---------------------------------------------------------------------------

def encode_dynamic(rec: dict) -> list[str]:
    """
    Legacy dynamic encoder. Very approximate — does not replay the board.
    Use encodev2.encode_dynamic_v2() for accurate dynamic tokens.
    """
    pv = rec.get("pv_before") or rec.get("pv_prev") or []
    tokens = []
    for mv in pv[:6]:
        if "@" in mv:
            tokens.append("has:drop")
            tokens.append(f"drop:{mv.split('@')[0].upper()}")
        if len(mv) >= 4 and mv[1].isdigit() and mv[3].isdigit():
            tokens.append("has:move")
    bm = rec.get("bestmove_before") or rec.get("best_prev")
    if bm and bm != "(none)":
        tokens.append(f"best:{bm}")
        if "@" in bm:
            tokens.append("best:drop")
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