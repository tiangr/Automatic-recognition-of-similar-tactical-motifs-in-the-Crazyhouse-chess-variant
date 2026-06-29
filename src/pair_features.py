"""
pair_features.py
================
Single source of truth for query<->candidate pairwise features.

Used in TWO places so the representation can never drift:
  1. train_retrieval_supervised.ipynb  -> builds the training matrix from
     exported expert-eval rows.
  2. app.py (retrieval path)           -> scores a WIDE candidate pool so the
     supervised model can change which documents are actually retrieved,
     not merely reorder the 10 already displayed.

Everything here is computed from fields that exist BOTH in the export rows
AND at serve time (static/dynamic token strings, solution UCI, mate, turn,
rating, pockets). No engine call, no feature-cache lookup required -> the
exact same function is trivially callable on the Pi.

Token families
--------------
A query/candidate is described by space-separated tokens. We bucket every
token into one family and compute set-overlap per family. The per-family
overlaps are what let the model say "dynamic-line overlap matters 3x more
than raw placement overlap" -- which maps directly back to how the BM25
query should be weighted (see learned_bm25_weights() in the notebook).

Static families:
  placement   wp@e4 / bk@g8 ...        (where pieces sit)
  relation    r<ph7 / k>Nd6 / q=Nd6    (attack/defend/relation graph)
  motif       Ia2 Lc2 Sc3 W[..] P(3)   (derived tactical descriptors)
  pocket      wpocket:Q / bpocket:N    (drop material available)
Dynamic families:
  pv          pv:attack:Q>K ...        (concrete move-by-move of the line)
  dyn         dyn:mating_piece:R ...   (aggregate properties of the line)
  sum         sum:hasDrop ...          (summary flags)
"""

from __future__ import annotations
import re
from collections import Counter

# ---------------------------------------------------------------------------
# Token family classification
# ---------------------------------------------------------------------------

_PLACEMENT_RE = re.compile(r"^[wbWB][pnbrqkPNBRQK]@")

# fixed family order -> stable column order for the feature matrix
STATIC_FAMILIES  = ["placement", "relation", "motif", "pocket"]
# mate = final mating-picture tokens, km = king-march/forcing-sequence tokens.
# Split out of "dyn" so retrieval can weight them independently (they were the
# focus of the picture/march work and have their own predictive strength that
# a shared dyn weight can't express).
DYNAMIC_FAMILIES = ["pv", "dyn", "sum", "mate", "km"]
ALL_FAMILIES     = STATIC_FAMILIES + DYNAMIC_FAMILIES


def _family_of(tok: str) -> str | None:
    if not tok:
        return None
    if tok.startswith("mate:"):
        return "mate"
    if tok.startswith("km:"):
        return "km"
    if tok.startswith("pv:"):
        return "pv"
    if tok.startswith("dyn:"):
        return "dyn"
    if tok.startswith("sum:"):
        return "sum"
    if "pocket:" in tok:
        return "pocket"
    if _PLACEMENT_RE.match(tok):
        return "placement"
    # static relation tokens carry an attack/defend/relation operator
    if any(op in tok for op in ("<", ">", "=")):
        return "relation"
    # leftover static descriptors: I.. L.. S.. W[..] P(..) and lowercase peers
    return "motif"


def bucket_tokens(text: str) -> dict[str, set]:
    """Split a token string into {family: set(tokens)}."""
    out = {f: set() for f in ALL_FAMILIES}
    if not text:
        return out
    for tok in text.split():
        fam = _family_of(tok)
        if fam is not None:
            out[fam].add(tok)
    return out


# ---------------------------------------------------------------------------
# Set / sequence similarity primitives
# ---------------------------------------------------------------------------

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def _overlap_coef(a: set, b: set) -> float:
    """|A n B| / min(|A|,|B|) -- robust when the sets differ a lot in size."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _levenshtein_sim(a, b) -> float:
    a = list(a or []); b = list(b or [])
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    prev = list(range(lb + 1))
    for ca in a:
        curr = [prev[0] + 1] + [0] * lb
        for j, cb in enumerate(b):
            curr[j + 1] = min(prev[j + 1] + 1, curr[j] + 1,
                              prev[j] + (0 if ca == cb else 1))
        prev = curr
    return 1.0 - prev[lb] / max(la, lb)


def _multiset_jaccard(a: list, b: list) -> float:
    ca, cb = Counter(a), Counter(b)
    keys = set(ca) | set(cb)
    if not keys:
        return 0.5  # both pockets empty -> neutral, matches app.py convention
    inter = sum(min(ca[k], cb[k]) for k in keys)
    union = sum(max(ca[k], cb[k]) for k in keys)
    return inter / union if union else 0.0


# ---------------------------------------------------------------------------
# Token-derived structured signals (mating piece, first action, drops)
# ---------------------------------------------------------------------------

def _mating_piece(toks: set) -> str:
    for t in toks:
        if t.startswith("dyn:mating_piece:"):
            return t.split(":", 2)[2].upper()
        if t.startswith("sum:mating_piece:"):
            return t.split(":", 2)[2].upper()
    return ""


def _first_action(toks: set) -> str:
    for t in toks:
        if t.startswith("dyn:first:"):
            return t.split("dyn:first:")[1]
    return ""


def _drop_pieces(toks: set) -> set:
    for t in toks:
        if t.startswith("dyn:dropPieces:"):
            return set(t.split("dyn:dropPieces:")[1].upper())
    return {t.split("pv:dropPiece:")[1].upper()
            for t in toks if t.startswith("pv:dropPiece:")}


def _captured_pieces(toks: set) -> list:
    """Piece types captured during the line, colour-folded (multiset)."""
    return [t.split(":", 2)[2].upper()
            for t in toks if t.startswith("pv:capturePiece:")]


def _capture_pairs(toks: set) -> set:
    """Consecutive capture pairs, colour-folded (e.g. 'BN')."""
    return {t.split(":", 2)[2].upper()
            for t in toks if t.startswith("dyn:capPair:")}


def _attack_relations(toks: set) -> set:
    """Piece-to-piece attack relations during the line, colour-folded
    (e.g. 'Q>K' = a queen attacks a king at some point in the solution)."""
    return {t.split("pv:attack:")[1].upper()
            for t in toks if t.startswith("pv:attack:")}


def _moved_pieces(toks: set) -> set:
    """Piece types that move during the line, colour-folded."""
    return {t.split("pv:pieceMoved:")[1].upper()
            for t in toks if t.startswith("pv:pieceMoved:")}


def _has_tok(toks: set, *prefixes) -> bool:
    return any(t == p or t.startswith(p) for t in toks for p in prefixes)


def _pocket_multiset(pocket) -> list:
    if not isinstance(pocket, dict):
        return []
    items = []
    for piece, cnt in pocket.items():
        try:
            items.extend([str(piece).upper()] * max(0, int(cnt or 0)))
        except (TypeError, ValueError):
            continue
    return sorted(items)


def _pockets_from_fen(fen: str):
    """Pull (white_list, black_list) drop material out of a crazyhouse FEN's
    [..] pocket segment. White = uppercase, black = lowercase."""
    if not fen or "[" not in fen or "]" not in fen:
        return [], []
    seg = fen[fen.index("[") + 1: fen.index("]")]
    w = sorted(c.upper() for c in seg if c.isupper())
    b = sorted(c.upper() for c in seg if c.islower())
    return w, b


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Mechanism-level features: describe HOW the mate is engineered (the sequence
# of tactical actions and how the king is trapped) rather than which piece
# delivers it or where it lands. Color- and location-invariant by design.
# ---------------------------------------------------------------------------

_FILES = "abcdefgh"

def _sq_to_fr(sq: str):
    if not sq or len(sq) < 2:
        return None
    f = _FILES.find(sq[0])
    try:
        r = int(sq[1]) - 1
    except ValueError:
        return None
    return (f, r) if (f >= 0 and 0 <= r <= 7) else None


def _uci_dest(uci: str) -> str:
    u = str(uci)
    if "@" in u:                       # drop move, e.g. "Q@h7"
        return u.split("@", 1)[1][:2]
    return u[2:4] if len(u) >= 4 else ""


def _expand_placement(placement: str) -> dict:
    board = {}
    rows = placement.split("/")[:8]
    for ri, row in enumerate(rows):
        rank = 7 - ri                  # FEN lists rank 8 first
        f = 0
        for ch in row:
            if ch.isdigit():
                f += int(ch)
            else:
                board[(f, rank)] = ch
                f += 1
    return board


def _mated_king_sq(fen: str, turn):
    """The king being mated = opponent of the side to move."""
    placement = (fen or "").split(" ")[0].split("[")[0]
    if not placement:
        return None
    board = _expand_placement(placement)
    t = (turn or "").lower()
    if not t:
        parts = (fen or "").split(" ")
        t = parts[1] if len(parts) > 1 else "w"
    mated = "k" if t.startswith("w") else "K"
    for sq, pc in board.items():
        if pc == mated:
            return sq
    return None


try:
    import chess
    import chess.variant
    _HAS_CHESS = True
except Exception:
    _HAS_CHESS = False

# Shared piece-symbol table for the mate-picture / king-march sections below.
_PT = {chess.PAWN: "P", chess.KNIGHT: "N", chess.BISHOP: "B",
       chess.ROOK: "R", chess.QUEEN: "Q", chess.KING: "K"} if _HAS_CHESS else {}

# =========================================================================
# Mating picture (matna slika) -- folded final-position king-cage, merged
# in from the former mate_picture.py module (now retired to keep file count
# down). Colour+orientation-fold the 3x3 box around the mated king so the
# same motif (e.g. smothered mate) collides to one token regardless of
# corner/colour. See pair_features()'s mp_* block below for usage.
# =========================================================================
_MP_EMPTY = {"picture_canon": "", "region": "", "checker": "",
             "cage_own": "", "net_enemy": "", "smother": False, "tokens": []}

def _mp_replay(board, solution_uci):
    """Push solution moves onto board in place; stop at first illegal move.
    (Used only by the mate_picture section below; king_march uses its own
    single-pass _core_from_start instead, since it needs per-ply detail.)"""
    for u in [str(x) for x in (solution_uci or [])]:
        try:
            mv = chess.Move.from_uci(u)
            if mv not in board.legal_moves and len(u) == 4:
                mv2 = chess.Move.from_uci(u + "q")     # promo stored without suffix
                if mv2 in board.legal_moves:
                    mv = mv2
            if mv not in board.legal_moves:
                break
            board.push(mv)
        except Exception:
            break
    return board

def _rot90(g):
    return [[g[2 - c][r] for c in range(3)] for r in range(3)]

def _mirror(g):
    return [list(reversed(row)) for row in g]

def _canon(grid):
    forms, g = [], grid
    for _ in range(4):
        forms.append("/".join("".join(r) for r in g))
        forms.append("/".join("".join(r) for r in _mirror(g)))
        g = _rot90(g)
    return min(forms)

def _picture_from_board(b) -> dict:
    if b is None or not b.is_checkmate():
        return dict(_MP_EMPTY)
    mated = b.turn
    ksq = b.king(mated)
    if ksq is None:
        return dict(_MP_EMPTY)
    kf, kr = chess.square_file(ksq), chess.square_rank(ksq)

    grid = [["#"] * 3 for _ in range(3)]
    own_adj, enemy_adj, empty_adj = [], [], []
    for dr in (1, 0, -1):
        for df in (-1, 0, 1):
            gr, gc = 1 - dr, df + 1
            f, r = kf + df, kr + dr
            if not (0 <= f <= 7 and 0 <= r <= 7):
                continue
            if df == 0 and dr == 0:
                grid[gr][gc] = "K"; continue
            sq = chess.square(f, r)
            pc = b.piece_at(sq)
            if pc is None:
                grid[gr][gc] = "."; empty_adj.append(sq)
            elif pc.color == mated:
                L = _PT[pc.piece_type]; grid[gr][gc] = L; own_adj.append(L)
            else:
                L = _PT[pc.piece_type].lower(); grid[gr][gc] = L; enemy_adj.append(L.upper())

    picture_canon = _canon(grid)
    walls = sum(row.count("#") for row in grid)
    region = "corner" if walls == 5 else "edge" if walls == 3 else "center"

    checkers = sorted({_PT[b.piece_at(s).piece_type] for s in b.checkers() if b.piece_at(s)})
    checker = "".join(checkers)
    cage_own = "".join(sorted(own_adj))

    net = set(enemy_adj)
    for sq in empty_adj:
        for asq in b.attackers(not mated, sq):
            apc = b.piece_at(asq)
            if apc:
                net.add(_PT[apc.piece_type])
    net |= set(checkers)
    net_enemy = "".join(sorted(net))

    smother = (checker == "N" and not empty_adj and not enemy_adj
               and len(own_adj) == (8 - walls))

    tokens = [
        f"mate:pic:{picture_canon}",
        f"mate:region:{region}",
        f"mate:checker:{checker}" if checker else "mate:checker:?",
        f"mate:cage:{cage_own}" if cage_own else "mate:cage:none",
        f"mate:net:{net_enemy}" if net_enemy else "mate:net:none",
    ]
    if smother:
        tokens.append("mate:smother")

    return {"picture_canon": picture_canon, "region": region, "checker": checker,
            "cage_own": cage_own, "net_enemy": net_enemy, "smother": smother,
            "tokens": tokens}

def mate_picture(fen, solution_uci=None, turn=None) -> dict:
    """Serve path: build from a (crazyhouse) FEN + solution and read the mate."""
    if not _HAS_CHESS:
        return dict(_MP_EMPTY)
    bfen = _to_bracket_fen(fen, turn)
    if not bfen:
        return dict(_MP_EMPTY)
    try:
        b = chess.variant.CrazyhouseBoard(bfen)
    except Exception:
        return dict(_MP_EMPTY)
    return _picture_from_board(_mp_replay(b, solution_uci))

def mate_picture_for_rec(rec: dict) -> dict:
    """Corpus path: reuse the dynamic encoder's event reconstruction (handles
    full fen / prefix / uci+ply / board_fen incl. pockets), then replay the
    solution and read the mate. Falls back to the FEN entry point."""
    if not _HAS_CHESS:
        return dict(_MP_EMPTY)
    try:
        from encodev2 import _reconstruct_event_board
        board = _reconstruct_event_board(rec)
        sol = rec.get("solution_uci") or rec.get("pv_before") or rec.get("pv_prev") or []
        return _picture_from_board(_mp_replay(board, sol))
    except Exception:
        return mate_picture(rec.get("fen", "") or rec.get("board_fen", ""),
                            rec.get("solution_uci"), rec.get("turn"))

def picture_from_tokens(text_dynamic: str) -> dict:
    out = {"picture_canon": "", "region": "", "checker": "",
           "cage_own": "", "net_enemy": "", "smother": False}
    found = False
    for t in (text_dynamic or "").split():
        if not t.startswith("mate:"):
            continue
        found = True
        if   t.startswith("mate:pic:"):     out["picture_canon"] = t[len("mate:pic:"):]
        elif t.startswith("mate:region:"):  out["region"]   = t[len("mate:region:"):]
        elif t.startswith("mate:checker:"): out["checker"]  = t[len("mate:checker:"):].replace("?", "")
        elif t.startswith("mate:cage:"):    out["cage_own"] = t[len("mate:cage:"):].replace("none", "")
        elif t.startswith("mate:net:"):     out["net_enemy"]= t[len("mate:net:"):].replace("none", "")
        elif t == "mate:smother":           out["smother"]  = True
    return out if found else {}

def get_picture(rec: dict) -> dict:
    """Prefer parsing from stored tokens (no replay); else replay fen+solution."""
    p = picture_from_tokens(rec.get("text_dynamic", ""))
    if p:
        return p
    return mate_picture(rec.get("fen", ""), rec.get("solution_uci"), rec.get("turn"))

def _ms_jac(a: str, b: str) -> float:
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    keys = set(ca) | set(cb)
    if not keys:
        return 0.5
    inter = sum(min(ca[k], cb[k]) for k in keys)
    union = sum(max(ca[k], cb[k]) for k in keys)
    return inter / union if union else 0.0

def compare_pictures(qp: dict, cp: dict) -> dict:
    have = bool(qp) and bool(cp) and qp.get("picture_canon") and cp.get("picture_canon")
    if not have:
        return {"mp_pic_match": 0.5, "mp_pic_cellsim": 0.5, "mp_region_match": 0.5,
                "mp_checker_match": 0.5, "mp_smother_match": 0.5, "mp_both_smother": 0.0,
                "mp_cage_sim": 0.5, "mp_net_sim": 0.5}
    a, b = qp["picture_canon"], cp["picture_canon"]
    f = {"mp_pic_match": 1.0 if a == b else 0.0}
    f["mp_pic_cellsim"] = (sum(1 for x, y in zip(a, b) if x == y) / len(a)
                           if (len(a) == len(b) and a) else 0.0)
    f["mp_region_match"] = 1.0 if qp.get("region") == cp.get("region") else 0.0
    qc, cc = qp.get("checker", ""), cp.get("checker", "")
    f["mp_checker_match"] = (1.0 if qc and cc and qc == cc else 0.0 if qc and cc else 0.5)
    qs, cs = bool(qp.get("smother")), bool(cp.get("smother"))
    f["mp_smother_match"] = 1.0 if qs == cs else 0.0
    f["mp_both_smother"]  = 1.0 if (qs and cs) else 0.0
    f["mp_cage_sim"] = _ms_jac(qp.get("cage_own", ""), cp.get("cage_own", ""))
    f["mp_net_sim"]  = _ms_jac(qp.get("net_enemy", ""), cp.get("net_enemy", ""))
    return f

def is_smother(fen, solution_uci=None, turn=None) -> bool:
    """Convenience flag for the diagnostic script."""
    return bool(mate_picture(fen, solution_uci, turn).get("smother"))

MP_FEATURE_NAMES = ["mp_pic_match", "mp_pic_cellsim", "mp_region_match",
                    "mp_checker_match", "mp_smother_match", "mp_both_smother",
                    "mp_cage_sim", "mp_net_sim"]


# =========================================================================
# King march (forcing-sequence features) -- merged in from the former
# king_march.py module. Encodes the PATH the king is forced along: which
# line/square each check comes from, whether the same piece/square checks
# repeatedly, whether the mating move itself is a drop vs a moved piece,
# and which figures (attacker net vs defender's own cage) cooperate around
# the final mated-king square. Complements the mate picture (final
# position only) rather than replacing it.
# =========================================================================
_KM_EMPTY = {"king_path": [], "king_displacement": 0, "forced_king_moves": 0,
             "check_lines": [], "same_line_run": 0, "checker_seq": [],
             "repeat_checker_run": 0, "same_square_checker_run": 0,
             "mate_is_drop": False, "cooperating_types": "", "tokens": []}

def _check_line(king_sq, from_sq):
    """Coarse geometric relation of a checking piece's square to the king."""
    kf, kr = chess.square_file(king_sq), chess.square_rank(king_sq)
    ff, fr = chess.square_file(from_sq), chess.square_rank(from_sq)
    df, dr = ff - kf, fr - kr
    if df == 0 and dr == 0:
        return "other"
    if max(abs(df), abs(dr)) == 1:
        return "adjacent"
    if df == 0:
        return "file"
    if dr == 0:
        return "rank"
    if abs(df) == abs(dr):
        return "diag"
    if (abs(df), abs(dr)) in ((1, 2), (2, 1)):
        return "knight"
    return "other"

def _longest_run(seq):
    if not seq:
        return 0
    best = cur = 1
    for i in range(1, len(seq)):
        cur = cur + 1 if seq[i] == seq[i - 1] else 1
        best = max(best, cur)
    return best

def _core_from_start(board_start, pushed, mating_side, mated_side):
    """Single forward replay from the TRUE pre-line position (no pop/copy
    tricks, which desync Crazyhouse pockets). board_start is mutated in place
    by pushing; caller must pass a board it's OK to consume."""
    out = dict(_KM_EMPTY)
    if not pushed:
        return out

    b = board_start
    king_path = [b.king(mated_side)]
    check_lines, checker_seq, from_sqs = [], [], []
    mate_is_drop = False
    board_final = None

    for i, mv in enumerate(pushed):
        mover = b.turn
        is_last = (i == len(pushed) - 1)
        b.push(mv)
        if mover == mated_side:
            ksq = b.king(mated_side)
            if ksq is not None and ksq != king_path[-1]:
                king_path.append(ksq)
        if mover == mating_side and b.is_check():
            ksq = b.king(mated_side)
            if ksq is not None:
                check_lines.append(_check_line(ksq, mv.to_square))
            pc = b.piece_at(mv.to_square)
            if pc is not None:
                checker_seq.append(_PT.get(pc.piece_type, "?"))
            from_sqs.append(mv.to_square)
        if is_last and mover == mating_side and mv.drop is not None:
            mate_is_drop = True
        if is_last:
            board_final = b   # final position, post all pushes

    out["king_path"] = king_path
    if len(king_path) >= 2:
        kf0, kr0 = chess.square_file(king_path[0]), chess.square_rank(king_path[0])
        kf1, kr1 = chess.square_file(king_path[-1]), chess.square_rank(king_path[-1])
        out["king_displacement"] = max(abs(kf1 - kf0), abs(kr1 - kr0))
    out["forced_king_moves"] = len(king_path) - 1
    out["check_lines"] = check_lines
    out["same_line_run"] = _longest_run(check_lines)
    out["checker_seq"] = checker_seq
    out["repeat_checker_run"] = _longest_run(checker_seq)
    out["mate_is_drop"] = mate_is_drop
    out["same_square_checker_run"] = _longest_run(from_sqs)

    # cooperating pieces: every checker TYPE (mating side) plus every piece
    # of EITHER side adjacent to the final mated-king square -- the mating
    # side's pieces forming the net, AND the mated side's own pieces forming
    # the cage (e.g. the smother's own rook+pawns). Colour-folded by role:
    # 'A:' prefix for attacker-adjacent types, 'D:' prefix for defender's own
    # adjacent types, so a smother (all-defender cage) and a net-mate
    # (all-attacker net) remain distinguishable rather than collapsing into
    # one undifferentiated multiset.
    atk_adj, def_adj = set(), set()
    ksq = king_path[-1]
    kf, kr = chess.square_file(ksq), chess.square_rank(ksq)
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            f, r = kf + df, kr + dr
            if 0 <= f <= 7 and 0 <= r <= 7:
                pc = board_final.piece_at(chess.square(f, r))
                if pc is None:
                    continue
                t = _PT.get(pc.piece_type, "?")
                if pc.color == mating_side:
                    atk_adj.add(t)
                else:
                    def_adj.add(t)
    atk_all = set(checker_seq) | atk_adj
    out["cooperating_types"] = "A" + "".join(sorted(atk_all)) + "D" + "".join(sorted(def_adj))

    out["tokens"] = [
        f"km:line:{'-'.join(check_lines)}" if check_lines else "km:line:none",
        f"km:checkers:{'-'.join(checker_seq)}" if checker_seq else "km:checkers:none",
        f"km:disp:{out['king_displacement']}",
        f"km:sameline:{out['same_line_run']}",
        f"km:samepiece:{out['repeat_checker_run']}",
        f"km:samesquare:{out['same_square_checker_run']}",
        f"km:coop:{out['cooperating_types']}" if out["cooperating_types"] else "km:coop:none",
    ]
    if mate_is_drop:
        out["tokens"].append("km:matebydrop")
    return out

def king_march(fen, solution_uci=None, turn=None) -> dict:
    if not _HAS_CHESS:
        return dict(_KM_EMPTY)
    bfen = _to_bracket_fen(fen, turn)
    if not bfen:
        return dict(_KM_EMPTY)
    try:
        board = chess.variant.CrazyhouseBoard(bfen)
    except Exception:
        return dict(_KM_EMPTY)
    mating_side = board.turn
    mated_side = not mating_side
    uci = [str(u) for u in (solution_uci or [])]
    pushed = []
    for u in uci:
        try:
            mv = chess.Move.from_uci(u)
            if mv not in board.legal_moves and len(u) == 4:
                mv2 = chess.Move.from_uci(u + "q")
                if mv2 in board.legal_moves:
                    mv = mv2
            if mv not in board.legal_moves:
                break
            pushed.append(mv)
        except Exception:
            break
    # validate the FULL line leads to checkmate before committing to the
    # real replay (cheap dry run on a throwaway copy)
    check_board = board.copy()
    ok = True
    for mv in pushed:
        if mv not in check_board.legal_moves:
            ok = False
            break
        check_board.push(mv)
    if not ok or not pushed or not check_board.is_checkmate():
        return dict(_KM_EMPTY)
    return _core_from_start(board, pushed, mating_side, mated_side)

def king_march_for_rec(rec: dict) -> dict:
    if not _HAS_CHESS:
        return dict(_KM_EMPTY)
    try:
        from encodev2 import _reconstruct_event_board
        board = _reconstruct_event_board(rec)
        mating_side = board.turn
        mated_side = not mating_side
        uci = [str(u) for u in (rec.get("solution_uci") or [])]
        pushed = []
        for u in uci:
            try:
                mv = chess.Move.from_uci(u)
                if mv not in board.legal_moves and len(u) == 4:
                    mv2 = chess.Move.from_uci(u + "q")
                    if mv2 in board.legal_moves:
                        mv = mv2
                if mv not in board.legal_moves:
                    break
                pushed.append(mv)
            except Exception:
                break
        check_board = board.copy()
        ok = True
        for mv in pushed:
            if mv not in check_board.legal_moves:
                ok = False
                break
            check_board.push(mv)
        if not ok or not pushed or not check_board.is_checkmate():
            return dict(_KM_EMPTY)
        return _core_from_start(board, pushed, mating_side, mated_side)
    except Exception:
        return king_march(rec.get("fen", "") or rec.get("board_fen", ""),
                          rec.get("solution_uci"), rec.get("turn"))

def march_from_tokens(text_dynamic: str) -> dict:
    out = {"check_lines": [], "checker_seq": [], "king_displacement": 0,
           "same_line_run": 0, "repeat_checker_run": 0,
           "same_square_checker_run": 0, "cooperating_types": "",
           "mate_is_drop": False}
    found = False
    for t in (text_dynamic or "").split():
        if not t.startswith("km:"):
            continue
        found = True
        if t.startswith("km:line:"):
            v = t[len("km:line:"):]
            out["check_lines"] = [] if v == "none" else v.split("-")
        elif t.startswith("km:checkers:"):
            v = t[len("km:checkers:"):]
            out["checker_seq"] = [] if v == "none" else v.split("-")
        elif t.startswith("km:disp:"):
            out["king_displacement"] = int(t[len("km:disp:"):])
        elif t.startswith("km:sameline:"):
            out["same_line_run"] = int(t[len("km:sameline:"):])
        elif t.startswith("km:samepiece:"):
            out["repeat_checker_run"] = int(t[len("km:samepiece:"):])
        elif t.startswith("km:samesquare:"):
            out["same_square_checker_run"] = int(t[len("km:samesquare:"):])
        elif t.startswith("km:coop:"):
            v = t[len("km:coop:"):]
            out["cooperating_types"] = "" if v == "none" else v
        elif t == "km:matebydrop":
            out["mate_is_drop"] = True
    return out if found else {}

def get_march(rec: dict) -> dict:
    p = march_from_tokens(rec.get("text_dynamic", ""))
    if p:
        return p
    return king_march(rec.get("fen", ""), rec.get("solution_uci"), rec.get("turn"))

def _list_jac(a: list, b: list) -> float:
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    keys = set(ca) | set(cb)
    if not keys:
        return 0.5
    inter = sum(min(ca[k], cb[k]) for k in keys)
    union = sum(max(ca[k], cb[k]) for k in keys)
    return inter / union if union else 0.0

def compare_marches(qm: dict, cm: dict) -> dict:
    have = bool(qm) and bool(cm) and (qm.get("checker_seq") is not None) \
           and (cm.get("checker_seq") is not None) and (qm.get("check_lines") or qm.get("checker_seq"))
    have = have and (cm.get("check_lines") or cm.get("checker_seq"))
    if not have:
        return {"km_line_seq_match": 0.5, "km_checker_seq_match": 0.5,
                "km_line_jac": 0.5, "km_checker_jac": 0.5,
                "km_disp_sim": 0.5, "km_sameline_match": 0.5,
                "km_samepiece_match": 0.5, "km_samesquare_match": 0.5,
                "km_coop_sim": 0.5, "km_drop_mate_match": 0.5}
    f = {}
    f["km_line_seq_match"]    = 1.0 if qm["check_lines"] == cm["check_lines"] else 0.0
    f["km_checker_seq_match"] = 1.0 if qm["checker_seq"] == cm["checker_seq"] else 0.0
    f["km_line_jac"]    = _list_jac(qm["check_lines"], cm["check_lines"])
    f["km_checker_jac"] = _list_jac(qm["checker_seq"], cm["checker_seq"])
    qd, cd = qm.get("king_displacement", 0), cm.get("king_displacement", 0)
    f["km_disp_sim"] = 1.0 - min(abs(qd - cd), 7) / 7.0
    f["km_sameline_match"]   = 1.0 if qm.get("same_line_run") == cm.get("same_line_run") else 0.0
    f["km_samepiece_match"]  = 1.0 if qm.get("repeat_checker_run") == cm.get("repeat_checker_run") else 0.0
    f["km_samesquare_match"] = 1.0 if qm.get("same_square_checker_run") == cm.get("same_square_checker_run") else 0.0
    f["km_coop_sim"] = _list_jac(list(qm.get("cooperating_types", "")), list(cm.get("cooperating_types", "")))
    f["km_drop_mate_match"] = 1.0 if bool(qm.get("mate_is_drop")) == bool(cm.get("mate_is_drop")) else 0.0
    return f

KM_FEATURE_NAMES = ["km_line_seq_match", "km_checker_seq_match", "km_line_jac",
                    "km_checker_jac", "km_disp_sim", "km_sameline_match",
                    "km_samepiece_match", "km_samesquare_match", "km_coop_sim",
                    "km_drop_mate_match"]



def _to_bracket_fen(fen, turn):
    """Convert any crazyhouse FEN (9th-rank-slash OR [..] bracket pocket) into
    python-chess bracket form: '<8 ranks>[<pocket>] <turn> - - 0 1'."""
    s = (fen or "").strip()
    if not s:
        return None
    field = s.split(" ")
    board = field[0]
    pocket = ""
    if "[" in board:
        board, pk = board.split("[", 1)
        pocket = pk.rstrip("]")
    else:
        parts = board.split("/")
        if len(parts) == 9:
            pocket = parts[8]
            board = "/".join(parts[:8])
    tc = field[1] if len(field) > 1 and field[1] in ("w", "b") else \
         ("b" if str(turn or "white").lower().startswith("b") else "w")
    return f"{board}[{pocket}] {tc} - - 0 1"


_PIECE_VAL = {"P": 1.0, "N": 3.0, "B": 3.0, "R": 5.0, "Q": 9.0}


def _line_stats(fen, solution_uci, turn):
    """Replay the mate line ONCE and return:
      - actions: piece/colour/location-INVARIANT label per move
                 ('D' drop / 'M' move  + 'x' capture + '=' promo + '#'/'+' )
      - n_drops: number of drop moves in the line
      - sac:     material the MATING side gives up during the line (a
                 sacrifice proxy: value of its pieces the defender captures);
                 None when the rich replay is unavailable.
    Uses python-chess when available; falls back to UCI-only labels."""
    uci = [str(u) for u in (solution_uci or [])]
    out = {"actions": [], "n_drops": sum(1 for u in uci if "@" in u),
           "sac": None, "player_caps": None, "opp_caps": None, "rich": False,
           "mate_king_sq": None, "mate_piece_sq": None,
           "cap_sqs": frozenset(), "checkers": frozenset()}
    if not uci:
        return out
    if _HAS_CHESS:
        bfen = _to_bracket_fen(fen, turn)
        if bfen:
            try:
                board = chess.variant.CrazyhouseBoard(bfen)
                mating_side = board.turn            # side to move delivers mate
                seq = []
                sac = 0.0
                pcaps = ocaps = 0
                cap_sqs = set(); checkers = set(); last_mover_to = None
                for idx, u in enumerate(uci):
                    mv = chess.Move.from_uci(u)
                    if mv not in board.legal_moves and len(u) == 4:
                        # promotions sometimes stored without suffix ("h2g1")
                        mv2 = chess.Move.from_uci(u + "q")
                        if mv2 in board.legal_moves:
                            mv = mv2
                    if mv not in board.legal_moves:
                        # keep the rich prefix; coarse labels for the rest
                        for uu in uci[idx:]:
                            lab = "D" if "@" in uu else "M"
                            if len(uu) == 5:
                                lab += "="
                            seq.append(lab)
                        break
                    mover = board.turn
                    is_cap = board.is_capture(mv)
                    if is_cap and mv.to_square is not None:
                        cap_sqs.add(mv.to_square)        # squares where exchanges happen
                    cap_val = 0.0
                    if is_cap and mv.drop is None:
                        if board.is_en_passant(mv):
                            cap_val = 1.0
                        else:
                            pc = board.piece_at(mv.to_square)
                            if pc is not None:
                                cap_val = _PIECE_VAL.get(pc.symbol().upper(), 0.0)
                    lab = "D" if mv.drop is not None else "M"
                    if is_cap:
                        lab += "x"
                        if mover == mating_side:
                            pcaps += 1
                        else:
                            ocaps += 1
                    if mv.promotion:
                        lab += "="
                    board.push(mv)
                    if board.is_checkmate():
                        lab += "#"
                    elif board.is_check():
                        lab += "+"
                    seq.append(lab)
                    if mover == mating_side:
                        last_mover_to = mv.to_square
                        if "#" in lab or "+" in lab:          # this move gave check/mate
                            pc2 = board.piece_at(mv.to_square)
                            if pc2 is not None:
                                checkers.add(pc2.symbol().upper())
                    # defender capturing => the mating side lost that material
                    if mover != mating_side and cap_val:
                        sac += cap_val
                if seq:
                    out["actions"] = seq
                    out["n_drops"] = sum(1 for a in seq if a.startswith("D"))
                    out["sac"] = sac
                    out["player_caps"] = pcaps
                    out["opp_caps"] = ocaps
                    out["rich"] = True
                    out["cap_sqs"] = frozenset(cap_sqs)
                    out["checkers"] = frozenset(checkers)
                    if board.is_checkmate():
                        out["mate_king_sq"]  = board.king(board.turn)   # mated king square
                        out["mate_piece_sq"] = last_mover_to            # square the mate lands on
                    return out
            except Exception:
                pass
    seq = []                                  # UCI-only fallback
    for u in uci:
        lab = "D" if "@" in u else "M"
        if len(u) == 5:
            lab += "="
        seq.append(lab)
    out["actions"] = seq
    return out


def _placement_only(fen):
    return (fen or "").split(" ")[0].split("[")[0]


def _board_material(fen):
    """Total material value of every piece on the board AND in both pockets."""
    total = 0.0
    seg = (fen or "").split(" ")[0]
    placement = _placement_only(fen)
    for ch in placement:
        if ch.upper() in _PIECE_VAL:
            total += _PIECE_VAL[ch.upper()]
    # pocket: bracket "[QPp]" form or 9th-rank-slash form
    pocket = ""
    if "[" in seg:
        pocket = seg.split("[", 1)[1].rstrip("]")
    else:
        parts = seg.split("/")
        if len(parts) == 9:
            pocket = parts[8]
    for ch in pocket:
        if ch.upper() in _PIECE_VAL:
            total += _PIECE_VAL[ch.upper()]
    return total


def _king_shelter(fen, turn):
    """Count friendly pawns adjacent to the MATED king (shelter / smother)."""
    ksq = _mated_king_sq(fen, turn)
    if ksq is None:
        return None
    board = _expand_placement(_placement_only(fen))
    kf, kr = ksq
    pawn = "P" if board.get((kf, kr), "k").isupper() else "p"
    cnt = 0
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            f, r = kf + df, kr + dr
            if 0 <= f <= 7 and 0 <= r <= 7 and board.get((f, r)) == pawn:
                cnt += 1
    return cnt


def _mech_signature(seq):
    """Coarse, order-free mechanism fingerprint of an action sequence."""
    if not seq:
        return {}
    b = lambda n: 0 if n == 0 else (1 if n == 1 else 2)
    drops  = sum(1 for a in seq if a.startswith("D"))
    checks = sum(1 for a in seq if ("+" in a or "#" in a))
    caps   = sum(1 for a in seq if "x" in a)
    return {
        "ends_drop_mate": 1 if (seq[-1].startswith("D") and "#" in seq[-1]) else 0,
        "drops":  b(drops),
        "checks": b(checks),
        "caps":   b(caps),
        "first":  seq[0][:2],
    }


def _king_box(fen, turn):
    """Trap geometry of the MATED king: own-piece smother, escape room, edge,
    corner. Piece-agnostic, location-tolerant description of the mating net."""
    ksq = _mated_king_sq(fen, turn)
    if ksq is None:
        return {}
    placement = (fen or "").split(" ")[0].split("[")[0]
    board = _expand_placement(placement)
    kf, kr = ksq
    king_upper = board.get((kf, kr), "k").isupper()
    own = empty = 0
    for df in (-1, 0, 1):
        for dr in (-1, 0, 1):
            if df == 0 and dr == 0:
                continue
            f, r = kf + df, kr + dr
            if not (0 <= f <= 7 and 0 <= r <= 7):
                continue
            pc = board.get((f, r))
            if pc is None:
                empty += 1
            elif pc.isupper() == king_upper:
                own += 1
    b = lambda n: 0 if n <= 1 else (1 if n <= 3 else 2)
    return {
        "own_adj":   b(own),
        "empty_adj": b(empty),
        "edge":   1 if (kf in (0, 7) or kr in (0, 7)) else 0,
        "corner": 1 if (kf in (0, 7) and kr in (0, 7)) else 0,
    }


def _agree_frac(da, db) -> float:
    """Fraction of shared keys whose values agree (0.5 if nothing to compare)."""
    if not da or not db:
        return 0.5
    keys = set(da) & set(db)
    if not keys:
        return 0.5
    return sum(1 for k in keys if da[k] == db[k]) / len(keys)


def pair_features(query: dict, cand: dict) -> dict:
    """
    Build the pairwise feature dict for one (query, candidate).

    Both `query` and `cand` are dicts that expose (any missing -> safe default):
        text_static   : str
        text_dynamic  : str
        solution_uci  : list[str]
        mate          : int
        turn          : "white"/"black"
        rating        : float
        fen           : str   (crazyhouse FEN incl. [pocket])  -- optional
        pockets_self  : dict  -- optional, overrides fen-derived pocket

    Returns an ordered-by-FEATURE_NAMES dict of floats.
    """
    qb = bucket_tokens(query.get("text_static", "") + " " + query.get("text_dynamic", ""))
    cb = bucket_tokens(cand.get("text_static", "")  + " " + cand.get("text_dynamic", ""))

    feats: dict[str, float] = {}

    # --- per-family overlap (the retrieval-steering signal) ---------------
    for fam in ALL_FAMILIES:
        feats[f"jac_{fam}"]  = _jaccard(qb[fam], cb[fam])
        feats[f"ovl_{fam}"]  = _overlap_coef(qb[fam], cb[fam])
        feats[f"nshare_{fam}"] = float(len(qb[fam] & cb[fam]))

    # all-token jaccard (coarse global similarity)
    q_all = set().union(*qb.values())
    c_all = set().union(*cb.values())
    feats["jac_all"] = _jaccard(q_all, c_all)

    # --- solution-line comparison ----------------------------------------
    q_uci = query.get("solution_uci") or []
    c_uci = cand.get("solution_uci") or []
    feats["move_lev"]   = _levenshtein_sim(q_uci, c_uci)
    feats["pvlen_q"]    = float(len(q_uci))
    feats["pvlen_diff"] = float(abs(len(q_uci) - len(c_uci)))

    # replay each line once (actions, drop count, sacrifice value, capture counts)
    q_st = _line_stats(query.get("fen", ""), q_uci, query.get("turn"))
    c_st = _line_stats(cand.get("fen", ""),  c_uci, cand.get("turn"))

    # dynamic token sets for structured comparisons
    q_dyn = qb["pv"] | qb["dyn"] | qb["sum"]
    c_dyn = cb["pv"] | cb["dyn"] | cb["sum"]

    q_mp, c_mp = _mating_piece(q_dyn), _mating_piece(c_dyn)
    feats["mating_match"] = (1.0 if (q_mp and c_mp and q_mp == c_mp)
                             else 0.0 if (q_mp and c_mp) else 0.5)

    q_fa, c_fa = _first_action(q_dyn), _first_action(c_dyn)
    feats["first_match"] = (1.0 if (q_fa and c_fa and q_fa == c_fa)
                            else 0.0 if (q_fa and c_fa) else 0.5)

    q_dp, c_dp = _drop_pieces(q_dyn), _drop_pieces(c_dyn)
    feats["drop_jac"] = (_jaccard(q_dp, c_dp) if (q_dp or c_dp) else 0.5)

    # === piece-event agreement (transferable from the chess-tactics catalog) ==
    # All colour-folded, since expert similarity here is colour-invariant.
    # Parsed from dynamic-solution tokens already present in the corpus.
    q_cap, c_cap = _captured_pieces(q_dyn), _captured_pieces(c_dyn)
    q_caps, c_caps = set(q_cap), set(c_cap)
    feats["captured_jac"]   = (_jaccard(q_caps, c_caps) if (q_caps or c_caps) else 0.5)
    feats["captured_msjac"] = _multiset_jaccard(q_cap, c_cap)
    for P in ("Q", "R", "B", "N", "P"):           # per-type captured agreement
        feats[f"capt_{P}_match"] = 1.0 if ((P in q_caps) == (P in c_caps)) else 0.0

    # consecutive capture-pair ordering agreement
    q_cpair, c_cpair = _capture_pairs(q_dyn), _capture_pairs(c_dyn)
    feats["cappair_jac"] = (_jaccard(q_cpair, c_cpair) if (q_cpair or c_cpair) else 0.5)

    # attack-relation agreement (which piece attacks which during the line)
    q_atk, c_atk = _attack_relations(q_dyn), _attack_relations(c_dyn)
    feats["attack_jac"] = (_jaccard(q_atk, c_atk) if (q_atk or c_atk) else 0.5)
    feats["attack_ovl"] = _overlap_coef(q_atk, c_atk)

    # piece-moved agreement (which piece types are active in the line)
    q_mv, c_mv = _moved_pieces(q_dyn), _moved_pieces(c_dyn)
    feats["moved_jac"] = (_jaccard(q_mv, c_mv) if (q_mv or c_mv) else 0.5)

    # promotion agreement
    q_promo = ("=" in "".join(q_st["actions"])) or _has_tok(q_dyn, "pv:promotion")
    c_promo = ("=" in "".join(c_st["actions"])) or _has_tok(c_dyn, "pv:promotion")
    feats["promo_match"] = 1.0 if (q_promo == c_promo) else 0.0

    # player- / opponent-capture presence agreement (who initiates captures)
    if q_st["player_caps"] is None or c_st["player_caps"] is None:
        feats["player_cap_match"] = 0.5
        feats["opp_cap_match"]    = 0.5
    else:
        feats["player_cap_match"] = 1.0 if ((q_st["player_caps"] > 0) == (c_st["player_caps"] > 0)) else 0.0
        feats["opp_cap_match"]    = 1.0 if ((q_st["opp_caps"]    > 0) == (c_st["opp_caps"]    > 0)) else 0.0

    # === Crazyhouse drop & pocket dynamics (axis the chess catalog lacks) =====
    feats["drops_diff"]       = float(abs(q_st["n_drops"] - c_st["n_drops"]))
    feats["drop_count_match"] = 1.0 if (q_st["n_drops"] == c_st["n_drops"]) else 0.0

    qdc = _has_tok(q_dyn, "dyn:hasDropCheck", "pv:dropCheck")
    cdc = _has_tok(c_dyn, "dyn:hasDropCheck", "pv:dropCheck")
    feats["drop_check_match"] = 1.0 if (qdc == cdc) else 0.0

    qds = _has_tok(q_dyn, "pv:dropSacrifice", "dyn:dropSacrifice")
    cds = _has_tok(c_dyn, "pv:dropSacrifice", "dyn:dropSacrifice")
    feats["drop_sac_match"] = 1.0 if (qds == cds) else 0.0

    def _mate_by_drop(st, toks):
        if _has_tok(toks, "dyn:mateDrop"):
            return True
        a = st["actions"]
        return bool(a) and a[-1].startswith("D") and "#" in a[-1]
    feats["mate_drop_match"] = 1.0 if (_mate_by_drop(q_st, q_dyn) ==
                                       _mate_by_drop(c_st, c_dyn)) else 0.0

    qnk = _has_tok(q_dyn, "dyn:dropNearKing")
    cnk = _has_tok(c_dyn, "dyn:dropNearKing")
    feats["drop_near_king_match"] = 1.0 if (qnk == cnk) else 0.0

    # --- pockets ----------------------------------------------------------
    def _pockets(rec):
        if isinstance(rec.get("pockets_self"), list):
            return rec["pockets_self"]
        turn = (rec.get("turn") or "white").lower()
        w, b = _pockets_from_fen(rec.get("fen", ""))
        return w if turn.startswith("w") else b
    feats["pocket_jac"] = _multiset_jaccard(_pockets(query), _pockets(cand))
    feats["pocket_size_diff"] = float(abs(len(_pockets(query)) - len(_pockets(cand))))

    # --- scalar position metadata ----------------------------------------
    qm, cm = query.get("mate"), cand.get("mate")
    feats["mate_diff"] = float(abs((qm or 0) - (cm or 0)))

    # side-to-move agreement (re-added by request; note it can penalise
    # genuine cross-colour motif matches, so watch its sign in importances)
    qt = str(query.get("turn") or "").lower()
    ct = str(cand.get("turn") or "").lower()
    feats["turn_match"] = 1.0 if (qt and ct and qt[0] == ct[0]) else 0.0

    # mechanism-level features (how the mate is engineered) -- color/location
    # invariant; target the expert's "same tactical idea" notion that
    # mating-piece and mate-geometry features could not capture.
    q_seq, c_seq = q_st["actions"], c_st["actions"]
    feats["movetype_sim"] = _levenshtein_sim(q_seq, c_seq)         # same action ORDER
    feats["mech_match"]   = _agree_frac(_mech_signature(q_seq),    # same action MIX
                                        _mech_signature(c_seq))
    feats["king_box_sim"] = _agree_frac(_king_box(query.get("fen", ""), query.get("turn")),
                                        _king_box(cand.get("fen", ""),  cand.get("turn")))

    # sacrifice: do both lines give up material on the way to mate?
    if q_st["sac"] is None or c_st["sac"] is None:
        feats["sac_match"] = 0.5
    else:
        feats["sac_match"] = 1.0 if ((q_st["sac"] > 0) == (c_st["sac"] > 0)) else 0.0

    # king shelter: similarity of friendly-pawn cover around the mated king
    qsh, csh = (_king_shelter(query.get("fen", ""), query.get("turn")),
                _king_shelter(cand.get("fen", ""),  cand.get("turn")))
    feats["king_shelter_sim"] = (0.5 if (qsh is None or csh is None)
                                 else 1.0 - abs(qsh - csh) / 8.0)

    # total board+pocket material difference (position density)
    feats["board_material_diff"] = abs(_board_material(query.get("fen", "")) -
                                       _board_material(cand.get("fen", ""))) / 10.0

    qr = query.get("rating") or 0.0
    cr = cand.get("rating") or 0.0
    feats["rating_absdiff"] = float(abs(qr - cr)) / 1000.0  # scaled

    # --- expert-feedback features (iteration 5; validated on the critical pairs) ---
    # The expert's reasoning about "similar" mates reduced to: same square the king
    # is mated on, same square the mating piece lands on, exchanges on shared squares,
    # and the same kind of pieces giving check. (square index: rank = sq>>3, file = sq&7)
    qk, ck = q_st.get("mate_king_sq"), c_st.get("mate_king_sq")
    if qk is not None and ck is not None:
        feats["fb_same_mate_sq"]   = 1.0 if qk == ck else 0.0
        feats["fb_same_mate_rank"] = 1.0 if (qk >> 3) == (ck >> 3) else 0.0
        feats["fb_same_mate_file"] = 1.0 if (qk & 7) == (ck & 7) else 0.0
    else:
        feats["fb_same_mate_sq"] = feats["fb_same_mate_rank"] = feats["fb_same_mate_file"] = 0.0
    qp, cp = q_st.get("mate_piece_sq"), c_st.get("mate_piece_sq")
    feats["fb_same_matepiece_sq"] = 1.0 if (qp is not None and cp is not None and qp == cp) else 0.0
    feats["fb_capsq_jac"]   = _jaccard(set(q_st.get("cap_sqs") or ()), set(c_st.get("cap_sqs") or ()))
    feats["fb_checker_jac"] = _jaccard(set(q_st.get("checkers") or ()), set(c_st.get("checkers") or ()))

    # --- mating picture (matna slika): the colour+orientation-folded cage
    # around the mated king at the FINAL position. Picture tokens are read
    # from stored dynamic text when present (corpus), else replayed live.
    # (get_picture/compare_pictures degrade to neutral 0.5s on their own if
    # python-chess is unavailable or the line doesn't resolve to mate, so no
    # outer availability flag is needed here.)
    q_pic = get_picture({"text_dynamic": query.get("text_dynamic", ""),
                         "fen": query.get("fen", ""),
                         "solution_uci": query.get("solution_uci"),
                         "turn": query.get("turn")})
    c_pic = get_picture({"text_dynamic": cand.get("text_dynamic", ""),
                         "fen": cand.get("fen", ""),
                         "solution_uci": cand.get("solution_uci"),
                         "turn": cand.get("turn")})
    feats.update(compare_pictures(q_pic, c_pic))

    # --- king march (forcing sequence): which line/squares the checks
    # come from, whether the same piece/square checks repeatedly, whether
    # the mating move is a drop vs a moved piece, and which figures (both
    # attacker net and defender cage) cooperate around the final mated
    # king. Complements the mating picture (final position only) with the
    # PATH taken to get there.
    q_km = get_march({"text_dynamic": query.get("text_dynamic", ""),
                      "fen": query.get("fen", ""),
                      "solution_uci": query.get("solution_uci"),
                      "turn": query.get("turn")})
    c_km = get_march({"text_dynamic": cand.get("text_dynamic", ""),
                      "fen": cand.get("fen", ""),
                      "solution_uci": cand.get("solution_uci"),
                      "turn": cand.get("turn")})
    feats.update(compare_marches(q_km, c_km))

    return feats


# Stable feature-name ordering -- compute once from a probe pair.
def feature_names() -> list[str]:
    probe = pair_features(
        {"text_static": "wp@e4", "text_dynamic": "pv:capture",
         "solution_uci": ["e2e4"], "mate": 3, "turn": "white", "rating": 1500,
         "fen": "7k/8/8/8/8/8/8/7K w - - 0 1"},
        {"text_static": "bp@e5", "text_dynamic": "dyn:hasDrop",
         "solution_uci": ["e7e5"], "mate": 3, "turn": "black", "rating": 1500,
         "fen": "7k/8/8/8/8/8/8/7K b - - 0 1"},
    )
    return list(probe.keys())


FEATURE_NAMES = feature_names()