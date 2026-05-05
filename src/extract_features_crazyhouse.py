"""
extract_features_crazyhouse.py
------------------------------
Converts corpus_full.jsonl into a flat feature DataFrame (1 row per puzzle).
Mirrors the mentor's extract_features() from puzzle_dataset notebook,
adapted for Crazyhouse (adds PK_ pocket features, drop flags, etc.).

v2 enrichments (feature_descriptions_summary_2_.md alignment, April 2026):
  Static connectivity (section 3.3):
    white_attacks, white_defends, white_xrays (and black equiv.)
    parsed from X>Y, X<Y, X=Y tokens in text_static
  Static pawn structure (section 3.4):
    white_isolated, white_passed, white_protected_passed, white_backward,
    white_doubled, white_pawn_chain, white_pawn_islands (and black equiv.)
    parsed from I/F/'F/L/S/W/P tokens in text_static (static_pawns field)

v4 enrichments (encodev2.py v4 new tokens):
  DS_!piece_moved_X   — from pv:pieceMoved:X tokens
  DS_!attack_X>Y      — from pv:attack:X>Y tokens
  DG_!F_p+            — from dyn:hasPromotion token (FIXED: was missing)
  DG_!mating_piece_X  — from dyn:mating_piece:X (FIXED naming, was matingPiece)
  DG_!matingPiecePair_XY — from dyn:matingPiecePair:XY (NEW)

Writes output incrementally in chunks so progress is visible.

Output:
    data/derived/puzzle_features_mates.parquet   (primary ML input)
    data/derived/puzzle_features_mates.csv       (human-readable)

Usage:
    python extract_features_crazyhouse.py
"""

from pathlib import Path
from collections import defaultdict
import json
import re
import pandas as pd

ROOT        = Path(__file__).resolve().parents[1]
CORPUS      = ROOT / "data" / "derived" / "corpus_mates.jsonl"
OUT_PARQUET = ROOT / "data" / "derived" / "puzzle_features_mates.parquet"
OUT_CSV     = ROOT / "data" / "derived" / "puzzle_features_mates.csv"

CHUNK_SIZE = 50_000   # flush to disk every N records

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
pocket_tok = re.compile(r'^([wb])pocket:([QRBNP])$')
weight_re  = re.compile(r'\|(\d+\.?\d*)$')

# Static connectivity: X>Ysq, X<Ysq, X=Ysq
# Uppercase first char = White piece acting; lowercase = Black piece acting
conn_re = re.compile(r'^([A-Za-z])(>|<|=)([A-Za-z])([a-h][1-8])$')

# Pawn structure tokens (from static_pawns / encode_pawn_structure output)
# I{sq} i{sq} F{sq} f{sq} 'F{sq} 'f{sq} L{sq} l{sq} S{sq} s{sq}
pawn_struct_re = re.compile(
    r"^(\'?)([IiFfLlSs])([a-h][1-8])$"   # single-pawn tags
)
pawn_chain_re  = re.compile(r"^([Ww])\[([a-h][1-8/]+)\]$")
pawn_island_re = re.compile(r"^([Pp])\((\d+)\)$")


def _sq_zone(sq: str) -> str:
    if len(sq) < 2:
        return "unknown"
    file = sq[0]
    rank = int(sq[1]) if sq[1].isdigit() else 0
    if rank == 1 or rank == 8:
        return "back_rank"
    if file in "efgh":
        return "kingside"
    if file in "abcd":
        return "queenside"
    return "center"


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------
def extract_features(rec: dict) -> dict:
    f = defaultdict(float)

    # =========================================================================
    # 1) Static board features
    # =========================================================================
    static_tokens = rec.get("text_static", "").split()
    mob_sum = 0.0
    mob_n   = 0

    for tok in static_tokens:
        # ── Piece placement ──────────────────────────────────────────────
        m = re.match(r'^([wb])([KQRBNPkqrbnp])@([a-h][1-8])$', tok)
        if m:
            color_char, piece_char, sq = m.group(1), m.group(2), m.group(3)
            side = "white" if color_char == 'w' else "black"
            kind = piece_char.lower()
            f[f"{side}_{kind}_count"] += 1
            if kind == 'k':
                f[f"{side}_king_rank"] = int(sq[1])
                f[f"{side}_king_file"] = ord(sq[0]) - 96
            m_w = weight_re.search(tok)
            if m_w:
                mob_sum += float(m_w.group(1))
                mob_n   += 1
            continue

        # ── Pocket tokens ────────────────────────────────────────────────
        m = pocket_tok.match(tok)
        if m:
            side  = "white" if m.group(1) == 'w' else "black"
            piece = m.group(2)
            f[f"PK_{side}_{piece}"] += 1
            continue

        # ── Connectivity tokens (v2 enrichment) ─────────────────────────
        # Format: ActorPiece > TargetPiece TargetSquare
        # e.g. B>pg7, R<Kg1, Q=pe6
        # Uppercase actor = White piece acting; lowercase = Black piece acting
        m = conn_re.match(tok)
        if m:
            actor_sym, relation, target_sym, sq = m.groups()
            # Determine acting side from case of actor symbol
            actor_side = "white" if actor_sym.isupper() else "black"
            if relation == ">":
                f[f"{actor_side}_attacks"] += 1
            elif relation == "<":
                f[f"{actor_side}_defends"] += 1
            elif relation == "=":
                f[f"{actor_side}_xrays"] += 1
            continue

        # ── Pawn structure tokens (v2 enrichment) ───────────────────────
        # Single-pawn tags: I/i=isolated, F/f=passed, 'F/'f=protected_passed,
        #                   L/l=backward, S/s=doubled
        m = pawn_struct_re.match(tok)
        if m:
            prefix, letter, sq = m.group(1), m.group(2), m.group(3)
            side = "white" if letter.isupper() else "black"
            is_protected = (prefix == "'")

            letter_lower = letter.lower()
            if letter_lower == 'i':
                f[f"{side}_isolated"] += 1
            elif letter_lower == 'f':
                if is_protected:
                    f[f"{side}_protected_passed"] += 1
                else:
                    f[f"{side}_passed"] += 1
            elif letter_lower == 'l':
                f[f"{side}_backward"] += 1
            elif letter_lower == 's':
                f[f"{side}_doubled"] += 1
            continue

        # Pawn chain: W[sq/sq/...] or w[sq/sq/...]
        m = pawn_chain_re.match(tok)
        if m:
            sym, sqs = m.group(1), m.group(2)
            side = "white" if sym == "W" else "black"
            n_chain = len(sqs.split("/"))
            f[f"{side}_pawn_chain"] += n_chain
            continue

        # Pawn island count: P(n) or p(n)
        m = pawn_island_re.match(tok)
        if m:
            sym, n = m.group(1), int(m.group(2))
            side = "white" if sym == "P" else "black"
            f[f"{side}_pawn_islands"] = float(n)
            continue

    if mob_n:
        f["mobility_weight_sum"] = mob_sum
        f["mobility_weight_n"]   = mob_n
        f["avg_piece_mobility"]  = mob_sum / mob_n

    for piece in "QRBNP":
        f.setdefault(f"PK_white_{piece}", 0.0)
        f.setdefault(f"PK_black_{piece}", 0.0)

    total_w = sum(f[f"PK_white_{p}"] for p in "QRBNP")
    total_b = sum(f[f"PK_black_{p}"] for p in "QRBNP")
    f["PK_total_white"] = total_w
    f["PK_total_black"] = total_b

    turn = rec.get("turn", "white")
    if turn == "white":
        f["PK_total_solver"]   = total_w
        f["PK_total_opponent"] = total_b
    else:
        f["PK_total_solver"]   = total_b
        f["PK_total_opponent"] = total_w

    # =========================================================================
    # 2) Dynamic features
    # =========================================================================
    dyn_tokens = rec.get("text_dynamic", "").split()
    seen_dyn = set()

    for tok in dyn_tokens:
        if tok in seen_dyn:
            continue
        seen_dyn.add(tok)

        # ── Dynamic General flags (DG_) ──────────────────────────────────
        if tok == "dyn:hasMate":
            f["DG_!F_checkmate"] = 1
        elif tok == "dyn:hasDrop":
            f["DG_!F_drop"] = 1
        elif tok == "dyn:hasDropCheck":
            f["DG_!F_drop+"] = 1
        elif tok == "dyn:hasCapture":
            f["DG_!F_px"] = 1
        elif tok == "dyn:hasCheck":
            f["DG_!F_+"] = 1
        elif tok == "dyn:hasPromotion":          # NEW v4 — was missing, maps to DG_!F_p+
            f["DG_!F_p+"] = 1
        elif tok == "dyn:hasSacrifice":
            f["DG_!F_sacrifice"] = 1
        elif tok == "dyn:oppCaptures":
            f["DG_!F_ox"] = 1
        elif tok == "dyn:noOppCaptures":
            f["DG_!F_noOx"] = 1
        elif tok == "dyn:noDrop":
            f["DG_!F_noDrop"] = 1
        elif tok == "dyn:noCapture":
            f["DG_!F_noCapture"] = 1

        # ── Mating piece identity (FIXED naming: mating_piece not matingPiece) ──
        elif tok.startswith("dyn:mating_piece:"):
            piece = tok.split("dyn:mating_piece:")[1]
            f[f"DG_!mating_piece_{piece}"] = 1
        elif tok.startswith("dyn:mateDrop:"):
            piece = tok.split("dyn:mateDrop:")[1]
            f[f"DG_!mateDrop_{piece}"] = 1
            f["DG_!F_mateByDrop"] = 1

        # ── Mating piece PAIRS (NEW v4) ──────────────────────────────────
        # e.g. dyn:matingPiecePair:bb → DG_!mating_piece_bb (LogReg top-10)
        elif tok.startswith("dyn:matingPiecePair:"):
            pair = tok.split("dyn:matingPiecePair:")[1]
            f[f"DG_!mating_piece_{pair}"] = 1

        # ── Per-move tokens (DS_) ────────────────────────────────────────
        elif tok == "pv:drop":
            f["DS_!drop"] = 1
        elif tok == "pv:dropCheck":
            f["DS_!dropCheck"] = 1
        elif tok == "pv:mate":
            f["DS_!mate"] = 1
        elif tok == "pv:capture":
            f["DS_!capture"] = 1
        elif tok == "pv:captureCheck":
            f["DS_!captureCheck"] = 1
        elif tok == "pv:promotion":              # NEW v4
            f["DS_!promotion"] = 1

        elif tok.startswith("pv:dropPiece:"):
            piece = tok.split("pv:dropPiece:")[1]
            f[f"DS_!drop_piece_{piece}"] = 1
        elif tok.startswith("pv:dropSq:"):
            sq   = tok.split("pv:dropSq:")[1]
            zone = _sq_zone(sq)
            f[f"DS_!drop_sq_zone_{zone}"] = 1
        elif tok.startswith("pv:capturePiece:"):
            piece = tok.split("pv:capturePiece:")[1]
            f[f"DS_!piece_captured_{piece}"] = 1
        elif tok.startswith("pv:promotionPiece:"):  # NEW v4
            piece = tok.split("pv:promotionPiece:")[1]
            f[f"DS_!promotionPiece_{piece}"] = 1

        # ── Piece-moved tokens (NEW v4) ──────────────────────────────────
        # pv:pieceMoved:X → DS_!piece_moved_X (permutation rank 17)
        elif tok.startswith("pv:pieceMoved:"):
            piece = tok.split("pv:pieceMoved:")[1]
            f[f"DS_!piece_moved_{piece}"] += 1   # count occurrences

        # ── Attack-relation tokens (NEW v4) ──────────────────────────────
        # pv:attack:X>Y → DS_!attack_X>Y (permutation rank 14)
        elif tok.startswith("pv:attack:"):
            relation = tok.split("pv:attack:")[1]  # e.g. Q>r
            f[f"DS_!attack_{relation}"] += 1        # count occurrences

        elif tok == "pv:sacrifice":
            f["DS_!sacrifice"] = 1
        elif tok == "pv:dropSacrifice":
            f["DS_!dropSacrifice"] = 1
        elif tok.startswith("pv:sacrificePiece:"):
            piece = tok.split("pv:sacrificePiece:")[1]
            f[f"DS_!sacrificePiece_{piece}"] = 1

        # ── Aggregate counts (dyn: tokens) ───────────────────────────────
        elif tok.startswith("dyn:drops:"):
            f[f"dyn_drops_{tok.split(':')[2]}"] = 1
        elif tok.startswith("dyn:captures:"):
            f[f"dyn_captures_{tok.split(':')[2]}"] = 1
        elif tok.startswith("dyn:pvLen:"):
            f[f"dyn_pvLen_{tok.split(':')[2]}"] = 1
        elif tok.startswith("dyn:first:"):
            f[f"dyn_first_{tok.split(':')[2]}"] = 1
        elif tok.startswith("dyn:dropPieces:"):
            pieces = tok.split("dyn:dropPieces:")[1]
            f[f"dyn_dropPieces_{pieces}"] = 1
            for p in pieces:
                f[f"dyn_dropHas_{p}"] = 1

        # ── Consecutive capture pairs (Tier-3) ────────────────────────────
        elif tok.startswith("dyn:capPair:"):
            pair = tok.split("dyn:capPair:")[1]
            f[f"DS_!piece_captured_{pair}"] = 1   # DS_!piece_captured_XY (consecutive)

        # ── Per-piece capture totals ──────────────────────────────────────
        elif tok.startswith("dyn:capPieceTotal:"):
            rest  = tok.split("dyn:capPieceTotal:")[1]
            piece = rest[0]
            count = int(rest[1]) if len(rest) > 1 and rest[1].isdigit() else 1
            f[f"DS_!capTotal_{piece}"] += count

        # ── Drop proximity ────────────────────────────────────────────────
        elif tok == "dyn:dropNearKing":
            f["DS_!dropNearKing"] = 1
        elif tok == "dyn:dropFarKing":
            f["DS_!dropFarKing"] = 1

        # ── Summary block tokens (sum:) ───────────────────────────────────
        elif tok.startswith("sum:mating_piece:"):
            piece = tok.split("sum:mating_piece:")[1]
            f[f"sum_mating_piece_{piece}"] = 1
        elif tok == "sum:hasSacrifice":
            f["sum_hasSacrifice"] = 1
        elif tok == "sum:oppCaptures":
            f["sum_oppCaptures"] = 1
        elif tok == "sum:hasPromotion":          # NEW v4
            f["sum_hasPromotion"] = 1

        elif tok.startswith("sum:pocketGain:"):
            rest  = tok.split("sum:pocketGain:")[1]
            piece = rest[0]
            count = int(rest[1]) if len(rest) > 1 and rest[1].isdigit() else 1
            f[f"sum_pocketGain_{piece}"] += count
        elif tok.startswith("sum:pocketSpend:"):
            rest  = tok.split("sum:pocketSpend:")[1]
            piece = rest[0]
            count = int(rest[1]) if len(rest) > 1 and rest[1].isdigit() else 1
            f[f"sum_pocketSpend_{piece}"] += count

    f["ply"] = float(rec.get("ply", 0))

    # =========================================================================
    # 3) Metadata features (section 4.3 of feature_descriptions_summary_2_.md)
    #    MD fields: length, rating, popularity
    #    Our equivalents (game-mined tactics have no Lichess rating/popularity):
    #      meta_length    — PV length in half-moves (solution complexity)
    #      meta_delta     — centipawn loss triggering the tactic (difficulty)
    #      meta_cp_before — engine eval before tactic (position sharpness)
    #      meta_mate_in   — forced mate depth (0 = not a forced mate)
    # =========================================================================
    # ── Engine-derived metadata ──────────────────────────────────────────────
    meta_length = rec.get("meta_length")
    if meta_length is not None:
        f["meta_length"] = float(meta_length)

    meta_delta = rec.get("meta_delta")
    if meta_delta is not None:
        f["meta_delta"] = float(min(meta_delta, 10_000))   # cap at mate-level

    meta_cp = rec.get("meta_cp_before")
    if meta_cp is not None:
        f["meta_cp_before"] = float(max(-2_000, min(2_000, meta_cp)))  # clip outliers

    meta_mate = rec.get("meta_mate_in")
    f["meta_mate_in"] = float(meta_mate) if meta_mate is not None else 0.0

    # ── Lichess-API-derived metadata (from enrich_with_ratings.py) ───────────
    # These map to the MD spec's "rating" and "popularity" fields.
    # They are None for records not yet enriched — stored as 0.0 (treated as missing).

    meta_avg_rating = rec.get("meta_avg_rating")
    if meta_avg_rating is not None:
        # Clip to [500, 3500] — valid Elo range on Lichess
        f["meta_avg_rating"] = float(max(500, min(3_500, meta_avg_rating)))

    meta_solver_rating = rec.get("meta_solver_rating")
    if meta_solver_rating is not None:
        f["meta_solver_rating"] = float(max(500, min(3_500, meta_solver_rating)))

    meta_white_rating = rec.get("meta_white_rating")
    if meta_white_rating is not None:
        f["meta_white_rating"] = float(max(500, min(3_500, meta_white_rating)))

    meta_black_rating = rec.get("meta_black_rating")
    if meta_black_rating is not None:
        f["meta_black_rating"] = float(max(500, min(3_500, meta_black_rating)))

    meta_est_time = rec.get("meta_estimated_time")
    if meta_est_time is not None:
        # Clip to [30, 7200] seconds — from bullet to classical
        f["meta_estimated_time"] = float(max(30, min(7_200, meta_est_time)))

    meta_clock_init = rec.get("meta_clock_initial")
    if meta_clock_init is not None:
        f["meta_clock_initial"] = float(meta_clock_init)

    meta_clock_inc = rec.get("meta_clock_inc")
    if meta_clock_inc is not None:
        f["meta_clock_inc"] = float(meta_clock_inc)

    # Speed as ordinal: bullet=1, blitz=2, rapid=3, classical=4, correspondence=5
    _SPEED_ORD = {"bullet": 1, "blitz": 2, "rapid": 3, "classical": 4, "correspondence": 5}
    meta_speed = rec.get("meta_speed")
    if meta_speed is not None:
        f["meta_speed_ord"] = float(_SPEED_ORD.get(meta_speed, 0))

    return dict(f)


# ---------------------------------------------------------------------------
# Chunked writer
# ---------------------------------------------------------------------------
def flush_chunk(rows: list, first_chunk: bool, columns: list) -> list:
    df = pd.DataFrame(rows).fillna(0)

    if first_chunk:
        id_cols = ["puzzle_id", "site", "ply", "board_fen", "turn",
                   "pockets_white_raw", "pockets_black_raw"]
        other   = sorted(c for c in df.columns if c not in id_cols)
        columns = id_cols + other
        for col in columns:
            if col not in df.columns:
                df[col] = 0
        df = df[columns]
        df.to_csv(OUT_CSV, index=False, mode='w')
    else:
        for col in columns:
            if col not in df.columns:
                df[col] = 0
        new_cols = [c for c in df.columns if c not in columns]
        if new_cols:
            columns = columns + sorted(new_cols)
        df = df.reindex(columns=columns, fill_value=0)
        df.to_csv(OUT_CSV, index=False, mode='a', header=False)

    return columns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

    chunk   = []
    columns = []
    n_total = 0
    n_chunk = 0
    first   = True

    print(f"Reading: {CORPUS}")
    print(f"Writing: {OUT_CSV}")
    print(f"Chunk size: {CHUNK_SIZE:,}")
    print()

    with CORPUS.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            pid = rec.get("id") or rec.get("event_id")
            if not pid:
                continue

            features = extract_features(rec)
            base = {
                "puzzle_id":         pid,
                "site":              rec.get("site", ""),
                "ply":               rec.get("ply", 0),
                "board_fen":         rec.get("board_fen", ""),
                "turn":              rec.get("turn", "white"),
                "pockets_white_raw": json.dumps(rec.get("pockets_white", {})),
                "pockets_black_raw": json.dumps(rec.get("pockets_black", {})),
            }
            chunk.append({**base, **features})
            n_total += 1
            n_chunk += 1

            if n_chunk >= CHUNK_SIZE:
                columns = flush_chunk(chunk, first, columns)
                first   = False
                chunk   = []
                n_chunk = 0
                print(f"  written: {n_total:,} rows ...")

    if chunk:
        columns = flush_chunk(chunk, first, columns)
        print(f"  written: {n_total:,} rows (final chunk)")

    print(f"\nTotal puzzles: {n_total:,}")
    print(f"Total columns: {len(columns)}")

    print(f"\nConverting CSV to parquet (chunked, tolerates column mismatches)...")
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    schema = None
    n_rows = 0

    for chunk_df in pd.read_csv(OUT_CSV, chunksize=50_000, low_memory=False, on_bad_lines="skip"):
        chunk_df = chunk_df.fillna(0)
        table = pa.Table.from_pandas(chunk_df, preserve_index=False)

        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(OUT_PARQUET, schema)

        try:
            table = table.cast(schema)
        except Exception:
            for col in schema.names:
                if col not in chunk_df.columns:
                    chunk_df[col] = 0
            chunk_df = chunk_df[schema.names]
            table = pa.Table.from_pandas(chunk_df, schema=schema, preserve_index=False)

        writer.write_table(table)
        n_rows += len(chunk_df)

    if writer:
        writer.close()

    print(f"\nDone.")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_PARQUET}")
    print(f"  Rows written: {n_rows:,}")


if __name__ == "__main__":
    main()