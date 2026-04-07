"""
extract_features_crazyhouse.py
------------------------------
Converts corpus_full.jsonl into a flat feature DataFrame (1 row per puzzle).
Mirrors the mentor's extract_features() from puzzle_dataset notebook,
adapted for Crazyhouse (adds PK_ pocket features, drop flags, etc.).

Writes output incrementally in chunks so progress is visible and
the file exists on disk throughout the run.

Output:
    data/derived/puzzle_features.parquet   (primary ML input)
    data/derived/puzzle_features.csv       (human-readable)

Usage:
    python extract_features_crazyhouse.py
"""

from pathlib import Path
from collections import defaultdict
import json
import re
import pandas as pd

ROOT   = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "derived" / "corpus_mates.jsonl"
OUT_PARQUET = ROOT / "data" / "derived" / "puzzle_features_mates.parquet"
OUT_CSV     = ROOT / "data" / "derived" / "puzzle_features_mates.csv"

CHUNK_SIZE = 50_000   # flush to disk every N records

# ---------------------------------------------------------------------------
# Regex helpers
# ---------------------------------------------------------------------------
pocket_tok = re.compile(r'^([wb])pocket:([QRBNP])$')
weight_re  = re.compile(r'\|(\d+\.?\d*)$')


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

    # --- 1) Static board features ---
    static_tokens = rec.get("text_static", "").split()
    mob_sum = 0.0
    mob_n   = 0

    for tok in static_tokens:
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

        m = pocket_tok.match(tok)
        if m:
            side  = "white" if m.group(1) == 'w' else "black"
            piece = m.group(2)
            f[f"PK_{side}_{piece}"] += 1

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

    # --- 2) Dynamic features ---
    dyn_tokens = rec.get("text_dynamic", "").split()
    seen_dyn = set()

    for tok in dyn_tokens:
        if tok in seen_dyn:
            continue
        seen_dyn.add(tok)

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
        elif tok == "pv:drop":
            f["DS_!drop"] = 1
        elif tok == "pv:dropCheck":
            f["DS_!dropCheck"] = 1
        elif tok == "pv:mate":
            f["DS_!mate"] = 1
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
        elif tok == "pv:captureCheck":
            f["DS_!captureCheck"] = 1
        elif tok == "pv:capture":
            f["DS_!capture"] = 1
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
        elif tok == "dyn:noDrop":
            f["DG_!F_noDrop"] = 1
        elif tok == "dyn:noCapture":
            f["DG_!F_noCapture"] = 1

    f["ply"] = float(rec.get("ply", 0))
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

    with CORPUS.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
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

    print(f"\nConverting CSV to parquet ...")
    df = pd.read_csv(OUT_CSV, low_memory=False)
    df.to_parquet(OUT_PARQUET, index=False)

    print(f"\nDone.")
    print(f"  {OUT_CSV}")
    print(f"  {OUT_PARQUET}")
    print(f"  Shape: {df.shape}")


if __name__ == "__main__":
    main()