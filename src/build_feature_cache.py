"""
build_feature_cache.py
----------------------
One-time script: converts puzzle_features_mates.parquet into a
numpy memmap (.npy binary) + a compact id→row index (.json).

The memmap is ~3.3 GB on disk but the OS loads only the pages
(rows) that are actually accessed — so RAM usage is proportional
to the number of unique positions queried, not the full corpus.

app.py uses this for fast on-demand re-ranking without loading
the full feature matrix into RAM at startup.

Run once after extract_features_crazyhouse.py:
    python build_feature_cache.py
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT        = Path(__file__).resolve().parents[1]
FEAT_PATH   = ROOT / "data" / "derived" / "puzzle_features_mates.parquet"
CACHE_DIR   = ROOT / "data" / "models"
CACHE_NPY   = CACHE_DIR / "feat_matrix.npy"
CACHE_IDX   = CACHE_DIR / "feat_index.json"   # puzzle_id -> row int
CACHE_COLS  = CACHE_DIR / "feat_cols.json"    # ordered list of feature col names

META_COLS = [
    "puzzle_id", "site", "ply", "board_fen", "turn",
    "pockets_white_raw", "pockets_black_raw",
]

CHUNK_SIZE = 200_000


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Reading schema from: {FEAT_PATH}")
    sample = pd.read_parquet(FEAT_PATH, columns=["puzzle_id"])
    n_rows = len(sample)
    del sample

    # Get feature columns
    all_cols = pd.read_parquet(FEAT_PATH, columns=None).columns.tolist()
    feat_cols = [c for c in all_cols if c not in META_COLS]
    n_feats   = len(feat_cols)
    print(f"  Rows     : {n_rows:,}")
    print(f"  Features : {n_feats}")

    size_mb = n_rows * n_feats * 4 / 1024 / 1024
    print(f"  Memmap   : {size_mb:.0f} MB ({size_mb/1024:.1f} GB) on disk")

    # Save feature column names
    with open(CACHE_COLS, "w") as f:
        json.dump(feat_cols, f)
    print(f"Saved feat_cols: {CACHE_COLS}")

    # Create memmap
    mm = np.lib.format.open_memmap(
        str(CACHE_NPY),
        mode="w+",
        dtype=np.float32,
        shape=(n_rows, n_feats),
    )

    id_index = {}
    row = 0

    print(f"\nWriting memmap in chunks of {CHUNK_SIZE:,}...")
    reader = pd.read_parquet(FEAT_PATH, columns=["puzzle_id"] + feat_cols)

    # Process in chunks manually since parquet doesn't natively chunk
    # Use numpy slicing on the loaded df
    n_chunks = (n_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
    df = reader  # full load needed for parquet, but we write chunk-by-chunk to memmap

    for chunk_start in range(0, n_rows, CHUNK_SIZE):
        chunk = df.iloc[chunk_start:chunk_start + CHUNK_SIZE]
        chunk_vals = chunk[feat_cols].fillna(0).values.astype(np.float32)
        mm[chunk_start:chunk_start + len(chunk)] = chunk_vals

        for pid in chunk["puzzle_id"]:
            id_index[pid] = row
            row += 1

        print(f"  written: {row:,} / {n_rows:,}")

    mm.flush()
    print(f"\nMemmap written: {CACHE_NPY}")

    # Save id index
    with open(CACHE_IDX, "w") as f:
        json.dump(id_index, f)
    print(f"Index written : {CACHE_IDX}  ({len(id_index):,} entries)")
    print("\nDone. app.py will use this cache for fast re-ranking.")


if __name__ == "__main__":
    main()
