"""
csv_to_parquet.py
-----------------
Converts the large puzzle_features.csv to parquet in chunks,
without loading the whole file into memory.

Usage:
    python csv_to_parquet.py
"""

from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT        = Path(__file__).resolve().parents[1]
OUT_CSV     = ROOT / "data" / "derived" / "puzzle_features.csv"
OUT_PARQUET = ROOT / "data" / "derived" / "puzzle_features.parquet"

CHUNK_SIZE = 200_000  # rows per chunk


def main():
    print(f"Input:  {OUT_CSV}")
    print(f"Output: {OUT_PARQUET}")
    print(f"Chunk:  {CHUNK_SIZE:,} rows")
    print()

    writer = None
    n_total = 0

    reader = pd.read_csv(OUT_CSV, chunksize=CHUNK_SIZE, low_memory=False)

    for i, chunk in enumerate(reader):
        chunk = chunk.fillna(0)
        table = pa.Table.from_pandas(chunk, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(OUT_PARQUET, table.schema)

        writer.write_table(table)
        n_total += len(chunk)

        if (i + 1) % 10 == 0:
            print(f"  written: {n_total:,} rows ...")

    if writer:
        writer.close()

    print(f"\nDone. Total rows: {n_total:,}")
    print(f"Saved: {OUT_PARQUET}")


if __name__ == "__main__":
    main()