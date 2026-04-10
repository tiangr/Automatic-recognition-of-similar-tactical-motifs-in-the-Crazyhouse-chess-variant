# run this once from your src/ directory to convert the existing CSV
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

OUT_CSV = Path(r"C:\Users\tgrum\Desktop\magistrska\crazyhouse\data\derived\puzzle_features_mates.csv")
OUT_PARQUET = Path(r"C:\Users\tgrum\Desktop\magistrska\crazyhouse\data\derived\puzzle_features_mates.parquet")

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
print(f"Done. {n_rows:,} rows written to {OUT_PARQUET}")