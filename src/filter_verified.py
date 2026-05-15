"""
filter_verified.py
------------------
Quick one-pass filter: reads checkmates5_verified.jsonl and keeps only
records where the solution is exactly 5 plies (mate in 3 moves).

Run:
    python filter_verified.py
"""

from pathlib import Path
import orjson

ROOT     = Path(__file__).resolve().parents[1]
IN_FILE  = ROOT / "data" / "derived" / "checkmates5_verified.jsonl"
OUT_FILE = ROOT / "data" / "derived" / "checkmates5_verified_filtered.jsonl"

EXACT_PLIES = 5

n_read = n_kept = n_discarded = 0

with IN_FILE.open("rb") as f_in, OUT_FILE.open("wb") as f_out:
    for raw in f_in:
        raw = raw.strip()
        if not raw:
            continue
        n_read += 1
        try:
            rec = orjson.loads(raw)
        except Exception:
            n_discarded += 1
            continue

        sol = rec.get("solution_uci") or []
        if len(sol) != EXACT_PLIES:
            n_discarded += 1
            continue

        f_out.write(orjson.dumps(rec) + b"\n")
        n_kept += 1

        if n_kept % 1000 == 0:
            print(f"  kept: {n_kept:,}  discarded: {n_discarded:,}  read: {n_read:,}")

print(f"\nDone.")
print(f"  Read      : {n_read:,}")
print(f"  Kept      : {n_kept:,}")
print(f"  Discarded : {n_discarded:,}")
print(f"  Output    : {OUT_FILE}")
