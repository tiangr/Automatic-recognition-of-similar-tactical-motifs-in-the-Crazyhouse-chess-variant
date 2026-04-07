import json
from pathlib import Path

ROOT = Path(r"C:\Users\tgrum\Desktop\magistrska\crazyhouse")
INP  = ROOT / "data" / "derived" / "corpus_mates.jsonl"
OUT  = ROOT / "data" / "derived" / "mates_meta.jsonl"

n = 0
with INP.open("r", encoding="utf-8") as f, OUT.open("w", encoding="utf-8") as out:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        meta = {
            "event_id":      rec.get("id"),
            "site":          rec.get("site"),
            "ply":           rec.get("ply"),
            "board_fen":     rec.get("board_fen"),
            "pockets_white": rec.get("pockets_white"),
            "pockets_black": rec.get("pockets_black"),
        }
        out.write(json.dumps(meta) + "\n")
        n += 1
        if n % 500_000 == 0:
            print(f"  processed: {n:,} ...")

print(f"Done. Total: {n:,}")