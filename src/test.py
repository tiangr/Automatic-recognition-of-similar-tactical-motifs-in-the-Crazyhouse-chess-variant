import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "derived" / "corpus_full.jsonl"

with CORPUS.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= 20:
            break
        rec = json.loads(line)
        print(f"[{i}] id={rec.get('id')} ply={rec.get('ply')} pockets_w={rec.get('pockets_white')} pockets_b={rec.get('pockets_black')}")
        print(f"     text_dynamic={rec.get('text_dynamic','')[:120]}")
        print()