"""
build_corpus_full.py
--------------------
Builds the BM25 corpus from tactics_full_tactical.jsonl
(the full multi-month filtered dataset).

Identical logic to build_corpus.py — only the input/output paths differ.
"""

from pathlib import Path
import json

from encode   import encode_static, encode_pockets, encode_dynamic   # noqa: F401 (kept for reference)
from encodev2 import encode_dynamic_v2

ROOT = Path(__file__).resolve().parents[1]
INP  = ROOT / "data" / "derived" / "tactics_full_tactical.jsonl"
OUT  = ROOT / "data" / "derived" / "corpus_full.jsonl"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0

    with INP.open("r", encoding="utf-8", errors="replace") as f, \
         OUT.open("w", encoding="utf-8") as out:

        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            board_fen = rec.get("board_fen")
            pw = rec.get("pockets_white", {})
            pb = rec.get("pockets_black", {})
            if not board_fen:
                continue

            static  = encode_static(board_fen)
            pocket  = encode_pockets(pw, pb)
            dynamic = encode_dynamic_v2(rec)

            rec_out = {
                "id":            rec.get("event_id") or (rec.get("site") + "_" + str(rec.get("ply"))),
                "site":          rec.get("site"),
                "ply":           rec.get("ply"),
                "board_fen":     board_fen,
                "pockets_white": pw,
                "pockets_black": pb,
                # same weighting as the tuned baseline: dynamic×3 + pocket×2 + static
                "text_static":   " ".join(static + pocket),
                "text_dynamic":  " ".join(dynamic),
                "text_all":      " ".join(dynamic + dynamic + dynamic + pocket + pocket + static),
            }
            out.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            n += 1

            if n % 10_000 == 0:
                print(f"  corpus docs written: {n} …")

    print(f"DONE  corpus docs: {n}  ->  {OUT}")


if __name__ == "__main__":
    main()
