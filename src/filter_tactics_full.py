"""
filter_tactics_full.py
----------------------
Filters tactics_full_aug.jsonl to keep only Crazyhouse-relevant positions
(drop in PV/bestmove, or non-empty pocket on either side).

Identical logic to filter_tactics.py — only the input/output paths differ.
"""

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
INP  = ROOT / "data" / "derived" / "tactics_full_aug.jsonl"
OUT  = ROOT / "data" / "derived" / "tactics_full_tactical.jsonl"


def pockets_nonempty(p):
    return bool(p) and sum(int(v) for v in p.values()) > 0


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    total = kept = kept_drop = kept_pockets = 0

    with INP.open("r", encoding="utf-8", errors="replace") as f, \
         OUT.open("w", encoding="utf-8") as out:

        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1

            try:
                r = json.loads(line)
            except Exception:
                continue

            pv = r.get("pv_before") or r.get("pv_prev") or []
            bm = r.get("bestmove_before") or r.get("best_prev") or ""

            has_drop    = any("@" in m for m in pv[:10]) or ("@" in bm)
            has_pockets = (
                pockets_nonempty(r.get("pockets_white", {})) or
                pockets_nonempty(r.get("pockets_black", {}))
            )

            if has_drop or has_pockets:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept += 1
                if has_drop:
                    kept_drop += 1
                if has_pockets:
                    kept_pockets += 1

            if total % 10_000 == 0:
                print(f"  processed: {total}, kept: {kept} …")

    print("DONE")
    print(f"  total:          {total}")
    print(f"  kept:           {kept}")
    print(f"  kept_by_drop:   {kept_drop}")
    print(f"  kept_by_pocket: {kept_pockets}")
    print(f"  out:            {OUT}")


if __name__ == "__main__":
    main()
