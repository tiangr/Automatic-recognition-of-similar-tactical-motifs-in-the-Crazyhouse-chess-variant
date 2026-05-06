"""
patch_dataset.py
----------------
Patches dataset_full.jsonl by filling in missing/empty text fields
(query_text_static, query_text_dynamic, candidate_text_static, candidate_text_dynamic)
using the corpus_mates.jsonl as the source of truth.

Also fixes the spaced-out text_static bug (e.g. "b r @ h 8" -> "br@h8")
by always overwriting from corpus regardless of existing value.

Usage:
    python patch_dataset.py
    python patch_dataset.py --data dataset_full.jsonl --corpus ../data/derived/corpus_mates.jsonl --out dataset_full_patched.jsonl
"""

import json
import argparse
from pathlib import Path

def fix_spaced(s: str) -> str:
    """Fix spaced-out tokens like 'b r @ h 8' -> 'br@h8'."""
    if s and ' ' in s:
        # Heuristic: if average token length after split is ~1, it's spaced out
        tokens = s.split(' ')
        avg_len = sum(len(t) for t in tokens) / max(len(tokens), 1)
        if avg_len <= 1.5:
            return s.replace(' ', '')
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",   default="dataset_full.jsonl")
    ap.add_argument("--corpus", default="../data/derived/corpus_mates.jsonl")
    ap.add_argument("--out",    default="dataset_full_patched.jsonl")
    args = ap.parse_args()

    # ── 1. Load corpus into memory ────────────────────────────────────────────
    print(f"Loading corpus from {args.corpus} ...")
    corpus = {}
    with open(args.corpus, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            corpus[doc["id"]] = doc
            if (i + 1) % 500_000 == 0:
                print(f"  {i+1:,} docs loaded...")
    print(f"Corpus loaded: {len(corpus):,} docs\n")

    # ── 2. Patch dataset ──────────────────────────────────────────────────────
    data_path = Path(args.data)
    out_path  = Path(args.out)

    n_total = 0
    n_patched_query = 0
    n_patched_candidate = 0
    n_missing_query = 0
    n_missing_candidate = 0

    with open(data_path, encoding="utf-8") as fin, \
         open(out_path,  "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_total += 1

            # Patch query fields
            qid  = row.get("query_id", "")
            qdoc = corpus.get(qid)
            if qdoc:
                row["query_text_static"]  = qdoc.get("text_static",  "")
                row["query_text_dynamic"] = qdoc.get("text_dynamic", "")
                n_patched_query += 1
            else:
                # query was search_by_fen — try to fix spacing at least
                row["query_text_static"]  = fix_spaced(row.get("query_text_static", ""))
                row["query_text_dynamic"] = fix_spaced(row.get("query_text_dynamic", ""))
                n_missing_query += 1

            # Patch candidate fields
            cid  = row.get("candidate_id", "")
            cdoc = corpus.get(cid)
            if cdoc:
                row["candidate_text_static"]  = cdoc.get("text_static",  "")
                row["candidate_text_dynamic"] = cdoc.get("text_dynamic", "")
                n_patched_candidate += 1
            else:
                row["candidate_text_static"]  = fix_spaced(row.get("candidate_text_static", ""))
                row["candidate_text_dynamic"] = fix_spaced(row.get("candidate_text_dynamic", ""))
                n_missing_candidate += 1

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Done. {n_total:,} rows processed -> {args.out}")
    print(f"  Query   patched: {n_patched_query:,}  |  not in corpus: {n_missing_query:,}")
    print(f"  Candid. patched: {n_patched_candidate:,}  |  not in corpus: {n_missing_candidate:,}")

    if n_missing_query > 0:
        print(f"\n  NOTE: {n_missing_query:,} rows had search_by_fen query IDs.")
        print("  These rows will only have static features (no dynamic) during training.")
        print("  Consider re-exporting those queries from real indexed positions.")

if __name__ == "__main__":
    main()
