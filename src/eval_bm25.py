"""
eval_bm25_simple.py
-------------------
Evaluates BM25 retrieval quality using only corpus_mates.jsonl —
no aug file needed.

Relevance label: two positions are "relevant" if they share the same
tactical signature, defined as a tuple of:
  - mate_before  (mate-in-N value)
  - first_action (first move type: dropCheck, capCheck, drop, capture, ...)
  - mating_piece (which piece delivers the final mate, if encoded)
  - has_drop     (whether the solution involves any drop)

This is a proxy for expert similarity — positions with the same mate depth,
same first-move type, and same mating piece are likely to be tactically similar.
Multiple label granularities are evaluated so you can see how BM25 performs
at each level of strictness.

Usage:
    python eval_bm25_simple.py
    python eval_bm25_simple.py --field text_all --topk 10 --max_queries 500
"""

from pathlib import Path
import json
import argparse
from collections import defaultdict
import random

from rank_bm25 import BM25Okapi

ROOT   = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "derived" / "corpus_mates.jsonl"


def base_game_id(doc_id: str) -> str:
    return doc_id.rsplit("_", 1)[0]


def extract_label(doc: dict) -> dict:
    """
    Extract relevance labels directly from corpus tokens — no aug file needed.
    All information comes from text_dynamic which was built by encodev2.py.
    """
    toks = set(doc.get("text_dynamic", "").split())

    # mate-in-N from corpus field
    mate = doc.get("mate_before")

    # first action from dyn:first:X token
    first_action = None
    for t in toks:
        if t.startswith("dyn:first:"):
            first_action = t.split(":", 2)[2]
            break

    # mating piece from dyn:matingPiece:X token (v3 new)
    mating_piece = None
    for t in toks:
        if t.startswith("dyn:matingPiece:"):
            mating_piece = t.split("dyn:matingPiece:")[1]
            break

    # has drop
    has_drop = "dyn:hasDrop" in toks

    # PV length bucket
    pv_len = None
    for t in toks:
        if t.startswith("dyn:pvLen:"):
            pv_len = t.split(":")[2]
            break

    # drop piece set (e.g. "NQ", "R")
    drop_pieces = None
    for t in toks:
        if t.startswith("dyn:dropPieces:"):
            drop_pieces = t.split("dyn:dropPieces:")[1]
            break

    return {
        "mate":         mate,
        "first_action": first_action,
        "mating_piece": mating_piece,
        "has_drop":     has_drop,
        "pv_len":       pv_len,
        "drop_pieces":  drop_pieces,
    }


def make_label_coarse(lbl: dict):
    """Coarse: mate depth + has_drop. Many matches expected."""
    if lbl["mate"] is None or lbl["first_action"] is None:
        return None
    return (lbl["mate"], lbl["has_drop"])


def make_label_medium(lbl: dict):
    """Medium: mate depth + first action type."""
    if lbl["mate"] is None or lbl["first_action"] is None:
        return None
    return (lbl["mate"], lbl["first_action"])


def make_label_fine(lbl: dict):
    """Fine: mate depth + first action + mating piece."""
    if lbl["mate"] is None or lbl["first_action"] is None:
        return None
    return (lbl["mate"], lbl["first_action"], lbl["mating_piece"])


def make_label_finest(lbl: dict):
    """Finest: mate depth + first action + mating piece + drop pieces used."""
    if lbl["mate"] is None or lbl["first_action"] is None:
        return None
    return (lbl["mate"], lbl["first_action"], lbl["mating_piece"], lbl["drop_pieces"])


LABEL_FNS = {
    "coarse":  make_label_coarse,
    "medium":  make_label_medium,
    "fine":    make_label_fine,
    "finest":  make_label_finest,
}


def load_corpus(field: str, max_docs: int | None = None):
    docs = []
    tokenized = []
    with CORPUS.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            docs.append(rec)
            tokenized.append(rec.get(field, "").split())
            if max_docs and len(docs) >= max_docs:
                break
    return docs, tokenized


def evaluate(docs, bm25, field: str, topk: int, max_queries: int,
             label_fn, label_name: str, seed: int = 42):
    """Run retrieval evaluation for one label granularity."""

    # Build label index
    id_to_label = {}
    label_to_ids = defaultdict(set)

    for d in docs:
        lbl_raw = extract_label(d)
        lbl = label_fn(lbl_raw)
        if lbl is None:
            continue
        id_to_label[d["id"]] = lbl
        label_to_ids[lbl].add(d["id"])

    # Sample query docs
    rng = random.Random(seed)
    candidates = [d for d in docs if d["id"] in id_to_label]
    rng.shuffle(candidates)
    query_docs = candidates[:max_queries]

    n_used = 0
    recall_hits = 0
    mrr_sum = 0.0
    p_at_k_sum = 0.0

    for qdoc in query_docs:
        qid = qdoc["id"]
        qlab = id_to_label.get(qid)
        if qlab is None:
            continue

        qbase = base_game_id(qid)
        rel = {
            rid for rid in label_to_ids[qlab]
            if rid != qid and base_game_id(rid) != qbase
        }
        if not rel:
            continue

        scores = bm25.get_scores(qdoc.get(field, "").split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        top = []
        for idx in ranked:
            cid = docs[idx]["id"]
            if cid == qid:
                continue
            if base_game_id(cid) == qbase:
                continue
            top.append(cid)
            if len(top) >= topk:
                break

        n_used += 1

        # Recall@k: was any relevant doc in top-k?
        hit_rank = None
        hits_in_top = 0
        for r, cid in enumerate(top, start=1):
            if cid in rel:
                hits_in_top += 1
                if hit_rank is None:
                    hit_rank = r

        if hit_rank is not None:
            recall_hits += 1
            mrr_sum += 1.0 / hit_rank

        p_at_k_sum += hits_in_top / topk

    if n_used == 0:
        return None

    return {
        "label":       label_name,
        "n_queries":   n_used,
        "recall_at_k": recall_hits / n_used,
        "mrr_at_k":    mrr_sum / n_used,
        "p_at_k":      p_at_k_sum / n_used,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="text_all",
                    choices=["text_dynamic", "text_static", "text_all",
                             "text_dynamic_general", "text_dynamic_solution"])
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max_queries", type=int, default=500)
    ap.add_argument("--max_docs", type=int, default=None,
                    help="Limit corpus size for faster testing (default: all)")
    ap.add_argument("--labels", default="coarse,medium,fine,finest",
                    help="Comma-separated label granularities to evaluate")
    args = ap.parse_args()

    label_names = [l.strip() for l in args.labels.split(",")]

    print(f"Loading corpus from {CORPUS} ...")
    docs, tokenized = load_corpus(args.field, args.max_docs)
    print(f"Loaded {len(docs):,} docs")

    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized)
    print("Done.\n")

    print(f"{'Label':<10} {'Queries':>8} {'Recall@'+str(args.topk):>12} {'MRR@'+str(args.topk):>10} {'P@'+str(args.topk):>8}")
    print("-" * 55)

    results = []
    for lname in label_names:
        fn = LABEL_FNS.get(lname)
        if fn is None:
            print(f"Unknown label: {lname}, skipping")
            continue

        res = evaluate(docs, bm25, args.field, args.topk,
                       args.max_queries, fn, lname)
        if res is None:
            print(f"{lname:<10} {'N/A — no usable queries'}")
            continue

        results.append(res)
        print(
            f"{res['label']:<10} "
            f"{res['n_queries']:>8,} "
            f"{res['recall_at_k']:>12.3f} "
            f"{res['mrr_at_k']:>10.3f} "
            f"{res['p_at_k']:>8.4f}"
        )

    print()
    print(f"FIELD={args.field}  TOPK={args.topk}  MAX_QUERIES={args.max_queries}")
    if args.max_docs:
        print(f"(corpus limited to {args.max_docs:,} docs for speed)")

    # Also run a quick comparison across fields if using default settings
    if args.field == "text_all" and not args.max_docs and len(label_names) > 0:
        print("\n--- Field comparison (medium label, same query sample) ---")
        medium_fn = LABEL_FNS["medium"]
        print(f"{'Field':<28} {'Recall@'+str(args.topk):>12} {'MRR@'+str(args.topk):>10}")
        print("-" * 52)
        for fname in ["text_dynamic_general", "text_dynamic_solution",
                      "text_dynamic", "text_static", "text_all"]:
            _, tok2 = load_corpus(fname, args.max_docs)
            bm2 = BM25Okapi(tok2)
            r2 = evaluate(docs, bm2, fname, args.topk,
                          args.max_queries, medium_fn, fname)
            if r2:
                print(f"  {fname:<26} {r2['recall_at_k']:>12.3f} {r2['mrr_at_k']:>10.3f}")


if __name__ == "__main__":
    main()