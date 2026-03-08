from pathlib import Path
import json
import argparse
from collections import defaultdict

from rank_bm25 import BM25Okapi
from encodev2 import encode_dynamic_v2


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "derived" / "corpus_tactical.jsonl"
AUG = ROOT / "data" / "derived" / "tactics_1k_aug.jsonl"


def base_game_id(doc_id: str) -> str:
    return doc_id.rsplit("_", 1)[0]


def load_aug_index():
    idx = {}
    with AUG.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            _id = r.get("event_id") or f"{r.get('site')}_{r.get('ply')}"
            idx[_id] = r
    return idx


def first_action_from_aug(aug_rec) -> str | None:
    toks = encode_dynamic_v2(aug_rec)
    for t in toks:
        if t.startswith("dyn:first:"):
            return t.split(":", 2)[2]
    return None


def label_for_doc(aug_rec) -> tuple[str, str] | None:
    bm = aug_rec.get("bestmove_before") or aug_rec.get("best_prev")
    fa = first_action_from_aug(aug_rec)
    if not bm or bm == "(none)" or not fa:
        return None
    return (bm, fa)


def load_corpus(field: str):
    docs = []
    tokenized = []
    with CORPUS.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            docs.append(rec)
            tokenized.append(rec[field].split())
    return docs, tokenized


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="text_dynamic", choices=["text_dynamic", "text_static", "text_all"])
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--max_queries", type=int, default=300)
    args = ap.parse_args()

    aug = load_aug_index()
    docs, tokenized = load_corpus(args.field)
    bm25 = BM25Okapi(tokenized)

    id_to_label = {}
    label_to_ids = defaultdict(set)

    for d in docs:
        _id = d["id"]
        a = aug.get(_id)
        if not a:
            continue
        lab = label_for_doc(a)
        if not lab:
            continue
        id_to_label[_id] = lab
        label_to_ids[lab].add(_id)

    n_used = 0
    recall_hits = 0
    mrr_sum = 0.0

    for qdoc in docs[: min(args.max_queries, len(docs))]:
        qid = qdoc["id"]
        qlab = id_to_label.get(qid)
        if not qlab:
            continue

        qbase = base_game_id(qid)
        rel = {rid for rid in label_to_ids[qlab] if rid != qid and base_game_id(rid) != qbase}
        if not rel:
            continue

        scores = bm25.get_scores(qdoc[args.field].split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        top = []
        for idx in ranked:
            cid = docs[idx]["id"]
            if cid == qid:
                continue
            if base_game_id(cid) == qbase:
                continue
            top.append(cid)
            if len(top) >= args.topk:
                break

        n_used += 1

        hit_rank = None
        for r, cid in enumerate(top, start=1):
            if cid in rel:
                hit_rank = r
                break

        if hit_rank is not None:
            recall_hits += 1
            mrr_sum += 1.0 / hit_rank

    if n_used == 0:
        print("No usable queries found (labels/relevants too sparse). Try increasing --max_queries or loosening label.")
        return

    recall_at_k = recall_hits / n_used
    mrr_at_k = mrr_sum / n_used

    print(f"FIELD={args.field}  TOPK={args.topk}  QUERIES_USED={n_used}")
    print(f"Recall@{args.topk}: {recall_at_k:.3f}")
    print(f"MRR@{args.topk}:    {mrr_at_k:.3f}")


if __name__ == "__main__":
    main()