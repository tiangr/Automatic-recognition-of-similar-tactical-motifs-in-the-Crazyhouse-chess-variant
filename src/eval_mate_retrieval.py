from pathlib import Path
import json
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "derived" / "corpus_tactical_with_queryply.jsonl"
AUG = ROOT / "data" / "derived" / "tactics_1k_aug.jsonl"

FIELD = "text_static"
TOPK = 10

def load_aug_index():
    idx = {}
    with AUG.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            _id = r.get("event_id") or (r.get("site") + "_" + str(r.get("ply")))
            idx[_id] = r
    return idx

def load_docs():
    docs, texts = [], []
    with CORPUS.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            docs.append(rec)
            texts.append(rec[FIELD].split())
    return docs, texts

def main():
    aug = load_aug_index()
    docs, tokenized = load_docs()
    bm25 = BM25Okapi(tokenized)

    # only evaluate docs that exist in AUG and have mate label 2/3
    q_indices = []
    for i, d in enumerate(docs):
        rid = d["id"]
        r = aug.get(rid)
        if not r:
            continue
        if r.get("mate_before") in (2, 3):
            q_indices.append(i)

    hits = 0
    rr_sum = 0.0
    prec_sum = 0.0
    n = 0

    for qi in q_indices:
        qid = docs[qi]["id"]
        q_label = aug[qid]["mate_before"]

        scores = bm25.get_scores(docs[qi][FIELD].split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        rel_in_topk = 0
        first_rank = None
        shown = 0
        for idx in ranked:
            if idx == qi:
                continue
            cid = docs[idx]["id"]
            c = aug.get(cid)
            if not c:
                continue

            shown += 1
            if c.get("mate_before") == q_label:
                rel_in_topk += 1
                if first_rank is None:
                    first_rank = shown
            if shown >= TOPK:
                break

        n += 1
        if rel_in_topk > 0:
            hits += 1
            rr_sum += 1.0 / first_rank
        prec_sum += rel_in_topk / TOPK

    print("FIELD:", FIELD)
    print("queries:", n)
    if n:
        print(f"Recall@{TOPK}:", hits / n)
        print("MRR:", rr_sum / n)
        print(f"Precision@{TOPK}:", prec_sum / n)

if __name__ == "__main__":
    main()