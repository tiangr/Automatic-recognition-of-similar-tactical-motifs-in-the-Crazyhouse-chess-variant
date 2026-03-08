from pathlib import Path
import json
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]

CORPUS = ROOT / "data" / "derived" / "corpus_tactical_with_queryply.jsonl"
AUG = ROOT / "data" / "derived" / "tactics_1k_aug.jsonl"

# choose what to retrieve on:
FIELD = "text_all"       # try: "text_dynamic" or "text_static" "text_all"
TOPK = 5
QUERY_IDX = 0

def base_game_id(doc_id: str) -> str:
    return doc_id.rsplit("_", 1)[0]

def pocket_nonempty(p):
    return bool(p) and sum(p.values()) > 0

def is_tactical(aug_rec) -> bool:
    pv = aug_rec.get("pv_before") or aug_rec.get("pv_prev") or []
    has_drop = any("@" in m for m in pv[:10])
    has_pockets = pocket_nonempty(aug_rec.get("pockets_white")) or pocket_nonempty(aug_rec.get("pockets_black"))
    return has_drop or has_pockets

def pocket_str(p):
    if not p:
        return "-"
    return " ".join([f"{k}{v}" for k, v in sorted(p.items())])

def load_aug_index():
    idx = {}
    with AUG.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            _id = r.get("event_id") or (r.get("site") + "_" + str(r.get("ply")))
            idx[_id] = r
    return idx

def load_docs():
    docs = []
    texts = []
    with CORPUS.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            docs.append(rec)
            texts.append(rec[FIELD].split())
    return docs, texts

def show(prefix: str, doc_id: str, rec: dict):
    print(f"\n{prefix} {doc_id}")
    print("ply:", rec.get("ply"), "delta:", rec.get("delta"))
    print("mate_before:", rec.get("mate_before"), "cp_before:", rec.get("cp_before"))
    print("played:", rec.get("played_move"), "best:", rec.get("bestmove_before"))
    pv = rec.get("pv_before") or []
    print("PV:", " ".join(pv[:10]))
    print("PV has drop?:", any("@" in m for m in pv))
    pw = rec.get("pockets_white") or {}
    pb = rec.get("pockets_black") or {}
    pw_s = " ".join([f"{k}{v}" for k, v in sorted(pw.items())]) or "-"
    pb_s = " ".join([f"{k}{v}" for k, v in sorted(pb.items())]) or "-"
    print("pockets W:", pw_s, "| B:", pb_s)

def main(query_idx: int = 0, topk: int = 10):
    aug_index = load_aug_index()
    docs, tokenized = load_docs()
    print("Loaded docs:", len(docs), "from", CORPUS)

    bm25 = BM25Okapi(tokenized)

    query = docs[query_idx]
    q_id = query["id"]
    q_base = base_game_id(q_id)

    # query tokens depend on FIELD
    q_tokens = query[FIELD].split()
    scores = bm25.get_scores(q_tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    print(f"USING FIELD: {FIELD}")
    show("QUERY", q_id, aug_index[q_id])

    shown = 0
    for idx in ranked:
        cand = docs[idx]
        c_id = cand["id"]

        if c_id == q_id:
            continue
        if base_game_id(c_id) == q_base:
            continue

        # tactical filter (skip boring hits)
        cand_aug = aug_index.get(c_id)

        shown += 1
        if cand_aug:
            show(f"HIT {shown} score={scores[idx]:.2f}", c_id, cand_aug)
        else:
            # synth docs are not in AUG -> print minimal info
            print(f"\nHIT {shown} score={scores[idx]:.2f} {c_id}")
            print("ply:", cand.get("ply"), "(no AUG record - likely SYN)")
        if shown >= topk:
            break

        if shown >= topk:
            break

if __name__ == "__main__":
    main(query_idx=QUERY_IDX, topk=TOPK)
