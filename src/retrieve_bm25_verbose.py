from pathlib import Path
import json
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "derived" / "corpus_tactical.jsonl"
AUG = ROOT / "data" / "derived" / "tactics_1k_aug.jsonl"

#removes final _ply from filenames
def base_game_id(doc_id: str) -> str:
    return doc_id.rsplit("_", 1)[0]

#pretty print {"N": 2, "P": 1} -> "N2 P1"
def pocket_str(p):
    if not p:
        return "-"
    return " ".join([f"{k}{v}" for k,v in sorted(p.items())])

# Load augmented tactic metadata so retrieved docs can be shown with PV, best move, pockets, delta...
def load_aug_index():
    idx = {}
    with AUG.open("r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            _id = r.get("event_id") or (r.get("site") + "_" + str(r.get("ply")))
            idx[_id] = r
    return idx
# Load BM25 documents and tokenize text_all for ranking
def load_docs():
    docs = []
    texts = []
    with CORPUS.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            docs.append(rec)
            texts.append(rec["text_all"].split())
    return docs, texts
# Display one query/hit with key tactical information for manual inspection
def show(label, doc_id, aug):
    pv = aug.get("pv_before") or aug.get("pv_prev") or []
    bm = aug.get("bestmove_before") or aug.get("best_prev")
    played = aug.get("played_move")
    d = aug.get("delta")
    print(f"\n{label} {doc_id}")
    print("ply:", aug.get("ply"), "delta:", d)
    print("played:", played, "best:", bm)
    print("PV:", " ".join(pv[:10]))
    print("PV has drop?:", any("@" in m for m in pv))
    print("pockets W:", pocket_str(aug.get("pockets_white")), "| B:", pocket_str(aug.get("pockets_black")))

def main(query_idx: int = 0, topk: int = 10):
    aug_index = load_aug_index()
    docs, tokenized = load_docs()
    bm25 = BM25Okapi(tokenized)
    # Pick one corpus document as the query
    query = docs[query_idx]
    q_id = query["id"]
    q_base = base_game_id(q_id)
    # Rank all corpus docs against the query using BM25 over text_all
    scores = bm25.get_scores(query["text_all"].split())
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    show("QUERY", q_id, aug_index[q_id])

    shown = 0
    for idx in ranked:
        cand = docs[idx]
        if cand["id"] == q_id:#skip the same document
            continue
        if base_game_id(cand["id"]) == q_base:
            continue

        shown += 1
        show(f"HIT {shown} score={scores[idx]:.2f}", cand["id"], aug_index[cand["id"]])
        if shown >= topk:
            break

if __name__ == "__main__":
    main(query_idx=0, topk=5)
