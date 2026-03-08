from pathlib import Path
import json
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "derived" / "corpus_1k.jsonl"

def load_docs():
    docs = []
    texts = []
    with CORPUS.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            docs.append(rec)
            texts.append(rec["text_all"].split())
    return docs, texts

def base_game_id(doc_id: str) -> str:
    # "https://lichess.org/N7w3FJpI_5" -> "https://lichess.org/N7w3FJpI"
    return doc_id.rsplit("_", 1)[0]

def main(query_idx: int = 0, topk: int = 10):
    docs, tokenized = load_docs()
    bm25 = BM25Okapi(tokenized)

    query = docs[query_idx]
    q_tokens = query["text_all"].split()
    scores = bm25.get_scores(q_tokens)

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    q_base = base_game_id(query["id"])

    print("QUERY:", query["id"], query["site"], query["ply"])
    print("Skipping same game as:", q_base)
    print("\nTOP", topk, "(different games):")

    shown = 0
    for idx in ranked:
        if docs[idx]["id"] == query["id"]:
            continue

        # skip same lichess game
        if base_game_id(docs[idx]["id"]) == q_base:
            continue

        print(shown + 1, scores[idx], docs[idx]["id"], docs[idx]["site"], docs[idx]["ply"])
        shown += 1
        if shown >= topk:
            break

if __name__ == "__main__":
    # change query index here:
    main(query_idx=0, topk=10)
