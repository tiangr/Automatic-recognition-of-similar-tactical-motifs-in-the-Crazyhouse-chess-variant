"""
diagnose_smother.py
===================
Answers the mentor's two questions without needing Lucene running:

  (A) WHERE ARE THE SMOTHERS?  A census of mating pictures over the corpus:
      how many smothered mates exist, how many distinct pictures, and how
      concentrated the new picture key is. Confirms the motif is common and
      now has a shared retrieval key.

  (B) WHERE ARE THERE NO SIMILAR GAMES?  For each labelled query in the
      exports, the count of expert "similar" candidates; it lists the queries
      with zero genuine similars and flags which of those are smothers — the
      exact cases the mentor noticed.

Run:
  python diagnose_smother.py --corpus ../data/derived/corpus_checkmates5.jsonl
  python diagnose_smother.py --exports ../data/exports/v1 ../data/exports/v2
  python diagnose_smother.py --corpus <corpus.jsonl> --exports <dir> [<dir> ...]

Needs mate_picture.py beside this file (and python-chess).
"""
import argparse, glob, json, os, sys
from collections import Counter, defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mate_picture import mate_picture, mate_picture_for_rec, get_picture


def census(corpus_path, limit=None):
    n = smothers = 0
    pics = Counter(); regions = Counter(); checkers = Counter()
    with open(corpus_path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # prefer stored tokens if the corpus was already rebuilt with them
            p = get_picture({"text_dynamic": rec.get("text_dynamic", ""),
                             "fen": rec.get("fen", "") or rec.get("board_fen", ""),
                             "solution_uci": rec.get("solution_uci"),
                             "turn": rec.get("turn")})
            if not p or not p.get("picture_canon"):
                p = mate_picture_for_rec(rec)
            if not p.get("picture_canon"):
                continue
            n += 1
            pics[p["picture_canon"]] += 1
            regions[p["region"]] += 1
            checkers[p["checker"]] += 1
            if p["smother"]:
                smothers += 1
    print(f"\n=== CORPUS MATE-PICTURE CENSUS ({corpus_path}) ===")
    print(f"positions with a readable mate picture : {n}")
    print(f"smothered mates                        : {smothers} ({100*smothers/max(n,1):.1f}%)")
    print(f"distinct pictures                      : {len(pics)}")
    print(f"region split                           : {dict(regions)}")
    print(f"checker split                          : {dict(checkers.most_common(6))}")
    print("top 8 pictures (the clusters BM25 can now key on):")
    for pic, c in pics.most_common(8):
        print(f"    {c:5d}  {pic}")


def no_similar(export_dirs):
    files = []
    for d in export_dirs:
        files += ([d] if d.endswith(".jsonl")
                  else sorted(glob.glob(os.path.join(d, "*.jsonl"))))
    if not files:
        raise SystemExit(f"no export files under {export_dirs}")
    queries = defaultdict(list)
    for fp in files:
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                queries[r.get("query_id") or fp].append(r)

    print(f"\n=== PER-QUERY SIMILAR COUNTS ({len(queries)} queries) ===")
    zero = []
    for qid, rows in queries.items():
        r0 = rows[0]
        sims = [r for r in rows
                if str(r.get("label", "")).lower() in ("similar", "1", "1.0")
                and r.get("candidate_id") != qid]            # genuine, excl. self
        # query mate picture (needs the query's own solution line)
        qpic = mate_picture(r0.get("query_fen", ""),
                            r0.get("query_solution_uci"),
                            r0.get("query_turn"))
        tag = "SMOTHER" if qpic.get("smother") else qpic.get("region", "")
        if len(sims) == 0:
            zero.append((qid, tag, qpic.get("picture_canon", "")))
        print(f"  {str(qid)[:48]:48s} similar={len(sims):2d}  {tag:8s} {qpic.get('picture_canon','')}")
    print(f"\nQUERIES WITH ZERO GENUINE SIMILAR GAMES: {len(zero)}")
    for qid, tag, pic in zero:
        flag = "  <-- smother the program missed" if tag == "SMOTHER" else ""
        print(f"    {str(qid)[:48]:48s} {tag:8s} {pic}{flag}")
    if not any(t == "SMOTHER" for _, t, _ in zero):
        print("    (no smother among the zero-similar queries in THIS export; the "
              "census above shows how many smothers exist corpus-wide to surface)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--exports", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap corpus rows (quick run)")
    a = ap.parse_args()
    if not a.corpus and not a.exports:
        ap.error("give --corpus and/or --exports")
    if a.corpus:
        census(a.corpus, a.limit)
    if a.exports:
        no_similar(a.exports)
