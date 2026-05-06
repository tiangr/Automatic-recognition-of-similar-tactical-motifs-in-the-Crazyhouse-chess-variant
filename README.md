# Chess Tactical Similarity Retrieval

Short intro TBA

---

## Overview

Each puzzle is encoded as a text document from two types of features:

- **Static features** — piece counts, mobility, connectivity (attack/defense/x-ray relations), pawn structure
- **Dynamic features** — solution move properties: captures, checks, drops, promotions, mating piece, piece-move sequences, attack relations per move

Documents are indexed in Apache Lucene and retrieved via BM25 (for now). A supervised classifier (XGBoost/Decision Tree) will be trained on human-labeled pairs to re-rank results.

---

## Repository Structure

```
src/
├── encode.py                     # Static feature encoder (pieces, connectivity, pawn structure). Used by buil corpus script.
├── encodev2.py                   # Dynamic feature encoder (solution move tokens). Used by buil corpus script.
├── build_corpus_full.py          # Builds corpus_mates.jsonl from raw PGN/tactics data
├── CrazyhouseLuceneServer.java   # Lucene BM25 search server
├── extract_features_crazyhouse.py# Parses corpus → puzzle_features_mates.parquet
├── build_feature_cache.py        # Builds feat_matrix.npy + feat_index.json for fast lookup
├── train_similarity.py           # Trains similarity classifier on human-labeled pairs
├── train_similarity_v2.py        # Extended trainer with model comparison plots
├── app.py                        # Flask web app (search UI + export labeled pairs)
├── eval_bm25.py                  # BM25 retrieval evaluation (Recall@K, MRR@K). Not needed for main pipeline.
├── tactic_miner_full.py          # Mine tactics from Lichess PGN archives
├── filter_tactics_full.py        # Filter and clean mined tactics
├── augment_tactics.py_full.py    # Data augmentation for tactics
└── ...

data/
├── derived/
│   ├── corpus_mates.jsonl        # Encoded puzzle documents (Lucene input)
│   ├── lucene_index/             # Lucene index (auto-built on server start)
│   ├── puzzle_features_mates.parquet  # Feature matrix per puzzle
│   ├── feat_matrix.npy           # Cached feature matrix
│   └── feat_index.json           # Puzzle ID → row mapping
└── models/
    └── similarity_classifier.pkl # Trained classifier from unsupervised method (Unused)

lucene/
└── lucene-core-9.12.3.jar        # Lucene JARs (one level above src/)
    lucene-analysis-common-9.12.3.jar
    lucene-queryparser-9.12.3.jar
```

---

## Setup

**Requirements:** Python 3.10+, Java 11+

```bash
pip install -r requirements.txt
```

---

## Pipeline

### 1. Build the corpus

```bash
python build_corpus_full.py
# Output: data/derived/corpus_mates.jsonl
```

### 2. Extract features

```bash
python extract_features_crazyhouse.py
# Output: puzzle_features_mates.parquet

python build_feature_cache.py
# Output: feat_matrix.npy, feat_index.json
```

### 3. Compile and start the Lucene server

```powershell
# From src/
javac -cp "..\lucene\lucene-core-9.12.3.jar;..\lucene\lucene-analysis-common-9.12.3.jar;..\lucene\lucene-queryparser-9.12.3.jar" CrazyhouseLuceneServer.java

java -cp ".;..\lucene\lucene-core-9.12.3.jar;..\lucene\lucene-analysis-common-9.12.3.jar;..\lucene\lucene-queryparser-9.12.3.jar" CrazyhouseLuceneServer
# Listens on http://localhost:8983, auto-indexes corpus_mates.jsonl
```

### 4. Start the web app

```bash
python app.py
# Open http://localhost:5000
```

---

## How human labels were done


1. In the web app, search for a query position, mark similar results, click **Export** → downloads `ch_export_XXXXX.jsonl`
2. Repeat for ~70 query positions
3. Combine exports and train:

```powershell
Get-Content ch_export_*.jsonl | Set-Content dataset_full.jsonl
python train_similarity.py --data dataset_full.jsonl --out output
```

Output: `output/` with confusion matrices, SHAP plots, feature importance, and `similarity_classifier.pkl`.

---

## Re-indexing Lucene

Required after any change to `build_corpus_full.py` or `CrazyhouseLuceneServer.java`:

```powershell
# 1. Stop the server (Ctrl+C)
# 2. Delete old index
rmdir /s /q ..\data\derived\lucene_index
# 3. Recompile (if Java changed)
javac -cp "..\lucene\..." CrazyhouseLuceneServer.java
# 4. Restart (auto-rebuilds index)
java -cp "..." CrazyhouseLuceneServer
```

---


## Feature Summary

1,123 features per puzzle pair across five categories:

| Category   | Features | AUC (alone) |
|------------|:--------:|:-----------:|
| Static     | 105      | 0.554       |
| Dynamic    | 856      | 0.901       |
| Theme      | 149      | 0.774       |
| Pair-level | 4        | 0.759       |
| Difference | 355      | 0.900       |
| **All**    | **1,123**| **0.904**   |

See `feature_descriptions_summary_2_.md` for the full feature reference.

---

## Known Gaps

- **Re-ranking not wired in** — `app.py` returns raw BM25 results only; the trained classifier in `feat_matrix.npy` is not yet applied as a re-ranking step.
- **Mobility tokens** — `mob_sum`/`avg_piece_mobility` always 0 (distance-weighted ray casting not implemented; negligible AUC impact).
- `!F_draw` not implemented (irrelevant for a mate-only corpus).

---

## References

- Miha Bizjak, *Automatic Recognition of Related Chess Motifs* (thesis)
- [Lichess open database](https://database.lichess.org/)
- [Apache Lucene 9.12](https://lucene.apache.org/)