"""
train_classifier_weak.py
------------------------
Trains a similarity classifier using cluster-based weak labels.

Weak label definition:
  - Same cluster  → label 1 (similar)
  - Diff cluster  → label 0 (different)

The silhouette score was 0.058, so these labels are noisy.
That's fine — the classifier is a "rough first pass" to be refined
later with human-annotated labels from the web app.

Pipeline:
  1. Load puzzle_features_mates.parquet       (feature matrix, 178 features)
  2. Load puzzle_features_mates_clustered.parquet  (puzzle_id → cluster)
  3. Sample pairs: N_SAME same-cluster pairs + N_DIFF different-cluster pairs
  4. Build pair feature vector = |q_feat - c_feat|  (absolute difference)
     Also append q_feat and c_feat as raw features (mentor's approach)
  5. Train RandomForest + LogisticRegression
  6. Cross-validate, print AUC
  7. Save model to models/similarity_classifier.pkl

The saved model can be used by app.py to re-rank BM25 results.

Usage:
    python train_classifier_weak.py
    python train_classifier_weak.py --n_pairs 50000 --n_clusters_skip 4
"""

from pathlib import Path
import argparse
import random
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

ROOT       = Path(__file__).resolve().parents[1]
FEAT_PATH  = ROOT / "data" / "derived" / "puzzle_features_mates.parquet"
CLUS_PATH  = ROOT / "data" / "derived" / "puzzle_features_mates_clustered.parquet"
MODEL_DIR  = ROOT / "data" / "models"
MODEL_PATH = MODEL_DIR / "similarity_classifier.pkl"

# Metadata columns — not used as features
META_COLS = [
    "puzzle_id", "site", "ply", "board_fen", "turn",
    "pockets_white_raw", "pockets_black_raw",
]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_SAME          = 30_000   # same-cluster pairs to sample
N_DIFF          = 30_000   # different-cluster pairs to sample
RANDOM_STATE    = 42
CV_FOLDS        = 5


def get_feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in META_COLS
            and pd.api.types.is_numeric_dtype(df[c])]


def sample_pairs(df: pd.DataFrame, cluster_col: str,
                 n_same: int, n_diff: int,
                 skip_clusters: set | None = None,
                 rng: random.Random = None) -> list[tuple]:
    """
    Sample (idx_a, idx_b, label) pairs.
    skip_clusters: cluster ids to exclude from positive pairs
                   (e.g. the large no-drop cluster which is too different
                    from everything else to be a useful positive signal)
    """
    if rng is None:
        rng = random.Random(RANDOM_STATE)

    cluster_to_idxs: dict[int, list[int]] = {}
    for i, c in enumerate(df[cluster_col]):
        cluster_to_idxs.setdefault(c, []).append(i)

    all_clusters = list(cluster_to_idxs.keys())
    pos_clusters = [c for c in all_clusters
                    if (skip_clusters is None or c not in skip_clusters)
                    and len(cluster_to_idxs[c]) >= 2]

    pairs = []

    # Positive pairs (same cluster)
    attempts = 0
    while len(pairs) < n_same and attempts < n_same * 10:
        attempts += 1
        c = rng.choice(pos_clusters)
        idxs = cluster_to_idxs[c]
        a, b = rng.sample(idxs, 2)
        pairs.append((a, b, 1))

    # Negative pairs (different clusters)
    attempts = 0
    while len(pairs) < n_same + n_diff and attempts < n_diff * 10:
        attempts += 1
        c1, c2 = rng.sample(all_clusters, 2)
        a = rng.choice(cluster_to_idxs[c1])
        b = rng.choice(cluster_to_idxs[c2])
        pairs.append((a, b, 0))

    rng.shuffle(pairs)
    return pairs


def build_pair_features(feat_matrix: np.ndarray,
                        pairs: list[tuple]) -> tuple[np.ndarray, np.ndarray]:
    """
    For each (a, b, label) pair, build feature vector:
      [|feat_a - feat_b|, feat_a, feat_b]

    The difference features mirror the mentor's approach (Tier-1/2/3
    features are all difference-based). Raw features are included too
    because the mentor found some raw query/candidate features also matter.
    """
    X_list = []
    y_list = []

    for a, b, label in pairs:
        fa = feat_matrix[a]
        fb = feat_matrix[b]
        diff = np.abs(fa - fb)
        vec = np.concatenate([diff, fa, fb])
        X_list.append(vec)
        y_list.append(label)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_pairs", type=int, default=None,
                    help="Override N_SAME and N_DIFF with a single value each")
    ap.add_argument("--n_clusters_skip", type=int, default=None,
                    help="Skip cluster with this id from positive pairs "
                         "(use cluster 4 = no-drop cluster to avoid polluting positives)")
    ap.add_argument("--model", default="xgb",
                    choices=["rf", "lr", "gb", "xgb", "lgb"],
                    help="Model type: xgb=XGBoost GPU (default), lgb=LightGBM GPU, rf=RandomForest, lr=LogisticRegression, gb=GradientBoosting")
    ap.add_argument("--no_cv", action="store_true",
                    help="Skip cross-validation (faster)")
    args = ap.parse_args()

    n_same = args.n_pairs or N_SAME
    n_diff = args.n_pairs or N_DIFF
    skip   = {args.n_clusters_skip} if args.n_clusters_skip is not None else None

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1) Load features ────────────────────────────────────────────────
    print(f"Loading features: {FEAT_PATH}")
    feat_df = pd.read_parquet(FEAT_PATH)
    print(f"  Shape: {feat_df.shape}")

    feat_cols = get_feat_cols(feat_df)
    print(f"  Feature columns: {len(feat_cols)}")

    # ── 2) Load cluster assignments ──────────────────────────────────────
    print(f"Loading clusters: {CLUS_PATH}")
    clus_df = pd.read_parquet(CLUS_PATH)[["puzzle_id", "cluster", "cluster_name"]]
    print(f"  Cluster distribution:\n{clus_df['cluster'].value_counts().sort_index().to_string()}")

    # ── 3) Merge ─────────────────────────────────────────────────────────
    df = feat_df.merge(clus_df, on="puzzle_id", how="inner")
    print(f"  After merge: {len(df):,} rows")

    feat_matrix = df[feat_cols].values.astype(np.float32)

    # ── 4) Sample pairs ──────────────────────────────────────────────────
    print(f"\nSampling pairs: {n_same:,} same-cluster + {n_diff:,} diff-cluster ...")
    if skip:
        print(f"  Skipping cluster(s) {skip} from positive pairs")

    rng = random.Random(RANDOM_STATE)
    pairs = sample_pairs(df, "cluster", n_same, n_diff,
                         skip_clusters=skip, rng=rng)

    print(f"  Total pairs: {len(pairs):,}")
    labels = [p[2] for p in pairs]
    print(f"  Label dist: {labels.count(1):,} similar, {labels.count(0):,} different")

    # ── 5) Build pair feature matrix ────────────────────────────────────
    print("\nBuilding pair feature vectors ...")
    X, y = build_pair_features(feat_matrix, pairs)
    print(f"  X shape: {X.shape}  (pairs × features)")
    print(f"  y shape: {y.shape}")

    # ── 6) Train model ───────────────────────────────────────────────────
    print(f"\nTraining model: {args.model} ...")

    if args.model in ("rf", "xgb", "lgb"):
        model = Pipeline([
            ("clf", RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=10,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ))
        ])
    elif args.model == "lr":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            ))
        ])
    elif args.model == "gb":
        model = Pipeline([
            ("clf", GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                random_state=RANDOM_STATE,
            ))
        ])

    elif args.model == "xgb":
        # XGBoost with CUDA GPU acceleration
        model = Pipeline([
            ("clf", xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                device="cuda",
                eval_metric="auc",
                use_label_encoder=False,
                random_state=RANDOM_STATE,
                scale_pos_weight=1,
                tree_method="hist",   # required for GPU in xgb >=2.0
            ))
        ])
    elif args.model == "lgb":
        # LightGBM with CUDA GPU acceleration
        model = Pipeline([
            ("clf", lgb.LGBMClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                device="gpu",
                random_state=RANDOM_STATE,
                class_weight="balanced",
                verbose=-1,
            ))
        ])

    # ── 7) Cross-validation ──────────────────────────────────────────────────────
    if not args.no_cv:
        print(f"\nCross-validating ({CV_FOLDS}-fold stratified) ...")
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        print(f"  AUC scores: {auc_scores.round(3)}")
        print(f"  Mean AUC:   {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")
    else:
        print("  (skipping CV)")

    # ── 8) Fit on full training set ──────────────────────────────────────
    print("\nFitting on full training set ...")
    model.fit(X, y)

    # Quick train-set report
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    train_auc = roc_auc_score(y, y_prob)
    print(f"  Train AUC: {train_auc:.3f}")
    print(classification_report(y, y_pred, target_names=["different", "similar"]))

    # ── 9) Feature importance (RF only) ──────────────────────────────────
    if args.model in ("rf", "xgb", "lgb"):
        clf = model.named_steps["clf"]
        importances = clf.feature_importances_
        if hasattr(clf, "best_iteration"):
            print(f"  Best iteration: {clf.best_iteration}")

        # Feature names: diff_X, q_X, c_X
        diff_names = [f"d_{c}" for c in feat_cols]
        q_names    = [f"q_{c}" for c in feat_cols]
        c_names    = [f"c_{c}" for c in feat_cols]
        all_names  = diff_names + q_names + c_names

        top_idx = np.argsort(importances)[::-1][:20]
        print("\nTop 20 feature importances:")
        for i in top_idx:
            print(f"  {all_names[i]:<45} {importances[i]:.4f}")

    # ── 10) Save model ───────────────────────────────────────────────────
    payload = {
        "model":      model,
        "feat_cols":  feat_cols,
        "model_type": args.model,
        "n_pairs":    len(pairs),
        "label_type": "cluster_weak",
        "gpu": args.model in ("xgb", "lgb"),
        "n_clusters": int(df["cluster"].nunique()),
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"\nModel saved: {MODEL_PATH}")
    print("\nDone.")
    print("Next step: use this model to re-rank BM25 results in app.py,")
    print("or collect human labels and retrain with train_classifier_supervised.py")


if __name__ == "__main__":
    main()