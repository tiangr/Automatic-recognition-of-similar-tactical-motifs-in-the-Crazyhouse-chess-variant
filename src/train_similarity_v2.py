"""
train_similarity_v2.py
=======================
Improvements over v1:
  - Richer pawn structure features (passed, doubled, isolated, advancement)
  - King safety features (rank, zone, proximity to center)
  - Grid search for both Decision Tree and XGBoost
  - Cross-validated evaluation

Usage:
    python train_similarity_v2.py --data dataset_full.jsonl --out output/
"""

import argparse
import json
import re
import warnings
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="dataset_full.jsonl")
parser.add_argument("--out",  default="output_v2")
args = parser.parse_args()

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
print(f"[1/7] Loading {args.data} ...")
rows = []
with open(args.data, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except:
                pass
print(f"       {len(rows):,} rows loaded")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────
PIECES = ["K", "Q", "R", "B", "N", "P"]

def parse_static_tokens(text: str) -> dict:
    f = defaultdict(float)

    # Normalise spaced format → compact
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(
        r'([wb])\s+([KQRBNPkqrbnp])\s+@\s+([a-h])\s+([1-8])',
        lambda m: f"{m.group(1)}{m.group(2)}@{m.group(3)}{m.group(4)}",
        text
    )

    # Collect pawn positions for pawn structure analysis
    w_pawns = []   # list of (file_int 1-8, rank_int 1-8)
    b_pawns = []

    for tok in text.split():
        # Piece placement
        m = re.match(r'^([wb])([KQRBNPkqrbnp])@([a-h])([1-8])$', tok, re.IGNORECASE)
        if m:
            color = m.group(1).lower()
            piece = m.group(2).upper()
            file_c = m.group(3).lower()
            rank_c = m.group(4)
            side = "w" if color == "w" else "b"
            file_i = ord(file_c) - ord('a') + 1  # 1=a .. 8=h
            rank_i = int(rank_c)                  # 1=rank1 .. 8=rank8

            f[f"{side}_{piece}_count"] += 1

            if piece == "K":
                f[f"{side}_king_file"]     = file_i
                f[f"{side}_king_rank"]     = rank_i
                f[f"{side}_king_kingside"] = 1.0 if file_i >= 5 else 0.0
                # King centralisation score (distance from centre d1/e1 area)
                f[f"{side}_king_center_dist"] = abs(file_i - 4.5) + abs(rank_i - 4.5)

            if piece == "P":
                if side == "w":
                    w_pawns.append((file_i, rank_i))
                else:
                    b_pawns.append((file_i, rank_i))
            continue

        # Pocket tokens
        m2 = re.match(r'^([wb])pocket:([QRBNP])$', tok, re.IGNORECASE)
        if m2:
            side  = "w" if m2.group(1).lower() == "w" else "b"
            piece = m2.group(2).upper()
            f[f"{side}_pocket_{piece}"] += 1

    # ── Pawn structure features ───────────────────────────────────────────────

    for side, pawns, forward in [("w", w_pawns, 1), ("b", b_pawns, -1)]:
        opp_pawns = b_pawns if side == "w" else w_pawns

        files = [p[0] for p in pawns]
        ranks = [p[1] for p in pawns]
        opp_files = [p[0] for p in opp_pawns]
        opp_ranks = [p[1] for p in opp_pawns]

        # Total pawns (already counted above as P_count)
        n = len(pawns)
        f[f"{side}_pawn_count"] = float(n)

        if n == 0:
            f[f"{side}_doubled_pawns"]  = 0.0
            f[f"{side}_isolated_pawns"] = 0.0
            f[f"{side}_passed_pawns"]   = 0.0
            f[f"{side}_pawn_advancement"] = 0.0
            f[f"{side}_pawn_spread"]    = 0.0
            f[f"{side}_pawn_center"]    = 0.0
        else:
            # Doubled: more than one pawn on same file
            file_counts = Counter(files)
            doubled = sum(c - 1 for c in file_counts.values() if c > 1)
            f[f"{side}_doubled_pawns"] = float(doubled)

            # Isolated: no friendly pawn on adjacent files
            isolated = 0
            for pf in files:
                if (pf - 1) not in file_counts and (pf + 1) not in file_counts:
                    isolated += 1
            f[f"{side}_isolated_pawns"] = float(isolated)

            # Passed: no enemy pawn on same or adjacent file ahead of this pawn
            passed = 0
            for pf, pr in pawns:
                # "ahead" for white = higher rank, for black = lower rank
                blocked = False
                for of, or_ in opp_pawns:
                    if abs(of - pf) <= 1:
                        if side == "w" and or_ > pr:
                            blocked = True; break
                        if side == "b" and or_ < pr:
                            blocked = True; break
                if not blocked:
                    passed += 1
            f[f"{side}_passed_pawns"] = float(passed)

            # Pawn advancement (white: higher rank = more advanced)
            if side == "w":
                f[f"{side}_pawn_advancement"] = float(sum(ranks)) / n
            else:
                # For black, rank 1 = most advanced
                f[f"{side}_pawn_advancement"] = float(sum(9 - r for r in ranks)) / n

            # Pawn spread (file range)
            f[f"{side}_pawn_spread"] = float(max(files) - min(files)) if files else 0.0

            # Pawn center control (pawns on d,e files = files 4,5)
            center_pawns = sum(1 for pf in files if pf in (4, 5))
            f[f"{side}_pawn_center"] = float(center_pawns)

    # Pawn tension: pawns that can capture each other
    tension = 0
    for wf, wr in w_pawns:
        for bf, br in b_pawns:
            if abs(wf - bf) == 1 and wr + 1 == br:
                tension += 1
    f["pawn_tension"] = float(tension)

    # ── Derived aggregate features ────────────────────────────────────────────
    for side in ["w", "b"]:
        f[f"{side}_total_pieces"]    = sum(f[f"{side}_{p}_count"] for p in PIECES)
        f[f"{side}_pocket_total"]    = sum(f[f"{side}_pocket_{p}"] for p in PIECES)
        f[f"{side}_material"]        = (
            9*f[f"{side}_Q_count"] + 5*f[f"{side}_R_count"] +
            3*f[f"{side}_B_count"] + 3*f[f"{side}_N_count"] + f[f"{side}_P_count"]
        )
        f[f"{side}_pocket_material"] = (
            9*f[f"{side}_pocket_Q"] + 5*f[f"{side}_pocket_R"] +
            3*f[f"{side}_pocket_B"] + 3*f[f"{side}_pocket_N"] + f[f"{side}_pocket_P"]
        )

    f["total_pieces_on_board"] = f["w_total_pieces"] + f["b_total_pieces"]
    f["material_balance"]      = f["w_material"] - f["b_material"]
    f["total_pawns"]           = f["w_pawn_count"] + f["b_pawn_count"]

    return dict(f)


def compute_pair_features(q: dict, c: dict) -> dict:
    all_keys = set(q.keys()) | set(c.keys())
    pair = {}
    for k in all_keys:
        pair[f"diff_{k}"] = abs(q.get(k, 0.0) - c.get(k, 0.0))
    return pair


# ─────────────────────────────────────────────────────────────────────────────
# BUILD FEATURE MATRIX
# ─────────────────────────────────────────────────────────────────────────────
print("[2/7] Extracting features ...")
feature_rows, labels = [], []
skipped = 0

for row in rows:
    q_text = row.get("query_text_static", "")
    c_text = row.get("candidate_text_static", "")
    label  = row.get("label", -1)
    if not q_text or not c_text or label == -1:
        skipped += 1
        continue
    q_feats = parse_static_tokens(q_text)
    c_feats = parse_static_tokens(c_text)
    pair    = compute_pair_features(q_feats, c_feats)
    pair["bm25_rank"] = float(row.get("bm25_rank", 10))
    feature_rows.append(pair)
    labels.append(int(label))

print(f"       {len(feature_rows):,} pairs  (skipped {skipped})")

df = pd.DataFrame(feature_rows).fillna(0.0)
y  = np.array(labels)
feature_cols = list(df.columns)
X = df.values

print(f"       Feature matrix: {X.shape[0]} x {X.shape[1]} features")
print(f"       Positives: {y.sum()}  Negatives: {(1-y).sum()}  ({y.mean()*100:.1f}% positive)")

# ─────────────────────────────────────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"\n[3/7] Train: {len(X_train)}  Test: {len(X_test)}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# DECISION TREE + GRID SEARCH
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/7] Grid search — Decision Tree ...")
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

dt_param_grid = {
    "max_depth":        [3, 4, 5, 6, 7, None],
    "min_samples_leaf": [4, 6, 8, 12, 16],
    "criterion":        ["gini", "entropy"],
    "class_weight":     [None, "balanced"],
}
dt_gs = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_param_grid,
    scoring="f1",
    cv=cv,
    n_jobs=-1,
    verbose=0,
)
dt_gs.fit(X_train, y_train)
best_dt = dt_gs.best_estimator_
print(f"  Best DT params: {dt_gs.best_params_}")
print(f"  Best CV F1:     {dt_gs.best_score_:.3f}")

y_pred_dt = best_dt.predict(X_test)
print("\n── Decision Tree (best) ───────────────────────────────")
print(classification_report(y_test, y_pred_dt, target_names=["not_similar", "similar"]))
print("\nTree rules (depth <= 3):")
print(export_text(best_dt, feature_names=feature_cols, max_depth=3))

# Tree diagram
fig, ax = plt.subplots(figsize=(30, 13))
plot_tree(
    best_dt,
    feature_names=feature_cols,
    class_names=["not_similar", "similar"],
    filled=True, rounded=True, fontsize=7,
    max_depth=4, ax=ax, impurity=False,
)
depth_str = str(best_dt.max_depth) if best_dt.max_depth else "full"
ax.set_title(
    f"Decision Tree (grid-searched) — Static Position Similarity\n"
    f"depth={depth_str}  min_leaf={best_dt.min_samples_leaf}  "
    f"criterion={best_dt.criterion}  CV F1={dt_gs.best_score_:.3f}",
    fontsize=12, pad=15
)
plt.tight_layout()
plt.savefig(OUT / "decision_tree.png", dpi=150, bbox_inches="tight")
plt.close()
print("  -> Saved: decision_tree.png")

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_dt),
                       display_labels=["not similar", "similar"]).plot(ax=ax, colorbar=False)
ax.set_title("Decision Tree — Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT / "dt_confusion_matrix.png", dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# XGBOOST + GRID SEARCH
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/7] Grid search — XGBoost ...")
try:
    import xgboost as xgb

    xgb_param_grid = {
        "n_estimators":    [100, 200, 300, 400],
        "max_depth":       [3, 4, 5, 6],
        "learning_rate":   [0.01, 0.05, 0.1, 0.2],
        "subsample":       [0.7, 0.8, 1.0],
        "min_child_weight":[1, 3, 5],
    }
    xgb_gs = GridSearchCV(
        xgb.XGBClassifier(
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        ),
        xgb_param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    xgb_gs.fit(X_train, y_train)
    best_xgb = xgb_gs.best_estimator_
    print(f"\n  Best XGB params: {xgb_gs.best_params_}")
    print(f"  Best CV F1:      {xgb_gs.best_score_:.3f}")

    y_pred_xgb = best_xgb.predict(X_test)
    print("\n── XGBoost (best) ─────────────────────────────────────")
    print(classification_report(y_test, y_pred_xgb, target_names=["not_similar", "similar"]))

    # Feature importance
    importances = best_xgb.feature_importances_
    top_n = 25
    top_idx   = np.argsort(importances)[::-1][:top_n]
    top_feats = [feature_cols[i] for i in top_idx]
    top_vals  = importances[top_idx]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, top_n))
    ax.barh(range(top_n), top_vals[::-1], color=colors, edgecolor="white")
    ax.set_yticks(range(top_n))
    clean_labels = [
        f.replace("diff_", "Δ ").replace("_count", "").replace("_", " ")
        for f in top_feats[::-1]
    ]
    ax.set_yticklabels(clean_labels, fontsize=9)
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(
        f"XGBoost (grid-searched) — Top {top_n} Features\n"
        f"CV F1={xgb_gs.best_score_:.3f}",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(OUT / "xgb_feature_importance.png", dpi=150)
    plt.close()
    print("  -> Saved: xgb_feature_importance.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_xgb),
                           display_labels=["not similar", "similar"]).plot(ax=ax, colorbar=False)
    ax.set_title("XGBoost — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUT / "xgb_confusion_matrix.png", dpi=150)
    plt.close()

    # SHAP
    print("\n[6/7] Computing SHAP values ...")
    try:
        import shap
        explainer   = shap.TreeExplainer(best_xgb)
        shap_values = explainer.shap_values(X_test)

        fig, ax = plt.subplots(figsize=(12, 9))
        shap.summary_plot(
            shap_values, X_test,
            feature_names=feature_cols,
            show=False, plot_type="dot",
            max_display=25,
        )
        plt.title("SHAP Summary — Features Driving Similarity Predictions", pad=15, fontsize=12)
        plt.tight_layout()
        plt.savefig(OUT / "xgb_shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  -> Saved: xgb_shap_summary.png")
    except ImportError:
        print("  (pip install shap to get SHAP plot)")

    # ── Grid search CV score comparison ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    models  = ["Decision Tree\n(default)", "Decision Tree\n(grid search)",
               "XGBoost\n(default)", "XGBoost\n(grid search)"]
    # recompute default scores for comparison
    from sklearn.tree import DecisionTreeClassifier as DTC
    dt_def_cv = cross_val_score(DTC(max_depth=5, min_samples_leaf=8, random_state=42),
                                 X, y, cv=cv, scoring="f1").mean()
    xgb_def_cv = cross_val_score(
        xgb.XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                           eval_metric="logloss", random_state=42, verbosity=0),
        X, y, cv=cv, scoring="f1").mean()

    scores = [dt_def_cv, dt_gs.best_score_, xgb_def_cv, xgb_gs.best_score_]
    colors_bar = ["#6db4e8", "#2980b9", "#e8a06d", "#c0392b"]
    bars = ax.bar(models, scores, color=colors_bar, edgecolor="white", width=0.5)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{score:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, min(1.0, max(scores) + 0.1))
    ax.set_ylabel("5-fold CV F1 (similar class)")
    ax.set_title("Model Comparison — CV F1 Score")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    plt.savefig(OUT / "model_comparison.png", dpi=150)
    plt.close()
    print("  -> Saved: model_comparison.png")

except ImportError:
    print("  XGBoost not installed — pip install xgboost")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
print("\n[7/7] Dataset overview ...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# BM25 rank vs label
ranks_pos = [r["bm25_rank"] for r in rows if r["label"] == 1]
ranks_neg = [r["bm25_rank"] for r in rows if r["label"] == 0]
axes[0].hist(ranks_pos, bins=10, alpha=0.65, color="#2ecc71", label="Similar (1)", edgecolor="white")
axes[0].hist(ranks_neg, bins=10, alpha=0.65, color="#e74c3c", label="Not similar (0)", edgecolor="white")
axes[0].set_xlabel("BM25 Rank")
axes[0].set_ylabel("Count")
axes[0].set_title("BM25 Rank by Label")
axes[0].legend()

# Positives per query
pos_per_query = Counter(r["query_fen"] for r in rows if r["label"] == 1)
axes[1].hist(list(pos_per_query.values()), bins=range(0, 12),
             alpha=0.75, color="#6db4e8", edgecolor="white")
axes[1].set_xlabel("Positives per Query")
axes[1].set_ylabel("Queries")
axes[1].set_title("Label=1 Count per Query")

# Feature count comparison v1 vs v2
axes[2].bar(["v1 features", "v2 features (+ pawn structure)"],
            [41, X.shape[1]], color=["#95a5a6", "#2ecc71"], edgecolor="white")
axes[2].set_ylabel("Number of features")
axes[2].set_title("Feature Count: v1 vs v2")
for i, v in enumerate([41, X.shape[1]]):
    axes[2].text(i, v + 0.3, str(v), ha="center", fontweight="bold")

plt.suptitle(
    f"Dataset Overview  |  {len(rows)} pairs  |  49 queries  |  {y.mean()*100:.0f}% positive",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig(OUT / "dataset_overview.png", dpi=150)
plt.close()
print("  -> Saved: dataset_overview.png")

print(f"\nDone. All outputs saved to: {OUT.resolve()}/")
print("  decision_tree.png           interpretable model (grid-searched)")
print("  dt_confusion_matrix.png")
print("  xgb_feature_importance.png  XGBoost top features")
print("  xgb_confusion_matrix.png")
print("  xgb_shap_summary.png        SHAP explanations")
print("  model_comparison.png        default vs grid-searched comparison")
print("  dataset_overview.png        dataset statistics")
