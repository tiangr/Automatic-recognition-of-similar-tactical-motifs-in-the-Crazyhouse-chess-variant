"""
train_similarity.py
====================
Reads labelled JSONL exports from the crazyhouse webapp and trains:
  1) Decision Tree  — interpretable, visualized as tree diagram
  2) XGBoost        — powerful, visualized with feature importance + SHAP

v2 enrichments (feature_descriptions_2_.md alignment, April 2026):
  - Dynamic features extracted from text_dynamic (DG_ and DS_ tokens)
    → DG_!F_checkmate, DG_!F_p+, DG_!F_sacrifice, DG_!F_ox,
      DG_!mating_piece_X, DS_!piece_captured_X, DS_!piece_moved_X,
      DS_!attack_X>Y, DS_!piece_captured_XY (consecutive pairs)
  - Pair-level features (f_ prefix per MD spec):
    → f_move_edit_sim, f_move_edit_dist (Levenshtein on UCI move sequences)
    → f_theme_jaccard, f_theme_intersection (theme set overlap)
  - Theme features extracted from themes field (TH_ per MD spec)
  - Difference features (d_..._abs) for all base features
  - Static features retained from v1 (piece counts, pocket counts)

Feature importance tiers (from MD):
  Tier 1: d_DG_!F_checkmate_abs (strongest predictor)
  Tier 2: f_move_edit_sim / f_move_edit_dist
  Tier 3: d_DS_!piece_captured_Q_abs, d_DS_!piece_captured_q_abs, etc.
  Tier 4: d_DG_!F_p+_abs, d_DG_!F_sacrifice_abs, d_DG_!mating_piece_q_abs

Usage:
    python train_similarity.py --data dataset_full.jsonl --out output/
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

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="dataset_full.jsonl")
parser.add_argument("--out",  default="output")
args = parser.parse_args()

OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)

# ── Load dataset ──────────────────────────────────────────────────────────────
print(f"[1/7] Loading dataset from {args.data} ...")
rows = []
with open(args.data, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception as e:
            print(f"  Skipping bad line: {e}")
print(f"       Loaded {len(rows):,} rows")

# ── Feature extraction ────────────────────────────────────────────────────────
PIECES = ["K", "Q", "R", "B", "N", "P"]


def parse_static_tokens(text: str) -> dict:
    """Extract static features: piece counts, pocket counts, king position."""
    f = defaultdict(float)
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(
        r'([wb])\s+([KQRBNPkqrbnp])\s+@\s+([a-h])\s+([1-8])',
        lambda m: f"{m.group(1)}{m.group(2)}@{m.group(3)}{m.group(4)}",
        text
    )

    # Connectivity regex: X>Ysq, X<Ysq, X=Ysq
    conn_re = re.compile(r'^([A-Za-z])(>|<|=)([A-Za-z])([a-h][1-8])$')
    # Pawn structure
    pawn_struct_re = re.compile(r"^(\'?)([IiFfLlSs])([a-h][1-8])$")
    pawn_chain_re  = re.compile(r"^([Ww])\[([a-h][1-8/]+)\]$")
    pawn_island_re = re.compile(r"^([Pp])\((\d+)\)$")

    for tok in text.split():
        # Piece placement
        m = re.match(r'^([wb])([KQRBNPkqrbnp])@([a-h])([1-8])$', tok, re.IGNORECASE)
        if m:
            color = m.group(1).lower()
            piece = m.group(2).upper()
            file_c = m.group(3).lower()
            rank_c = m.group(4)
            side = "w" if color == "w" else "b"
            f[f"{side}_{piece}_count"] += 1
            if piece == "K":
                f[f"{side}_king_file"] = ord(file_c) - ord('a') + 1
                f[f"{side}_king_rank"] = int(rank_c)
                f[f"{side}_king_kingside"] = 1.0 if ord(file_c) >= ord('e') else 0.0
            continue

        # Pocket tokens
        m2 = re.match(r'^([wb])pocket:([QRBNP])$', tok, re.IGNORECASE)
        if m2:
            side = "w" if m2.group(1).lower() == "w" else "b"
            piece = m2.group(2).upper()
            f[f"{side}_pocket_{piece}"] += 1
            continue

        # Connectivity tokens (v2 enrichment)
        m3 = conn_re.match(tok)
        if m3:
            actor_sym, relation, _, _ = m3.groups()
            actor_side = "white" if actor_sym.isupper() else "black"
            if relation == ">":
                f[f"{actor_side}_attacks"] += 1
            elif relation == "<":
                f[f"{actor_side}_defends"] += 1
            elif relation == "=":
                f[f"{actor_side}_xrays"] += 1
            continue

        # Pawn structure tokens (v2 enrichment)
        m4 = pawn_struct_re.match(tok)
        if m4:
            prefix, letter, sq = m4.group(1), m4.group(2), m4.group(3)
            side = "white" if letter.isupper() else "black"
            letter_lower = letter.lower()
            if letter_lower == 'i':
                f[f"{side}_isolated"] += 1
            elif letter_lower == 'f':
                if prefix == "'":
                    f[f"{side}_protected_passed"] += 1
                else:
                    f[f"{side}_passed"] += 1
            elif letter_lower == 'l':
                f[f"{side}_backward"] += 1
            elif letter_lower == 's':
                f[f"{side}_doubled"] += 1
            continue

        m5 = pawn_chain_re.match(tok)
        if m5:
            sym, sqs = m5.group(1), m5.group(2)
            side = "white" if sym == "W" else "black"
            f[f"{side}_pawn_chain"] += len(sqs.split("/"))
            continue

        m6 = pawn_island_re.match(tok)
        if m6:
            sym, n = m6.group(1), int(m6.group(2))
            side = "white" if sym == "P" else "black"
            f[f"{side}_pawn_islands"] = float(n)
            continue

    # Derived aggregate features
    for side in ["w", "b"]:
        f[f"{side}_total_pieces"] = sum(f[f"{side}_{p}_count"] for p in PIECES)
        f[f"{side}_pocket_total"] = sum(f[f"{side}_pocket_{p}"] for p in PIECES)
        f[f"{side}_material"] = (
            9*f[f"{side}_Q_count"] + 5*f[f"{side}_R_count"] +
            3*f[f"{side}_B_count"] + 3*f[f"{side}_N_count"] + f[f"{side}_P_count"]
        )
        f[f"{side}_pocket_material"] = (
            9*f[f"{side}_pocket_Q"] + 5*f[f"{side}_pocket_R"] +
            3*f[f"{side}_pocket_B"] + 3*f[f"{side}_pocket_N"] + f[f"{side}_pocket_P"]
        )
    f["total_pieces_on_board"] = f["w_total_pieces"] + f["b_total_pieces"]
    f["material_balance"]      = f["w_material"] - f["b_material"]
    return dict(f)


def parse_dynamic_tokens(text: str) -> dict:
    """
    Extract dynamic features from text_dynamic field.
    Maps tokens to feature names matching the MD spec (DG_ and DS_ prefix scheme).

    Key features (by importance tier):
      Tier 1: DG_!F_checkmate
      Tier 3: DS_!piece_captured_X (incl. consecutive pairs XY)
      Tier 4: DG_!F_p+, DG_!F_sacrifice, DG_!F_ox, DG_!mating_piece_X
    """
    f = defaultdict(float)
    seen = set()

    for tok in text.split():
        if tok in seen:
            continue
        seen.add(tok)

        # ── Dynamic General flags (DG_) ──────────────────────────────────
        if tok == "dyn:hasMate":
            f["DG_!F_checkmate"] = 1
        elif tok == "dyn:hasCapture":
            f["DG_!F_px"] = 1
        elif tok == "dyn:oppCaptures":
            f["DG_!F_ox"] = 1
        elif tok == "dyn:hasPromotion":
            f["DG_!F_p+"] = 1
        elif tok == "dyn:hasSacrifice":
            f["DG_!F_sacrifice"] = 1
        elif tok == "dyn:hasCheck":
            f["DG_!F_+"] = 1
        elif tok == "dyn:hasDrop":
            f["DG_!F_drop"] = 1
        elif tok == "dyn:hasDropCheck":
            f["DG_!F_drop+"] = 1

        # ── Mating piece (DG_!mating_piece_X and pairs) ──────────────────
        elif tok.startswith("dyn:mating_piece:"):
            piece = tok.split("dyn:mating_piece:")[1]
            f[f"DG_!mating_piece_{piece}"] = 1
        elif tok.startswith("dyn:matingPiecePair:"):
            pair = tok.split("dyn:matingPiecePair:")[1]
            f[f"DG_!mating_piece_{pair}"] = 1

        # ── Per-move: piece captured (DS_!piece_captured_X) ─────────────
        elif tok.startswith("pv:capturePiece:"):
            piece = tok.split("pv:capturePiece:")[1]
            f[f"DS_!piece_captured_{piece}"] = 1

        # ── Consecutive capture pairs (DS_!piece_captured_XY) ────────────
        elif tok.startswith("dyn:capPair:"):
            pair = tok.split("dyn:capPair:")[1]
            f[f"DS_!piece_captured_{pair}"] = 1

        # ── Piece-moved tokens (DS_!piece_moved_X) ───────────────────────
        elif tok.startswith("pv:pieceMoved:"):
            piece = tok.split("pv:pieceMoved:")[1]
            f[f"DS_!piece_moved_{piece}"] += 1

        # ── Attack-relation tokens (DS_!attack_X>Y) ──────────────────────
        elif tok.startswith("pv:attack:"):
            relation = tok.split("pv:attack:")[1]
            f[f"DS_!attack_{relation}"] += 1

        # ── Sacrifice tokens ──────────────────────────────────────────────
        elif tok.startswith("pv:sacrificePiece:"):
            piece = tok.split("pv:sacrificePiece:")[1]
            f[f"DG_!sacrifice_piece_{piece}"] = 1

    return dict(f)


def parse_themes(themes_str) -> set:
    """
    Parse themes field into a set of theme strings.
    Handles: space-separated string, JSON array string, or Python list.
    """
    if not themes_str:
        return set()
    if isinstance(themes_str, list):
        return set(str(t) for t in themes_str if t)
    if isinstance(themes_str, str):
        themes_str = themes_str.strip()
        # Try JSON array
        if themes_str.startswith("["):
            try:
                return set(json.loads(themes_str))
            except Exception:
                pass
        # Space-separated
        return set(themes_str.split())
    return set()


def _levenshtein_sim(seq1: list, seq2: list) -> tuple[float, float]:
    """
    Compute normalized Levenshtein distance and similarity for two sequences.
    Returns (distance, similarity) both in [0, 1].
    """
    if not seq1 and not seq2:
        return 0.0, 1.0

    n, m = len(seq1), len(seq2)
    dp = list(range(m + 1))

    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])

    raw_dist = dp[m]
    max_len  = max(n, m)
    norm_dist = raw_dist / max_len
    norm_sim  = 1.0 - norm_dist
    return norm_dist, norm_sim


def compute_pair_features(
    q_static: dict, c_static: dict,
    q_dyn: dict,    c_dyn: dict,
    q_themes: set,  c_themes: set,
    q_pv: list,     c_pv: list,
    q_meta: dict = None, c_meta: dict = None,
) -> dict:
    """
    Build the full feature vector for a (query, candidate) pair.

    Includes:
      q_{feature} and c_{feature} — raw values for each side
      d_{feature}_abs             — absolute difference |q - c|
      f_move_edit_sim/dist        — pair-level Levenshtein on UCI moves
      f_theme_jaccard             — Jaccard similarity of theme sets
      f_theme_intersection        — number of shared themes
      TH_{theme} for q and c      — binary theme indicators
      q_meta_* / c_meta_* / d_meta_*_abs — metadata features (length, delta, etc.)
    """
    if q_meta is None: q_meta = {}
    if c_meta is None: c_meta = {}
    pair = {}

    # ── Static features (q_ and c_ prefixed, plus diffs) ─────────────────
    all_static_keys = set(q_static.keys()) | set(c_static.keys())
    for k in all_static_keys:
        qv = q_static.get(k, 0.0)
        cv = c_static.get(k, 0.0)
        pair[f"q_{k}"] = qv
        pair[f"c_{k}"] = cv
        pair[f"d_{k}_abs"] = abs(qv - cv)

    # ── Dynamic features (q_ and c_ prefixed, plus diffs) ────────────────
    all_dyn_keys = set(q_dyn.keys()) | set(c_dyn.keys())
    for k in all_dyn_keys:
        qv = q_dyn.get(k, 0.0)
        cv = c_dyn.get(k, 0.0)
        pair[f"q_{k}"] = qv
        pair[f"c_{k}"] = cv
        pair[f"d_{k}_abs"] = abs(qv - cv)

    # ── Theme features (TH_ prefix per MD spec) ───────────────────────────
    all_themes = q_themes | c_themes
    for theme in all_themes:
        pair[f"q_TH_{theme}"] = 1.0 if theme in q_themes else 0.0
        pair[f"c_TH_{theme}"] = 1.0 if theme in c_themes else 0.0
        pair[f"d_TH_{theme}_abs"] = 0.0 if (theme in q_themes) == (theme in c_themes) else 1.0

    # ── Pair-level: theme overlap (f_ prefix per MD spec) ────────────────
    intersection = len(q_themes & c_themes)
    union        = len(q_themes | c_themes)
    pair["f_theme_jaccard"]      = intersection / union if union > 0 else 0.0
    pair["f_theme_intersection"] = float(intersection)

    # ── Pair-level: move sequence similarity (f_ prefix, Tier-2 feature) ──
    dist, sim = _levenshtein_sim(q_pv, c_pv)
    pair["f_move_edit_dist"] = dist
    pair["f_move_edit_sim"]  = sim

    # ── Metadata features (section 4.3 of MD spec) ────────────────────────
    # MD fields: length, rating, popularity
    # Our equivalents for game-mined Crazyhouse tactics:
    #   meta_length    — PV length (solution length in half-moves)
    #   meta_delta     — centipawn loss at the tactic (difficulty proxy)
    #   meta_cp_before — engine eval before tactic (position sharpness)
    #   meta_mate_in   — forced mate depth (0 = not a forced mate)
    all_meta_keys = set(q_meta.keys()) | set(c_meta.keys())
    for k in all_meta_keys:
        qv = float(q_meta.get(k, 0.0) or 0.0)
        cv = float(c_meta.get(k, 0.0) or 0.0)
        pair[f"q_{k}"] = qv
        pair[f"c_{k}"] = cv
        pair[f"d_{k}_abs"] = abs(qv - cv)

    return pair


# ── Build feature matrix ──────────────────────────────────────────────────────
print("[2/7] Extracting features ...")
feature_rows = []
labels = []
skipped = 0

for row in rows:
    q_text_static  = row.get("query_text_static", "")
    c_text_static  = row.get("candidate_text_static", "")
    q_text_dynamic = row.get("query_text_dynamic", "")
    c_text_dynamic = row.get("candidate_text_dynamic", "")
    label          = row.get("label", -1)

    if not q_text_static or not c_text_static or label == -1:
        skipped += 1
        continue

    q_static = parse_static_tokens(q_text_static)
    c_static = parse_static_tokens(c_text_static)
    q_dyn    = parse_dynamic_tokens(q_text_dynamic)
    c_dyn    = parse_dynamic_tokens(c_text_dynamic)

    # Parse themes (may be stored as field or embedded in text)
    q_themes = parse_themes(row.get("query_themes", ""))
    c_themes = parse_themes(row.get("candidate_themes", ""))

    # Parse PV move sequences for Levenshtein similarity
    # Stored as space-separated UCI strings or JSON arrays
    def _parse_pv(val):
        if not val:
            return []
        if isinstance(val, list):
            return val
        val = str(val).strip()
        if val.startswith("["):
            try:
                return json.loads(val)
            except Exception:
                pass
        return val.split()

    q_pv = _parse_pv(row.get("query_pv", "") or row.get("query_text_dynamic", ""))
    c_pv = _parse_pv(row.get("candidate_pv", "") or row.get("candidate_text_dynamic", ""))

    # Extract metadata fields exported by app.py via the corpus records
    META_KEYS = [
        "meta_length", "meta_delta", "meta_cp_before", "meta_mate_in",
        # Lichess-API-derived (present after enrich_with_ratings.py):
        "meta_avg_rating", "meta_solver_rating",
        "meta_white_rating", "meta_black_rating",
        "meta_estimated_time", "meta_clock_initial", "meta_clock_inc",
        "meta_speed_ord",  # ordinal-encoded speed (bullet=1 .. correspondence=5)
    ]
    q_meta = {k: row.get(f"query_{k}") for k in META_KEYS if row.get(f"query_{k}") is not None}
    c_meta = {k: row.get(f"candidate_{k}") for k in META_KEYS if row.get(f"candidate_{k}") is not None}

    pair = compute_pair_features(q_static, c_static, q_dyn, c_dyn,
                                  q_themes, c_themes, q_pv, c_pv,
                                  q_meta, c_meta)
    feature_rows.append(pair)
    labels.append(int(label))

print(f"       {len(feature_rows):,} pairs  (skipped {skipped})")

df = pd.DataFrame(feature_rows).fillna(0.0)
y  = np.array(labels)
feature_cols = list(df.columns)
X = df.values

print(f"       Feature matrix: {X.shape[0]} x {X.shape[1]}")
print(f"       Positives: {y.sum()}  Negatives: {(1-y).sum()}  ({y.mean()*100:.1f}% positive)")

# Save feature names for reference
with open(OUT / "feature_names.json", "w") as fh:
    json.dump(feature_cols, fh, indent=2)
print(f"       Feature names saved: {OUT / 'feature_names.json'}")

# ── Split ─────────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\n[3/7] Train: {len(X_train)}  Test: {len(X_test)}")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Decision Tree ─────────────────────────────────────────────────────────────
print("[4/7] Training Decision Tree ...")
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=8, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
dt_cv = cross_val_score(dt, X, y, cv=5, scoring="f1").mean()

print("\n── Decision Tree ──────────────────────────────────")
print(classification_report(y_test, y_pred_dt, target_names=["not_similar", "similar"]))
print(f"5-fold CV F1: {dt_cv:.3f}")
print("\nTree rules (depth <= 3):")
print(export_text(dt, feature_names=feature_cols, max_depth=3))

# Tree diagram
fig, ax = plt.subplots(figsize=(28, 12))
plot_tree(dt, feature_names=feature_cols, class_names=["not_similar", "similar"],
          filled=True, rounded=True, fontsize=7, max_depth=4, ax=ax, impurity=False)
ax.set_title("Decision Tree — Full Feature Similarity (depth <= 4)", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(OUT / "decision_tree.png", dpi=150, bbox_inches="tight")
plt.close()
print("  -> Saved: decision_tree.png")

# Confusion matrix
fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_dt),
                       display_labels=["not similar", "similar"]).plot(ax=ax, colorbar=False)
ax.set_title("Decision Tree — Confusion Matrix")
plt.tight_layout()
plt.savefig(OUT / "dt_confusion_matrix.png", dpi=150)
plt.close()

# ── XGBoost ───────────────────────────────────────────────────────────────────
print("\n[5/7] Training XGBoost ...")
try:
    import xgboost as xgb

    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        eval_metric="logloss", random_state=42, verbosity=0,
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred_xgb = xgb_model.predict(X_test)
    xgb_cv = cross_val_score(xgb_model, X, y, cv=5, scoring="f1").mean()

    print("\n── XGBoost ────────────────────────────────────────")
    print(classification_report(y_test, y_pred_xgb, target_names=["not_similar", "similar"]))
    print(f"5-fold CV F1: {xgb_cv:.3f}")

    # Feature importance
    importances = xgb_model.feature_importances_
    top_n = 25
    top_idx = np.argsort(importances)[::-1][:top_n]
    top_feats = [feature_cols[i] for i in top_idx]
    top_vals  = importances[top_idx]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.RdYlGn(np.linspace(0.15, 0.85, top_n))
    ax.barh(range(top_n), top_vals[::-1], color=colors, edgecolor="white")
    ax.set_yticks(range(top_n))
    clean_labels = [
        f.replace("d_DG_", "Δ DG ").replace("d_DS_", "Δ DS ")
         .replace("d_TH_", "Δ TH ").replace("f_", "pair:")
         .replace("q_", "Q:").replace("c_", "C:")
         .replace("_abs", "").replace("_", " ")
        for f in top_feats[::-1]
    ]
    ax.set_yticklabels(clean_labels, fontsize=8)
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title(f"XGBoost — Top {top_n} Features (Full Feature Set)\nCV F1={xgb_cv:.3f}")
    plt.tight_layout()
    plt.savefig(OUT / "xgb_feature_importance.png", dpi=150)
    plt.close()
    print("  -> Saved: xgb_feature_importance.png")

    # Confusion matrix
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
        explainer   = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_test)
        fig, ax = plt.subplots(figsize=(12, 9))
        shap.summary_plot(shap_values, X_test, feature_names=feature_cols,
                          show=False, plot_type="dot", max_display=25)
        plt.title("SHAP Summary — Features Driving Similarity Predictions", pad=15)
        plt.tight_layout()
        plt.savefig(OUT / "xgb_shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  -> Saved: xgb_shap_summary.png")
    except ImportError:
        print("  (shap not installed — pip install shap)")

except ImportError:
    print("  XGBoost not installed — pip install xgboost")

# ── Dataset overview plot ─────────────────────────────────────────────────────
print("\n[7/7] Saving dataset overview ...")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ranks_pos = [r["bm25_rank"] for r in rows if r.get("label") == 1]
ranks_neg = [r["bm25_rank"] for r in rows if r.get("label") == 0]
axes[0].hist(ranks_pos, bins=10, alpha=0.65, color="#2ecc71", label="Similar (1)", edgecolor="white")
axes[0].hist(ranks_neg, bins=10, alpha=0.65, color="#e74c3c", label="Not similar (0)", edgecolor="white")
axes[0].set_xlabel("BM25 Rank")
axes[0].set_ylabel("Count")
axes[0].set_title("BM25 Rank Distribution by Label")
axes[0].legend()

pos_per_query = Counter()
for row in rows:
    if row.get("label") == 1:
        pos_per_query[row.get("query_fen", "")] += 1
axes[1].hist(list(pos_per_query.values()), bins=range(0, 12),
             alpha=0.75, color="#6db4e8", edgecolor="white")
axes[1].set_xlabel("Positives per Query")
axes[1].set_ylabel("Number of Queries")
axes[1].set_title("Label=1 Count per Query")

# Feature category breakdown
categories = {
    "Static (d_w/b)":    sum(1 for c in feature_cols if c.startswith(("q_w_", "q_b_", "d_w_", "d_b_"))),
    "Dynamic (d_DG/DS)": sum(1 for c in feature_cols if "DG_" in c or "DS_" in c),
    "Theme (d_TH)":      sum(1 for c in feature_cols if "TH_" in c),
    "Pair-level (f_)":   sum(1 for c in feature_cols if c.startswith("f_")),
}
axes[2].bar(list(categories.keys()), list(categories.values()),
            color=["#95a5a6", "#2ecc71", "#e67e22", "#3498db"], edgecolor="white")
axes[2].set_ylabel("Number of Features")
axes[2].set_title("Feature Count by Category")
axes[2].tick_params(axis='x', rotation=15)
for i, v in enumerate(categories.values()):
    axes[2].text(i, v + 0.5, str(v), ha="center", fontweight="bold", fontsize=9)

plt.suptitle(
    f"Dataset Overview  |  {len(rows):,} pairs  |  {y.mean()*100:.0f}% positive  |  "
    f"{X.shape[1]} features",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig(OUT / "dataset_overview.png", dpi=150)
plt.close()
print("  -> Saved: dataset_overview.png")

print(f"\nDone. All outputs in: {OUT.resolve()}/")
print("  decision_tree.png           interpretable model")
print("  dt_confusion_matrix.png")
print("  xgb_feature_importance.png  XGBoost top 25 features")
print("  xgb_confusion_matrix.png")
print("  xgb_shap_summary.png        SHAP explanations (if shap installed)")
print("  dataset_overview.png        dataset statistics")
print("  feature_names.json          full feature list")