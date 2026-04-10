"""
unsupervised_clustering.py
--------------------------
Unsupervised learning on puzzle_features_mates.parquet.
Clusters Crazyhouse mate positions to discover natural tactical groups
without any labels.

Steps:
  1. Load features
  2. Select numeric feature columns (drop metadata)
  3. Standardize
  4. Reduce dimensions with PCA
  5. Cluster with KMeans
  6. Analyze and label clusters
  7. Save results to puzzle_features_mates_clustered.parquet

Usage:
    python unsupervised_clustering.py

Requirements:
    pip install pandas numpy scikit-learn pyarrow matplotlib seaborn
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

ROOT        = Path(__file__).resolve().parents[1]
INP         = ROOT / "data" / "derived" / "puzzle_features_mates.parquet"
OUT         = ROOT / "data" / "derived" / "puzzle_features_mates_clustered.parquet"
OUT_SUMMARY = ROOT / "data" / "derived" / "cluster_summary.csv"
OUT_PLOTS   = ROOT / "data" / "derived"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_CLUSTERS   = 10      # number of clusters — tune this
PCA_VARIANCE = 0.95    # keep components explaining 95% of variance
SAMPLE_SIZE  = 200_000 # use a sample for silhouette scoring (expensive on 4M rows)
RANDOM_STATE = 42

# Metadata columns — excluded from feature matrix
META_COLS = [
    "puzzle_id", "site", "ply", "board_fen", "turn",
    "pockets_white_raw", "pockets_black_raw",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_feature_cols(df: pd.DataFrame) -> list:
    """Return numeric feature columns, excluding metadata."""
    return [c for c in df.columns
            if c not in META_COLS
            and pd.api.types.is_numeric_dtype(df[c])]


def describe_cluster(df_cluster: pd.DataFrame, feat_cols: list) -> pd.Series:
    """Return mean of each feature for a cluster — useful for naming."""
    return df_cluster[feat_cols].mean()


def name_cluster(means: pd.Series) -> str:
    """
    Heuristic cluster name based on dominant features.
    Looks at drop flags, mate flags, first-action type, pocket size.
    """
    parts = []

    # First action
    first_cols = {c: means[c] for c in means.index if c.startswith("dyn_first_") and means[c] > 0.3}
    if first_cols:
        dominant = max(first_cols, key=first_cols.get)
        parts.append(dominant.replace("dyn_first_", "first:"))

    # Drop flag
    if means.get("DG_!F_drop", 0) > 0.5:
        parts.append("hasDrop")
    else:
        parts.append("noDrop")

    # Drop+ (drop check)
    if means.get("DG_!F_drop+", 0) > 0.4:
        parts.append("dropCheck")

    # Pocket size
    solver_pocket = means.get("PK_total_solver", 0)
    if solver_pocket > 2:
        parts.append(f"pocket>{solver_pocket:.1f}")
    elif solver_pocket < 0.5:
        parts.append("emptyPocket")

    # PV length
    if means.get("dyn_pvLen_short", 0) > 0.5:
        parts.append("shortMate")
    elif means.get("dyn_pvLen_long", 0) > 0.4:
        parts.append("longMate")

    # Drop piece types
    for p in "NBRQP":
        if means.get(f"dyn_dropHas_{p}", 0) > 0.4:
            parts.append(f"drops{p}")

    return " | ".join(parts) if parts else "misc"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ── 1) Load ──────────────────────────────────────────────────────────────
    print(f"Loading: {INP}")
    df = pd.read_parquet(INP)
    print(f"Shape: {df.shape}")

    feat_cols = get_feature_cols(df)
    print(f"Feature columns: {len(feat_cols)}")
    print(f"Sample features: {feat_cols[:10]}")
    print()

    X = df[feat_cols].values.astype(np.float32)

    # ── 2) Standardize ───────────────────────────────────────────────────────
    print("Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── 3) PCA ───────────────────────────────────────────────────────────────
    print(f"Running PCA (keeping {PCA_VARIANCE*100:.0f}% variance)...")
    pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_scaled)
    print(f"PCA components: {X_pca.shape[1]} (from {X_scaled.shape[1]})")
    print(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
    print()

    # ── 4) Elbow plot (optional, quick) ──────────────────────────────────────
    print("Computing elbow curve (k=2..15)...")
    inertias = []
    ks = range(2, 16)
    for k in ks:
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE,
                             batch_size=10_000, n_init=3)
        km.fit(X_pca)
        inertias.append(km.inertia_)
        print(f"  k={k:2d}  inertia={km.inertia_:.1f}")

    plt.figure(figsize=(8, 4))
    plt.plot(list(ks), inertias, 'bo-')
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow curve — Crazyhouse mate positions")
    plt.tight_layout()
    elbow_path = OUT_PLOTS / "elbow_curve.png"
    plt.savefig(elbow_path, dpi=150)
    plt.close()
    print(f"Saved elbow curve: {elbow_path}")
    print()

    # ── 5) Final clustering ───────────────────────────────────────────────────
    print(f"Clustering with k={N_CLUSTERS} (MiniBatchKMeans)...")
    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        batch_size=10_000,
        n_init=5,
        max_iter=300,
    )
    labels = kmeans.fit_predict(X_pca)
    df["cluster"] = labels
    print(f"Cluster sizes:")
    print(df["cluster"].value_counts().sort_index().to_string())
    print()

    # ── 6) Silhouette score on sample ─────────────────────────────────────────
    print(f"Computing silhouette score on {SAMPLE_SIZE:,} sample...")
    idx = np.random.default_rng(RANDOM_STATE).choice(
        len(X_pca), size=min(SAMPLE_SIZE, len(X_pca)), replace=False)
    sil = silhouette_score(X_pca[idx], labels[idx], sample_size=None)
    print(f"Silhouette score: {sil:.4f}  (range -1..1, higher=better)")
    print()

    # ── 7) Name clusters ──────────────────────────────────────────────────────
    print("Cluster profiles:")
    cluster_names = {}
    summary_rows  = []

    for c in range(N_CLUSTERS):
        mask  = df["cluster"] == c
        means = describe_cluster(df[mask], feat_cols)
        name  = name_cluster(means)
        cluster_names[c] = name
        size  = mask.sum()

        row = {"cluster": c, "size": size, "name": name,
               "silhouette": sil}
        # Add top feature means
        top = means.nlargest(10)
        for feat, val in top.items():
            row[f"mean_{feat}"] = round(val, 3)
        summary_rows.append(row)

        print(f"  Cluster {c:2d} ({size:7,} pos): {name}")
        print(f"    top features: {dict(top.round(3))}")

    df["cluster_name"] = df["cluster"].map(cluster_names)

    # ── 8) PCA scatter plot (first 2 components) ──────────────────────────────
    print("\nGenerating PCA scatter plot...")
    # Sample for plot
    plot_n = min(50_000, len(df))
    plot_idx = np.random.default_rng(RANDOM_STATE).choice(len(df), plot_n, replace=False)
    plot_df = pd.DataFrame({
        "PC1":     X_pca[plot_idx, 0],
        "PC2":     X_pca[plot_idx, 1],
        "cluster": labels[plot_idx].astype(str),
    })

    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="cluster",
                    palette="tab10", alpha=0.3, s=5, linewidth=0)
    plt.title(f"Crazyhouse mate positions — PCA + KMeans (k={N_CLUSTERS})")
    plt.tight_layout()
    scatter_path = OUT_PLOTS / "pca_clusters.png"
    plt.savefig(scatter_path, dpi=150)
    plt.close()
    print(f"Saved scatter plot: {scatter_path}")

    # ── 9) Feature heatmap per cluster ────────────────────────────────────────
    print("Generating cluster feature heatmap...")
    # Pick most discriminative features
    important_feats = [
        "DG_!F_checkmate", "DG_!F_drop", "DG_!F_drop+", "DG_!F_px",
        "DG_!F_+", "DG_!F_noDrop",
        "dyn_first_drop", "dyn_first_dropCheck", "dyn_first_capture",
        "dyn_first_capCheck", "dyn_first_mate",
        "dyn_pvLen_short", "dyn_pvLen_med", "dyn_pvLen_long",
        "dyn_dropHas_N", "dyn_dropHas_B", "dyn_dropHas_R",
        "dyn_dropHas_Q", "dyn_dropHas_P",
        "PK_total_solver", "PK_total_opponent",
        "DS_!drop_sq_zone_kingside", "DS_!drop_sq_zone_queenside",
        "DS_!drop_sq_zone_back_rank",
    ]
    # Keep only cols that exist
    important_feats = [f for f in important_feats if f in df.columns]

    heatmap_data = df.groupby("cluster")[important_feats].mean()
    plt.figure(figsize=(14, 6))
    sns.heatmap(heatmap_data.T, annot=True, fmt=".2f", cmap="YlOrRd",
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f"Mean feature values per cluster (k={N_CLUSTERS})")
    plt.tight_layout()
    heatmap_path = OUT_PLOTS / "cluster_heatmap.png"
    plt.savefig(heatmap_path, dpi=150)
    plt.close()
    print(f"Saved heatmap: {heatmap_path}")

    # ── 10) Save results ──────────────────────────────────────────────────────
    print(f"\nSaving clustered data: {OUT}")
    # Save only metadata + cluster (not full feature matrix to save space)
    out_cols = META_COLS + ["cluster", "cluster_name"]
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_parquet(OUT, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_SUMMARY, index=False)
    print(f"Saved cluster summary: {OUT_SUMMARY}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"  Silhouette score : {sil:.4f}")
    print(f"  Clusters         : {N_CLUSTERS}")
    print(f"  Total positions  : {len(df):,}")
    print(f"  Output           : {OUT}")
    print(f"  Plots            : {OUT_PLOTS}/elbow_curve.png, pca_clusters.png, cluster_heatmap.png")


if __name__ == "__main__":
    main()