"""
unsupervised_clustering.py
--------------------------
Unsupervised learning on puzzle_features_checkmates5_deduped.parquet.
Clusters Crazyhouse mate positions to discover natural tactical groups
without any labels.

Memory strategy (avoids the 9 GiB OOM on 5M × 247 float64):
  • Scaler + PCA are FIT on a bounded sample (FIT_SAMPLE rows, float32).
  • The full dataset is TRANSFORMED + clustered in chunks — peak RAM is
    roughly  chunk_size × n_components × 4 bytes, well under 1 GB.
  • Silhouette is computed on a sample (SILL_SAMPLE), not the full set.

Steps:
  1. Load metadata + feature column list (no feature values yet)
  2. Load FIT_SAMPLE rows → fit StandardScaler + IncrementalPCA
  3. Transform full dataset in CHUNK_SIZE chunks → assign cluster labels
  4. Analyse / name clusters (mean features per cluster, computed in chunks)
  5. Generate plots (elbow, PCA scatter, heatmap)
  6. Save clustered parquet (metadata + cluster columns only)

Usage:
    python unsupervised_clustering.py

Requirements:
    pip install pandas numpy scikit-learn pyarrow matplotlib seaborn
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths  — edit ROOT if you run from outside the src/ directory
# ---------------------------------------------------------------------------
ROOT        = Path(r"C:\Users\tgrum\Desktop\magistrska\crazyhouse")
DERIVED     = ROOT / "data" / "derived"
INP         = DERIVED / "puzzle_features_checkmates5_deduped.parquet"
OUT         = DERIVED / "puzzle_features_checkmates5_clustered_deduped.parquet"
OUT_SUMMARY = DERIVED / "cluster_summary.csv"
OUT_PLOTS   = DERIVED

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_CLUSTERS   = 10        # number of clusters — tune after seeing the elbow curve
PCA_VARIANCE = 0.95      # keep components explaining this fraction of variance
FIT_SAMPLE   = 300_000   # rows used to FIT scaler + PCA  (keeps RAM low)
CHUNK_SIZE   = 50_000    # rows processed at a time during transform + predict
SILL_SAMPLE  = 50_000    # rows used for silhouette score (expensive)
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
    return [c for c in df.columns
            if c not in META_COLS
            and pd.api.types.is_numeric_dtype(df[c])]


def name_cluster(means: pd.Series) -> str:
    """Heuristic label based on dominant features."""
    parts = []
    first_cols = {c: means[c] for c in means.index
                  if c.startswith("dyn_first_") and means[c] > 0.3}
    if first_cols:
        parts.append(max(first_cols, key=first_cols.get).replace("dyn_first_", "first:"))
    if means.get("DG_!F_drop", 0) > 0.5:
        parts.append("hasDrop")
    else:
        parts.append("noDrop")
    if means.get("DG_!F_drop+", 0) > 0.4:
        parts.append("dropCheck")
    sp = means.get("PK_total_solver", 0)
    if sp > 2:
        parts.append(f"pocket>{sp:.1f}")
    elif sp < 0.5:
        parts.append("emptyPocket")
    if means.get("dyn_pvLen_short", 0) > 0.5:
        parts.append("shortMate")
    elif means.get("dyn_pvLen_long", 0) > 0.4:
        parts.append("longMate")
    for p in "NBRQP":
        if means.get(f"dyn_dropHas_{p}", 0) > 0.4:
            parts.append(f"drops{p}")
    return " | ".join(parts) if parts else "misc"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(RANDOM_STATE)

    # ── 1) Schema pass — learn column list without loading features ───────────
    print(f"Reading schema: {INP}")
    schema_df = pd.read_parquet(INP, columns=None).iloc[0:0]   # 0 rows, all cols
    all_cols   = list(schema_df.columns)
    feat_cols  = get_feature_cols(schema_df)
    n_total    = pd.read_parquet(INP, columns=["puzzle_id"]).shape[0]
    print(f"Total rows  : {n_total:,}")
    print(f"Feature cols: {len(feat_cols)}")
    print(f"Sample feats: {feat_cols[:8]}")
    print()

    # ── 2) Fit scaler + IncrementalPCA on a sample ───────────────────────────
    fit_n   = min(FIT_SAMPLE, n_total)
    fit_idx = np.sort(rng.choice(n_total, fit_n, replace=False))
    print(f"Fitting scaler + PCA on {fit_n:,} rows (sample)…")

    # Read only the rows we need for fitting (via row-group filter workaround:
    # parquet doesn't support arbitrary row indices, so we read full then sample)
    fit_df  = pd.read_parquet(INP, columns=feat_cols)
    fit_X   = fit_df.iloc[fit_idx][feat_cols].fillna(0).values.astype(np.float32)
    del fit_df  # free memory immediately

    scaler  = StandardScaler()
    fit_X_s = scaler.fit_transform(fit_X).astype(np.float32)
    del fit_X

    # Determine n_components from a small PCA trial
    from sklearn.decomposition import PCA as _PCA
    trial_n = min(len(feat_cols), fit_X_s.shape[0], 200)
    trial   = _PCA(n_components=trial_n, random_state=RANDOM_STATE)
    trial.fit(fit_X_s)
    cumvar  = np.cumsum(trial.explained_variance_ratio_)
    n_comp  = int(np.searchsorted(cumvar, PCA_VARIANCE)) + 1
    n_comp  = max(2, min(n_comp, trial_n))
    print(f"PCA: {n_comp} components explain ≥{PCA_VARIANCE*100:.0f}% variance "
          f"(tried up to {trial_n})")

    # Fit the real IncrementalPCA on the same sample in one batch
    ipca = IncrementalPCA(n_components=n_comp)
    ipca.fit(fit_X_s)
    del fit_X_s
    print(f"Explained variance (sample): {ipca.explained_variance_ratio_.sum():.3f}")
    print()

    # ── 3) Elbow curve (transform sample once, try k=2..15) ──────────────────
    print("Elbow curve on sample (k=2..15)…")
    elbow_df = pd.read_parquet(INP, columns=feat_cols).iloc[fit_idx]
    elbow_X  = ipca.transform(
        scaler.transform(elbow_df.fillna(0).values.astype(np.float32))
    ).astype(np.float32)
    del elbow_df

    inertias = []
    for k in range(2, 16):
        km = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE,
                             batch_size=5_000, n_init=3)
        km.fit(elbow_X)
        inertias.append(km.inertia_)
        print(f"  k={k:2d}  inertia={km.inertia_:.1f}")

    plt.figure(figsize=(8, 4))
    plt.plot(range(2, 16), inertias, "bo-")
    plt.xlabel("k"); plt.ylabel("Inertia")
    plt.title("Elbow curve — Crazyhouse mate positions")
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "elbow_curve.png", dpi=150); plt.close()
    print(f"Saved: {OUT_PLOTS/'elbow_curve.png'}\n")

    # ── 4) Final clustering — fit on sample, predict in full chunks ───────────
    print(f"Fitting final MiniBatchKMeans (k={N_CLUSTERS}) on sample…")
    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS, random_state=RANDOM_STATE,
        batch_size=10_000, n_init=5, max_iter=300,
    )
    kmeans.fit(elbow_X)
    del elbow_X

    print(f"Predicting cluster labels in chunks of {CHUNK_SIZE:,}…")
    all_labels = np.empty(n_total, dtype=np.int32)
    full_feat  = pd.read_parquet(INP, columns=feat_cols)   # load features once
    for start in range(0, n_total, CHUNK_SIZE):
        end   = min(start + CHUNK_SIZE, n_total)
        chunk = full_feat.iloc[start:end].fillna(0).values.astype(np.float32)
        chunk_s   = scaler.transform(chunk).astype(np.float32)
        chunk_pca = ipca.transform(chunk_s).astype(np.float32)
        all_labels[start:end] = kmeans.predict(chunk_pca)
        if (start // CHUNK_SIZE) % 20 == 0:
            print(f"  …{end:,}/{n_total:,}")
    del full_feat
    print()

    # ── 5) Load metadata + attach labels ─────────────────────────────────────
    meta_read = [c for c in META_COLS if c in all_cols]
    print(f"Loading metadata columns: {meta_read}")
    df = pd.read_parquet(INP, columns=meta_read)
    df["cluster"] = all_labels

    print("Cluster sizes:")
    print(df["cluster"].value_counts().sort_index().to_string())
    print()

    # ── 6) Silhouette on sample ───────────────────────────────────────────────
    sil_n   = min(SILL_SAMPLE, n_total)
    sil_idx = rng.choice(n_total, sil_n, replace=False)
    print(f"Silhouette score on {sil_n:,} sample…")
    feat_sil = pd.read_parquet(INP, columns=feat_cols).iloc[sil_idx]
    X_sil    = ipca.transform(
        scaler.transform(feat_sil.fillna(0).values.astype(np.float32))
    ).astype(np.float32)
    del feat_sil
    sil = silhouette_score(X_sil, all_labels[sil_idx])
    del X_sil
    print(f"Silhouette: {sil:.4f}  (range −1…1, higher = better)\n")

    # ── 7) Name clusters (compute feature means per cluster, chunked) ─────────
    print("Computing per-cluster feature means (chunked)…")
    sum_    = np.zeros((N_CLUSTERS, len(feat_cols)), dtype=np.float64)
    counts  = np.zeros(N_CLUSTERS, dtype=np.int64)
    feat_df = pd.read_parquet(INP, columns=feat_cols)
    for start in range(0, n_total, CHUNK_SIZE):
        end    = min(start + CHUNK_SIZE, n_total)
        chunk  = feat_df.iloc[start:end].fillna(0).values.astype(np.float32)
        lbls   = all_labels[start:end]
        for c in range(N_CLUSTERS):
            mask = lbls == c
            if mask.any():
                sum_[c]    += chunk[mask].sum(axis=0)
                counts[c]  += mask.sum()
    del feat_df
    cluster_means = pd.DataFrame(
        sum_ / counts[:, None], columns=feat_cols
    )

    cluster_names  = {}
    summary_rows   = []
    print("Cluster profiles:")
    for c in range(N_CLUSTERS):
        means = cluster_means.iloc[c]
        name  = name_cluster(means)
        cluster_names[c] = name
        top   = means.nlargest(10)
        row   = {"cluster": c, "size": int(counts[c]),
                 "name": name, "silhouette": round(sil, 4)}
        for f, v in top.items():
            row[f"mean_{f}"] = round(v, 3)
        summary_rows.append(row)
        print(f"  Cluster {c:2d} ({int(counts[c]):7,}): {name}")
        print(f"    top: {dict(top.round(3))}")
    df["cluster_name"] = df["cluster"].map(cluster_names)
    print()

    # ── 8) PCA scatter (transform a small sample for the plot) ────────────────
    print("PCA scatter plot…")
    plot_n   = min(30_000, n_total)
    plot_idx = rng.choice(n_total, plot_n, replace=False)
    feat_plot = pd.read_parquet(INP, columns=feat_cols).iloc[plot_idx]
    X_plot    = ipca.transform(
        scaler.transform(feat_plot.fillna(0).values.astype(np.float32))
    ).astype(np.float32)
    del feat_plot

    plot_df = pd.DataFrame({
        "PC1": X_plot[:, 0], "PC2": X_plot[:, 1],
        "cluster": all_labels[plot_idx].astype(str),
    })
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="cluster",
                    palette="tab10", alpha=0.3, s=5, linewidth=0)
    plt.title(f"Crazyhouse mate positions — PCA + KMeans (k={N_CLUSTERS})")
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "pca_clusters.png", dpi=150); plt.close()
    del X_plot
    print(f"Saved: {OUT_PLOTS/'pca_clusters.png'}")

    # ── 9) Feature heatmap per cluster ────────────────────────────────────────
    print("Cluster feature heatmap…")
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
    important_feats = [f for f in important_feats if f in feat_cols]
    if important_feats:
        heatmap_data = cluster_means[important_feats]
        plt.figure(figsize=(14, 6))
        sns.heatmap(heatmap_data.T, annot=True, fmt=".2f", cmap="YlOrRd",
                    linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title(f"Mean feature values per cluster (k={N_CLUSTERS})")
        plt.tight_layout()
        plt.savefig(OUT_PLOTS / "cluster_heatmap.png", dpi=150); plt.close()
        print(f"Saved: {OUT_PLOTS/'cluster_heatmap.png'}")

    # ── 10) Save ──────────────────────────────────────────────────────────────
    print(f"\nSaving clustered parquet: {OUT}")
    out_cols = [c for c in META_COLS + ["cluster", "cluster_name"] if c in df.columns]
    df[out_cols].to_parquet(OUT, index=False)

    pd.DataFrame(summary_rows).to_csv(OUT_SUMMARY, index=False)
    print(f"Saved cluster summary: {OUT_SUMMARY}")

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Silhouette score : {sil:.4f}")
    print(f"  Clusters         : {N_CLUSTERS}")
    print(f"  Total positions  : {n_total:,}")
    print(f"  Output           : {OUT}")


if __name__ == "__main__":
    main()