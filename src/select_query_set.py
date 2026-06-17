"""
select_query_set.py
--------------------
Pick a SMALL, SMART set of query positions for an expert to label.

The expert can only review a limited number of queries, so the goal is to spend
that budget on positions that, together, COVER the corpus well rather than N
near-duplicates of the same motif. The selection combines three ideas:

  1. CLUSTER COVERAGE  — use the unsupervised clusters (from
     unsupervised_clustering.py) so every natural tactical group is represented.
     Budget is split across clusters proportional to size, with a floor of one
     per cluster so small/rare motif groups are never dropped.

  2. PROTOTYPE + SPREAD — inside each cluster pick the MEDOID (the row closest to
     the cluster centroid = the most "typical" example) plus a few DIVERSE points
     via farthest-point (k-center) sampling, so the expert sees both the canonical
     case and the edges of each group.

  3. STRATIFY / DE-DUP — drop exact board-FEN duplicates, and (optionally) balance
     across salient attributes if present (has-drop, mating piece, first action,
     rating band). These make the labeled set useful as evaluation ground truth
     because it is not skewed toward one position type.

Output: an .xlsx (and .csv) with one row per selected query:
    FEN | solution_san | solution_uci | source_game | cluster | cluster_name |
    turn | role (prototype/diverse) | dist_to_centroid | + any attribute columns

Usage:
    python select_query_set.py
    (or set RUN below to True and run cell-by-cell in a notebook)

Requirements:
    pip install pandas numpy scikit-learn pyarrow openpyxl
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config — adjust paths/sizes here
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parents[1]
FEATURES      = ROOT / "data" / "derived" / "puzzle_features_mates.parquet"
CLUSTERED     = ROOT / "data" / "derived" / "puzzle_features_mates_clustered.parquet"
CORPUS_JSONL  = ROOT / "data" / "corpus" / "checkmates5_600k.jsonl"   # for solutions
OUT_XLSX      = ROOT / "data" / "derived" / "query_selection.xlsx"
OUT_CSV       = ROOT / "data" / "derived" / "query_selection.csv"

N_QUERIES     = 60      # total positions to hand the expert
DIVERSE_PER   = 2       # extra "spread" picks per cluster (beyond the medoid)
RANDOM_STATE  = 42
N_CLUSTERS_FALLBACK = 10  # used only if no clustered file is found
CLUSTER_SAMPLE = 8000   # max rows per cluster used for the distance geometry
                        # (bounds memory/time; selecting ~60 queries needs no more)
STD_SAMPLE     = 200_000  # rows sampled to estimate per-feature std (for scaling)

# Metadata columns (same set as unsupervised_clustering.py) — not features
META_COLS = [
    "puzzle_id", "site", "ply", "board_fen", "turn",
    "pockets_white_raw", "pockets_black_raw", "cluster", "cluster_name",
]

RUN = (__name__ == "__main__")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_frame() -> tuple[pd.DataFrame, list[str]]:
    """Load features and attach cluster labels. Returns (df, feature_cols)."""
    if not FEATURES.exists():
        raise FileNotFoundError(
            f"Features parquet not found: {FEATURES}\n"
            "Run extract_features_crazyhouse / to_parquet first."
        )
    df = pd.read_parquet(FEATURES)

    # attach cluster labels if the clustered file exists, else cluster on the fly
    if CLUSTERED.exists():
        cl = pd.read_parquet(CLUSTERED)
        keep = [c for c in ("puzzle_id", "cluster", "cluster_name") if c in cl.columns]
        df = df.merge(cl[keep], on="puzzle_id", how="left")
    if "cluster" not in df.columns or df["cluster"].isna().all():
        print("No cluster labels found — clustering on the fly "
              f"(k={N_CLUSTERS_FALLBACK}).")
        df["cluster"] = _cluster_on_the_fly(df)
    df["cluster"] = df["cluster"].fillna(-1).astype(int)
    if "cluster_name" not in df.columns:
        df["cluster_name"] = df["cluster"].map(lambda c: f"cluster_{c}")

    feat_cols = [c for c in df.columns
                 if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]
    return df, feat_cols


def _cluster_on_the_fly(df: pd.DataFrame) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import MiniBatchKMeans
    feat_cols = [c for c in df.columns
                 if c not in META_COLS and pd.api.types.is_numeric_dtype(df[c])]
    X = df[feat_cols].values.astype(np.float32)
    X = StandardScaler().fit_transform(X)
    X = PCA(n_components=0.95, random_state=RANDOM_STATE).fit_transform(X)
    km = MiniBatchKMeans(n_clusters=N_CLUSTERS_FALLBACK,
                         random_state=RANDOM_STATE, n_init=3)
    return km.fit_predict(X)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def _farthest_point_order(X: np.ndarray, start: int, k: int) -> list[int]:
    """Greedy k-center: indices spread as far apart as possible, seeded at `start`."""
    n = X.shape[0]
    chosen = [start]
    if k <= 1 or n <= 1:
        return chosen[:k]
    d = np.linalg.norm(X - X[start], axis=1)
    while len(chosen) < min(k, n):
        nxt = int(np.argmax(d))
        if nxt in chosen:
            break
        chosen.append(nxt)
        d = np.minimum(d, np.linalg.norm(X - X[nxt], axis=1))
    return chosen


def select(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    # de-dup identical positions before anything else
    df = df.drop_duplicates(subset="board_fen").reset_index(drop=True)

    # Standardisation scale from a bounded random sample (column-wise std).
    # We never materialise the whole N×F matrix — geometry runs only on a small
    # per-cluster sample, which is plenty for picking a few representative queries.
    std_n = min(len(df), STD_SAMPLE)
    std_rows = df.sample(std_n, random_state=RANDOM_STATE) if std_n < len(df) else df
    stds = std_rows[feat_cols].to_numpy(dtype=np.float32)
    stds = np.nan_to_num(stds, nan=0.0, posinf=0.0, neginf=0.0).std(axis=0)
    inv_std = np.divide(1.0, stds, out=np.zeros_like(stds), where=stds > 0)
    inv_std[~np.isfinite(inv_std)] = 0.0
    inv_std = inv_std.astype(np.float32)

    clusters = sorted(df["cluster"].unique())
    sizes = df["cluster"].value_counts()

    # budget per cluster: floor of 1, remainder proportional to cluster size
    budget = {c: 1 for c in clusters}
    remaining = max(0, N_QUERIES - len(clusters))
    total = sizes.reindex(clusters).fillna(0).sum() or 1
    for c in clusters:
        budget[c] += int(round(remaining * (sizes.get(c, 0) / total)))

    rng = np.random.default_rng(RANDOM_STATE)
    picks = []  # (row_index, role, dist_to_centroid)
    for c in clusters:
        idx_all = df.index[df["cluster"] == c].to_numpy()
        if len(idx_all) == 0:
            continue
        # cap the rows used for geometry so memory/time stay bounded on 5M-row data
        if len(idx_all) > CLUSTER_SAMPLE:
            idx = idx_all[rng.choice(len(idx_all), CLUSTER_SAMPLE, replace=False)]
        else:
            idx = idx_all

        Xraw = df.loc[idx, feat_cols].to_numpy(dtype=np.float32)
        Xraw = np.nan_to_num(Xraw, nan=0.0, posinf=0.0, neginf=0.0)
        cen  = Xraw.mean(axis=0)          # centroid estimated from the cluster sample

        # distance to the centroid, in standardised space (scale by 1/std)
        dist = np.linalg.norm((Xraw - cen) * inv_std, axis=1)
        medoid_local = int(np.argmin(dist))            # prototype = most typical

        # standardised coords for farthest-point sampling (mean-shift is irrelevant)
        Z = Xraw * inv_std
        want = max(1, budget[c])
        order_local = _farthest_point_order(Z, medoid_local, want + DIVERSE_PER)
        for rank, loc in enumerate(order_local[:want]):
            picks.append((int(idx[loc]),
                          "prototype" if rank == 0 else "diverse",
                          float(dist[loc])))

    # trim to N_QUERIES: keep prototypes, then the most-diverse extras
    picks.sort(key=lambda p: (p[1] != "prototype", -p[2]))
    picks = picks[:N_QUERIES]

    rows = []
    for ridx, role, dist in picks:
        r = df.loc[ridx]
        def _joinval(v):
            return " ".join(v) if isinstance(v, (list, tuple, np.ndarray)) else (v if v is not None else "")
        rows.append({
            "puzzle_id":    r.get("puzzle_id", ""),
            "FEN":          r.get("board_fen", ""),
            "turn":         r.get("turn", ""),
            "cluster":      int(r.get("cluster", -1)),
            "cluster_name": r.get("cluster_name", ""),
            "role":         role,
            "dist_to_centroid": round(dist, 3),
            "site":         r.get("site", ""),
            "ply":          r.get("ply", ""),
            # use solutions straight from the parquet if they happen to be there
            "solution_san": _joinval(r.get("solution_san")) if "solution_san" in df.columns else "",
            "solution_uci": _joinval(r.get("solution_uci")) if "solution_uci" in df.columns else "",
        })
    out = pd.DataFrame(rows)
    return out


# ---------------------------------------------------------------------------
# Enrich: source game URL + solution from the corpus
# ---------------------------------------------------------------------------
def _source_url(site: str, ply) -> str:
    site = str(site or "")
    if "lichess.org" in site:
        return f"{site}#{ply}" if ply not in ("", None) else site
    return site


def add_solutions(out: pd.DataFrame) -> pd.DataFrame:
    out["source_game"] = [
        _source_url(s, p) for s, p in zip(out["site"], out["ply"])
    ]
    if "solution_san" not in out.columns: out["solution_san"] = ""
    if "solution_uci" not in out.columns: out["solution_uci"] = ""
    out["solution_san"] = out["solution_san"].fillna("")
    out["solution_uci"] = out["solution_uci"].fillna("")

    # already filled from the parquet? then no need to touch the corpus
    if (out["solution_san"].astype(str).str.len() > 0).all():
        return out
    if not CORPUS_JSONL.exists():
        print(f"Corpus not found ({CORPUS_JSONL}) — missing solutions left blank. "
              "Set CORPUS_JSONL to checkmates5_600k.jsonl to fill them in.")
        return out

    # build the set of keys we still need, then stream the corpus once
    need = {str(pid) for pid in out["puzzle_id"]}
    need |= {_source_url(s, p) for s, p in zip(out["site"], out["ply"])}
    found = {}  # key -> (san, uci)
    sample_keys = []
    with open(CORPUS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            # try several id forms so this is robust to the corpus schema
            keys = [
                str(rec.get("id", "")),
                str(rec.get("puzzle_id", "")),
                f'{rec.get("site","")}#{rec.get("ply","")}',
                f'{rec.get("site","")}_{rec.get("ply","")}',
            ]
            if len(sample_keys) < 5:
                sample_keys.append(keys[0] or keys[2])
            san = rec.get("solution_san") or rec.get("solution") or []
            uci = rec.get("solution_uci") or []
            for k in keys:
                if k and k in need:
                    found[k] = (san, uci)

    def lookup(pid, site, ply):
        for k in (str(pid), _source_url(site, ply),
                  f"{site}_{ply}", f"{site}#{ply}"):
            if k in found:
                return found[k]
        return (None, None)

    sans, ucis = list(out["solution_san"].astype(str)), list(out["solution_uci"].astype(str))
    for i, (pid, site, ply) in enumerate(zip(out["puzzle_id"], out["site"], out["ply"])):
        if sans[i]:                      # keep solutions already provided by the parquet
            continue
        san, uci = lookup(pid, site, ply)
        sans[i] = " ".join(san) if isinstance(san, list) else (san or "")
        ucis[i] = " ".join(uci) if isinstance(uci, list) else (uci or "")
    out["solution_san"] = sans
    out["solution_uci"] = ucis

    n_missing = int((out["solution_san"].astype(str).str.len() == 0).sum())
    if n_missing:
        print(f"{n_missing}/{len(out)} solutions not matched. "
              f"Corpus key samples: {sample_keys}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df, feat_cols = load_frame()
    print(f"Loaded {len(df):,} positions, {len(feat_cols)} feature columns, "
          f"{df['cluster'].nunique()} clusters.")
    out = select(df, feat_cols)
    out = add_solutions(out)

    # final column order — FEN / solution / source game first, as the mentor asked
    cols = ["FEN", "solution_san", "solution_uci", "source_game",
            "cluster", "cluster_name", "turn", "role", "dist_to_centroid",
            "puzzle_id"]
    out = out[[c for c in cols if c in out.columns]]

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    try:
        out.to_excel(OUT_XLSX, index=False, engine="openpyxl")
        print(f"Wrote {len(out)} queries → {OUT_XLSX}")
    except Exception as e:
        print(f"xlsx write failed ({e}); CSV is at {OUT_CSV}")
    print(f"Also wrote {OUT_CSV}")
    print("\nPer-cluster counts in the selection:")
    print(out["cluster"].value_counts().sort_index().to_string())


if RUN:
    main()