"""
select_query_set.py
-------------------
Sample N_QUERIES positions from the corpus for expert labelling.

Selection is cluster-proportional (from unsupervised_clustering.py) so
every natural tactical group is represented. Within each cluster the
MEDOID (most typical example) is picked first, then diverse spread via
farthest-point sampling.

Output columns (only):
    fen            — full Crazyhouse FEN (board/pockets turn - - 0 1)
    solution       — numbered figurine SAN, e.g. "1...♛xf3+ 2.♞@g3 ♝@g2+"
    solution_uci   — raw UCI tokens, space-separated
    lichess        — direct link to the source game on Lichess

Run:
    python select_query_set.py
"""

from pathlib import Path
import json, re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT         = Path(r"C:\Users\tgrum\Desktop\magistrska\crazyhouse")
DERIVED      = ROOT / "data" / "derived"

FEATURES     = DERIVED / "puzzle_features_checkmates5_deduped.parquet"
CLUSTERED    = DERIVED / "puzzle_features_checkmates5_clustered_deduped.parquet"
CORPUS_JSONL = DERIVED / "corpus_checkmates5.jsonl"   # has id + solution_san/uci

OUT_XLSX     = DERIVED / "query_selection.xlsx"
OUT_CSV      = DERIVED / "query_selection.csv"

N_QUERIES      = 500
CLUSTER_SAMPLE = 8_000   # max rows per cluster for geometry
STD_SAMPLE     = 200_000 # rows used to estimate per-feature std
RANDOM_STATE   = 42

META_COLS = [
    "puzzle_id", "site", "ply", "board_fen", "turn",
    "pockets_white_raw", "pockets_black_raw",
    "cluster", "cluster_name",
]

_FIG = {"K": "♚", "Q": "♛", "R": "♜", "B": "♝", "N": "♞"}

def _figurine(tok: str) -> str:
    tok = re.sub(r"=([KQRBN])", lambda m: "=" + _FIG[m.group(1)], tok)
    tok = re.sub(r"^([KQRBN])", lambda m: _FIG[m.group(1)], tok)
    return tok

def _fmt_solution(san_list, turn: str) -> str:
    if not san_list:
        return ""
    start_white = not (turn in ("b", "black"))
    parts, n = [], 1
    for i, tok in enumerate(san_list):
        is_white = (i % 2 == 0) if start_white else (i % 2 == 1)
        disp = _figurine(str(tok))
        if is_white:
            prefix = f"{n}."
        elif i == 0:
            prefix = f"{n}\u2026"
        else:
            prefix = ""
        if not is_white:
            n += 1
        parts.append((prefix + disp).strip())
    return " ".join(parts)

# ---------------------------------------------------------------------------
def load_frame():
    print(f"Loading features: {FEATURES}")
    df = pd.read_parquet(FEATURES)
    print(f"  {len(df):,} rows, {len(df.columns)} columns")

    if not CLUSTERED.exists():
        raise FileNotFoundError(
            f"Clustered parquet not found: {CLUSTERED}\n"
            "Run unsupervised_clustering.py first."
        )
    cl = pd.read_parquet(CLUSTERED)
    keep = [c for c in ("puzzle_id", "cluster", "cluster_name") if c in cl.columns]
    df = df.merge(cl[keep], on="puzzle_id", how="left")
    df["cluster"] = df["cluster"].fillna(-1).astype(int)
    if "cluster_name" not in df.columns:
        df["cluster_name"] = df["cluster"].map(lambda c: f"cluster_{c}")

    feat_cols = [c for c in df.columns
                 if c not in META_COLS
                 and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  Feature cols: {len(feat_cols)}, clusters: {df['cluster'].nunique()}")
    return df, feat_cols

# ---------------------------------------------------------------------------
def _farthest_points(X, start, k):
    chosen = [start]
    if k <= 1 or len(X) <= 1:
        return chosen
    dist = np.linalg.norm(X - X[start], axis=1)
    while len(chosen) < min(k, len(X)):
        nxt = int(np.argmax(dist))
        if nxt in chosen:
            break
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(X - X[nxt], axis=1))
    return chosen

def select(df, feat_cols):
    df = df.drop_duplicates(subset="board_fen").reset_index(drop=True)
    print(f"After board-FEN dedup: {len(df):,} rows")
    rng = np.random.default_rng(RANDOM_STATE)

    # per-feature scaling
    std_rows = df.sample(min(STD_SAMPLE, len(df)), random_state=RANDOM_STATE)
    stds = np.nan_to_num(std_rows[feat_cols].to_numpy(dtype=np.float32)).std(axis=0)
    inv_std = np.divide(1.0, stds, out=np.zeros_like(stds), where=stds > 0).astype(np.float32)

    # proportional budget
    clusters = sorted(df["cluster"].unique())
    sizes    = df["cluster"].value_counts().reindex(clusters).fillna(0).astype(int)
    raw      = (sizes / sizes.sum()) * N_QUERIES
    budget   = np.floor(raw).astype(int)
    for c in (raw - budget).sort_values(ascending=False).index[:N_QUERIES - budget.sum()]:
        budget[c] += 1
    budget = budget.to_dict()

    picks = []
    for c in clusters:
        want = budget.get(c, 0)
        if want <= 0:
            continue
        idx_all = df.index[df["cluster"] == c].to_numpy()
        if len(idx_all) == 0:
            continue
        geom_idx = idx_all
        if len(idx_all) > CLUSTER_SAMPLE:
            geom_idx = idx_all[rng.choice(len(idx_all), CLUSTER_SAMPLE, replace=False)]
        Xr = np.nan_to_num(df.loc[geom_idx, feat_cols].to_numpy(dtype=np.float32))
        Xs = Xr * inv_std
        centroid = Xs.mean(axis=0)
        dists = np.linalg.norm(Xs - centroid, axis=1)
        medoid = int(np.argmin(dists))
        order = _farthest_points(Xs, medoid, min(want, len(Xs)))
        for rank, loc in enumerate(order):
            picks.append((int(geom_idx[loc]), "prototype" if rank == 0 else "diverse"))

    # dedup + fill shortfall
    seen, unique = set(), []
    for p in picks:
        if p[0] not in seen:
            unique.append(p); seen.add(p[0])
    if len(unique) < N_QUERIES:
        extra_pool = np.setdiff1d(np.arange(len(df)), [p[0] for p in unique])
        for ridx in rng.choice(extra_pool, N_QUERIES - len(unique), replace=False):
            unique.append((int(ridx), "extra"))
    picks = unique[:N_QUERIES]

    keep_cols = [c for c in
                 ["puzzle_id", "site", "ply", "board_fen", "turn",
                  "pockets_white_raw", "pockets_black_raw", "cluster"]
                 if c in df.columns]
    result = df.loc[[p[0] for p in picks], keep_cols].copy()
    result["role"] = [p[1] for p in picks]
    return result.reset_index(drop=True)

# ---------------------------------------------------------------------------
def _lichess_url(site, ply):
    site = str(site or "").strip()
    if not site or site == "nan":
        return ""
    base = re.sub(r"_(mate\d+|\d+)$", "", site)
    if not base.startswith("http"):
        base = "https://lichess.org/" + base.split("/")[-1]
    try:
        return f"{base}#{int(ply)}" if ply and str(ply) not in ("", "nan") else base
    except Exception:
        return base

def add_solutions(result):
    print(f"Loading solutions from corpus: {CORPUS_JSONL}")
    if not CORPUS_JSONL.exists():
        print("  !! corpus not found — solutions will be empty")
        result["fen"] = result["board_fen"]
        result["solution"] = ""
        result["solution_uci"] = ""
        result["lichess"] = result.apply(
            lambda r: _lichess_url(r.get("site",""), r.get("ply","")), axis=1)
        return result

    # build lookup keys from the result rows
    need = set(result["puzzle_id"].astype(str))
    for _, row in result.iterrows():
        s = str(row.get("site", "") or "")
        p = str(row.get("ply",  "") or "")
        if s:
            need.update([s, f"{s}_{p}", f"{s}_mate5"])

    found = {}   # key -> dict with solution_san, solution_uci, turn, board_fen, pockets
    n_read = 0
    ORDER = "QRBNP"
    def pstr(d, up):
        s = ""
        for pc in ORDER:
            n = int((d or {}).get(pc, 0) or (d or {}).get(pc.lower(), 0) or 0)
            s += (pc if up else pc.lower()) * n
        return s

    with open(CORPUS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_read += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            doc_id = str(rec.get("id") or rec.get("puzzle_id") or "")
            site   = str(rec.get("site") or "")
            ply    = str(rec.get("ply")  or "")
            keys = {k for k in [doc_id, site, f"{site}_{ply}", f"{site}_mate5"] if k}
            if not (keys & need):
                continue
            pw  = rec.get("pockets_white") or {}
            pb  = rec.get("pockets_black") or {}
            brd = (rec.get("board_fen") or "").split(" ")[0].split("/")
            brd = "/".join(brd[:8])
            pocket = pstr(pw, True) + pstr(pb, False)
            tc = "b" if str(rec.get("turn","white")).lower().startswith("b") else "w"
            fen = f"{brd}/{pocket} {tc} - - 0 1" if pocket else f"{brd} {tc} - - 0 1"
            payload = {
                "solution_san": rec.get("solution_san") or [],
                "solution_uci": rec.get("solution_uci") or [],
                "turn":  rec.get("turn", "white"),
                "fen":   fen,
            }
            for k in keys:
                found[k] = payload
    print(f"  Read {n_read:,} lines, matched {len(found)} keys")

    fen_col, sol_col, uci_col, url_col = [], [], [], []
    for _, row in result.iterrows():
        pid  = str(row["puzzle_id"])
        site = str(row.get("site", "") or "")
        ply  = str(row.get("ply",  "") or "")
        turn = str(row.get("turn", "white") or "white")

        payload = (found.get(pid)
                   or found.get(site)
                   or found.get(f"{site}_{ply}")
                   or found.get(f"{site}_mate5"))

        if payload:
            san  = payload["solution_san"]
            uci  = payload["solution_uci"]
            t    = payload["turn"]
            fen  = payload["fen"]
        else:
            san, uci, t = [], [], turn
            # fall back to what the parquet stored
            pw_raw = str(row.get("pockets_white_raw", "") or "")
            pb_raw = str(row.get("pockets_black_raw", "") or "")
            def parse_raw(s):
                try: return json.loads(s)
                except: return {}
            pw, pb = parse_raw(pw_raw), parse_raw(pb_raw)
            brd = (row["board_fen"] or "").split(" ")[0].split("/")
            brd = "/".join(brd[:8])
            pocket = pstr(pw, True) + pstr(pb, False)
            tc = "b" if turn.lower().startswith("b") else "w"
            fen = f"{brd}/{pocket} {tc} - - 0 1" if pocket else f"{brd} {tc} - - 0 1"

        fen_col.append(fen)
        sol_col.append(_fmt_solution(san, t))
        uci_col.append(" ".join(uci) if isinstance(uci, list) else str(uci or ""))
        url_col.append(_lichess_url(site, ply))

    result["fen"]          = fen_col
    result["solution"]     = sol_col     # numbered figurine SAN
    result["solution_uci"] = uci_col
    result["lichess"]      = url_col
    return result

# ---------------------------------------------------------------------------
def main():
    df, feat_cols = load_frame()
    result = select(df, feat_cols)
    result = add_solutions(result)

    n_sol = (result["solution_uci"].str.len() > 0).sum()
    n_url = (result["lichess"].str.len() > 0).sum()
    print(f"\nSolutions filled : {n_sol}/{len(result)}")
    print(f"Lichess URLs     : {n_url}/{len(result)}")
    if n_sol < len(result):
        miss = result[result["solution_uci"].str.len() == 0]["puzzle_id"].head(5).tolist()
        print(f"Sample unmatched : {miss}")

    # final output: ONLY the three columns the expert needs
    out = result[["fen", "solution", "solution_uci", "lichess"]].copy()

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"\nCSV  → {OUT_CSV}")
    try:
        out.to_excel(OUT_XLSX, index=False, engine="openpyxl")
        print(f"XLSX → {OUT_XLSX}")
    except Exception as e:
        print(f"xlsx failed ({e}) — use the CSV")
    print(f"\nSample output:")
    print(out.head(3).to_string())

if __name__ == "__main__":
    main()