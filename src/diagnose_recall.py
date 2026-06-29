"""
diagnose_recall.py
==================
Answers "where did the known-similar games go?" for an eval export.

For every candidate the expert marked "similar", it reports:
  in_pool      - did the wide BM25 retrieval surface it at all?
  bm25_rank    - its position in the pool (None if not retrieved)
  model_prob   - what the supervised model scores the (query, candidate) pair
  model_rank   - its position once the whole pool is rescored by the model
  final_top10  - would it land in the shown top-10?

Reading it:
  not in_pool        -> RECALL problem: BM25 never retrieved it. Raise pool_k or
                        wire the learned family weights (Lever 1). The model can't
                        rank a game it never sees.
  in_pool, low rank  -> RANKING problem: it's there but the model buried it. Tunable.
  in_pool, top10     -> good: it survives to the shown set.

Needs: the Lucene server running, a retrieval model + meta in CACHE_DIR, and
pair_features.py beside this file. No engine needed - the query's dynamic tokens
are taken from the export (what the engine produced at eval time).

Usage:
  python diagnose_recall.py --export ch_export_XXX.jsonl
  python diagnose_recall.py --export ./exports --pool_k 200
  python diagnose_recall.py --export ./exports --model ../data/models/retrieval_model_batch2_iter4_pruned_ExtraTrees.joblib
"""

import argparse
import glob
import json
import os
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pair_features as pf
try:
    import mate_picture as mpx
    _HAS_MPX = True
except Exception:
    _HAS_MPX = False

LUCENE_URL = os.environ.get("LUCENE_URL", "http://localhost:8983")
CACHE_DIR  = os.environ.get("CACHE_DIR", "../data/models")


# --- Lucene -----------------------------------------------------------------

def lucene_search(query_tokens, field="text_all", topk=100):
    params = urllib.parse.urlencode({"q": query_tokens, "field": field,
                                     "topk": topk, "exclude_id": ""})
    with urllib.request.urlopen(f"{LUCENE_URL}/search?{params}", timeout=15) as r:
        return json.loads(r.read().decode())


def lucene_rrf(query_tokens, fields=None, weights=None, topk=100, k=60):
    """Reciprocal Rank Fusion across multiple fields (the /rrf endpoint).
    Returns the same doc-list shape as lucene_search. `fields` and `weights`
    are lists; weights default to all-1. Scores everything as one query string
    against each field independently and fuses by rank, so a doc that matches
    strongly on ONE field (e.g. text_motif) surfaces even if weak on others."""
    p = {"q": query_tokens, "topk": topk, "k": k, "exclude_id": ""}
    if fields:
        p["fields"] = ",".join(fields)
    if weights:
        p["weights"] = ",".join(str(w) for w in weights)
    params = urllib.parse.urlencode(p)
    with urllib.request.urlopen(f"{LUCENE_URL}/rrf?{params}", timeout=20) as r:
        return json.loads(r.read().decode())

def lucene_doc(doc_id):
    params = urllib.parse.urlencode({"id": doc_id})
    try:
        with urllib.request.urlopen(f"{LUCENE_URL}/doc?{params}", timeout=8) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


# --- pockets / records ------------------------------------------------------

def pocket_multiset_from_fen(fen, turn):
    s0 = (fen or "").split(" ")[0]
    seg = ""
    if "[" in s0 and "]" in s0:
        seg = s0[s0.index("[") + 1:s0.index("]")]
    else:
        parts = s0.split("/")
        if len(parts) == 9:
            seg = parts[8]
    w = sorted(c.upper() for c in seg if c.isupper())
    b = sorted(c.upper() for c in seg if c.islower())
    return w if str(turn or "white").startswith("w") else b

def query_rec_from_export(r):
    turn = r.get("query_turn")
    return {
        "text_static":  r.get("query_text_static", ""),
        "text_dynamic": r.get("query_text_dynamic", ""),
        "solution_uci": r.get("query_solution_uci", []),
        "mate":         r.get("query_mate"),
        "turn":         turn,
        "rating":       r.get("query_rating_avg") or 0,
        "fen":          r.get("query_fen", ""),
        "pockets_self": pocket_multiset_from_fen(r.get("query_fen", ""), turn),
    }

def _full_fen_from_doc(doc):
    """Rebuild the full Crazyhouse FEN (8 board ranks + 9th pocket rank + turn)
    from the corpus fields, so serve-time features match the training export
    (which stored candidate_fen in exactly this form). The corpus stores the
    position under 'board_fen', NOT 'fen'."""
    bf = (doc.get("board_fen") or doc.get("fen") or "").split(" ")[0]
    if "[" in bf:
        bf = bf.split("[")[0]
    parts = bf.split("/")
    placement = "/".join(parts[:8]) if len(parts) >= 8 else bf
    pw = doc.get("pockets_white") or {}
    pb = doc.get("pockets_black") or {}
    pocket = "".join(str(k).upper() * max(0, int(v or 0)) for k, v in pw.items())
    pocket += "".join(str(k).lower() * max(0, int(v or 0)) for k, v in pb.items())
    t = "b" if str(doc.get("turn") or "white").lower().startswith("b") else "w"
    base = placement + ("/" + pocket if pocket else "")
    return f"{base} {t} - - 0 1"


def cand_rec_from_doc(doc):
    turn = doc.get("turn")
    def pm(d):
        d = d or {}
        out = []
        for k, v in d.items():
            out += [str(k).upper()] * max(0, int(v or 0))
        return sorted(out)
    pself = pm(doc.get("pockets_white")) if str(turn or "white").startswith("w") else pm(doc.get("pockets_black"))
    return {
        "text_static":  doc.get("text_static", ""),
        "text_dynamic": doc.get("text_dynamic", ""),
        "solution_uci": doc.get("solution_uci", []) if isinstance(doc.get("solution_uci"), list)
                        else json.loads(doc.get("solution_uci") or "[]"),
        "mate":         doc.get("mate_in") or doc.get("mate_before"),
        "turn":         turn,
        "rating":       doc.get("game_rating_avg") or doc.get("meta_avg_rating") or 0,
        "fen":          _full_fen_from_doc(doc),
        "pockets_self": pself,
    }

_FAMILY_WEIGHTS_DEFAULT = {"placement": 2, "relation": 2, "motif": 2, "pocket": 2,
                          "pv": 4, "dyn": 2, "sum": 2, "mate": 3, "km": 2}
def _load_family_weights():
    """Load learned BM25 family weights (same file app.py uses); fall back to defaults."""
    for cand in (os.path.join(CACHE_DIR, "bm25_family_weights.json"),
                 "bm25_family_weights.json"):
        try:
            with open(cand) as fh:
                w = (json.load(fh) or {}).get("weights", {})
            if w:
                return {**_FAMILY_WEIGHTS_DEFAULT, **{k: int(v) for k, v in w.items()}}
        except FileNotFoundError:
            continue
        except Exception:
            break
    return dict(_FAMILY_WEIGHTS_DEFAULT)
FAMILY_WEIGHTS = _load_family_weights()

def wide_query_tokens(r, inject_mate=False):
    """Mirror the app: weight every token family by the LEARNED FAMILY_WEIGHTS.
    Routes every token through pf._family_of so it stays correct if family
    definitions change (e.g. mate:/km: now being their own families rather
    than folded into dyn). With inject_mate, append the query's mate-picture
    AND king-march tokens (computed live from query_fen + query_solution_uci)
    at their own family weights, simulating the deployed encoder so old exports
    can be tested without regeneration."""
    out = []
    # dynamic tokens (already include mate:/km: if the export was regenerated)
    for t in (r.get("query_text_dynamic", "") or "").split():
        fam = pf._family_of(t)
        out.extend([t] * FAMILY_WEIGHTS.get(fam, 1))
    # static tokens, bucketed by family
    buckets = pf.bucket_tokens(r.get("query_text_static", ""))
    for fam in ("placement", "relation", "motif", "pocket"):
        out.extend(list(buckets[fam]) * FAMILY_WEIGHTS.get(fam, 1))
    # live injection of motif tokens for old exports lacking them
    if inject_mate:
        mate_toks = pf.mate_picture(r.get("query_fen", ""),
                                    r.get("query_solution_uci"),
                                    r.get("query_turn")).get("tokens", [])
        km_toks = pf.king_march(r.get("query_fen", ""),
                                r.get("query_solution_uci"),
                                r.get("query_turn")).get("tokens", [])
        out.extend(mate_toks * FAMILY_WEIGHTS.get("mate", 1))
        out.extend(km_toks * FAMILY_WEIGHTS.get("km", 1))
    return " ".join(out)


# default RRF field set + weights (motif emphasised). Mirrors the Java
# /rrf defaults; kept here so the Python side can tune without a Java change.
RRF_FIELDS  = ["text_motif", "text_static", "text_dynamic_general", "text_dynamic_solution"]
RRF_WEIGHTS = [2.0, 1.0, 1.0, 1.0]

def rrf_query_tokens(r, inject_mate=False):
    """Token string for the /rrf endpoint. Unlike wide_query_tokens, tokens are
    NOT repeated by family weight -- RRF handles per-field emphasis via the
    endpoint's `weights` param, so repetition here would double-count. Returns
    the deduplicated union of the query's dynamic + static tokens (+ injected
    motif tokens for old exports)."""
    toks = set()
    toks.update((r.get("query_text_dynamic", "") or "").split())
    toks.update((r.get("query_text_static", "") or "").split())
    if inject_mate:
        toks.update(pf.mate_picture(r.get("query_fen", ""), r.get("query_solution_uci"),
                                    r.get("query_turn")).get("tokens", []))
        toks.update(pf.king_march(r.get("query_fen", ""), r.get("query_solution_uci"),
                                  r.get("query_turn")).get("tokens", []))
    return " ".join(sorted(toks))


# --- model ------------------------------------------------------------------

def load_model(model_path=None):
    """Load the deployed re-ranker and its feature list.

    Resolution order: an explicit --model path, then retrieval_model.joblib, then
    the most recent retrieval_model_*.joblib in CACHE_DIR (the tagged name the v2
    notebooks save, e.g. retrieval_model_batch2_iter4_pruned_ExtraTrees.joblib),
    matched to its sibling meta JSON. The meta's feature_names drives which subset
    of pair_features is used, so pruned models work unchanged.
    """
    import joblib

    candidates = []
    if model_path:
        candidates = [model_path]
    else:
        default = os.path.join(CACHE_DIR, "retrieval_model.joblib")
        if os.path.exists(default):
            candidates = [default]
        else:
            candidates = sorted(
                glob.glob(os.path.join(CACHE_DIR, "retrieval_model_*.joblib")),
                key=os.path.getmtime, reverse=True)
    if not candidates or not os.path.exists(candidates[0]):
        return None, pf.FEATURE_NAMES, None

    mp = candidates[0]
    model = joblib.load(mp)

    base = os.path.basename(mp)
    if base == "retrieval_model.joblib":
        meta_p = os.path.join(CACHE_DIR, "retrieval_model_meta.json")
    else:                                   # retrieval_model_<tag>.joblib -> ..._meta_<tag>.json
        tag = base[len("retrieval_model_"):-len(".joblib")]
        meta_p = os.path.join(CACHE_DIR, f"retrieval_model_meta_{tag}.json")

    feats = pf.FEATURE_NAMES
    if os.path.exists(meta_p):
        feats = json.load(open(meta_p)).get("feature_names", feats)

    missing = [f for f in feats if f not in set(pf.FEATURE_NAMES)]
    if missing:
        print(f"WARNING: model expects {len(missing)} feature(s) pair_features does NOT "
              f"produce -> they will be 0.0: {missing[:8]}")
    return model, feats, mp



# --- model explanation ------------------------------------------------------

def _linear_parts(model):
    """Return (scaler, linear_clf) if the model is a StandardScaler+linear pipeline,
    drilling through a CalibratedClassifierCV wrapper if present. Note: for a
    calibrated model these coefficients describe the BASE estimator's decision
    boundary, not the calibrated probability itself -- the sigmoid/isotonic
    stage that maps base-score -> calibrated probability has no per-feature
    coefficients, so the explain breakdown is necessarily approximate post-
    calibration (it explains direction/ranking of the contribution, not the
    exact probability shift)."""
    base = model
    if hasattr(model, "calibrated_classifiers_"):
        # use the first fold's base estimator as a representative explainer
        cc0 = model.calibrated_classifiers_[0]
        base = getattr(cc0, "estimator", None) or getattr(cc0, "base_estimator", None)
        if base is None:
            return None, None
    scaler = clf = None
    for v in getattr(base, "named_steps", {}).values():
        if hasattr(v, "mean_") and hasattr(v, "scale_"):
            scaler = v
        if hasattr(v, "coef_"):
            clf = v
    if clf is None and hasattr(base, "coef_"):
        clf = base
    return scaler, clf


# SHAP TreeExplainer support: gives per-pair SIGNED contributions for tree
# ensembles (RandomForest, ExtraTrees, GradientBoosting, HistGB, XGBoost,
# LightGBM), the same kind of "why did THIS pair score this way" breakdown
# the linear path gives, rather than global feature_importances_. NOT every
# tree-ish model is supported -- AdaBoost in particular is NOT supported by
# TreeExplainer (its boundary isn't expressed as the kind of additive tree
# structure SHAP's fast exact algorithm needs); for those, this falls back to
# the slow model-agnostic shap.Explainer, which is too slow for interactive
# --explain (~5-10s/row scales to many MINUTES for a pool_k=200 explain) --
# so AdaBoost-family models fall back further to global feature_importances_
# (the pre-existing "tree" path) rather than hang on a real query.
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

_TREE_EXPLAINER_CACHE = {}

def _tree_parts(model):
    """Return a fitted base tree-ensemble model (drilling through calibration)
    if TreeExplainer can handle it, else None."""
    base = model
    if hasattr(model, "calibrated_classifiers_"):
        cc0 = model.calibrated_classifiers_[0]
        base = getattr(cc0, "estimator", None) or getattr(cc0, "base_estimator", None)
        if base is None:
            return None
    if not hasattr(base, "feature_importances_"):
        return None
    return base


def _get_tree_explainer(model):
    """Cached-by-id TreeExplainer for the base tree model, or None if SHAP
    is unavailable or the model type isn't one TreeExplainer supports."""
    if not _HAS_SHAP:
        return None, None
    base = _tree_parts(model)
    if base is None:
        return None, None
    key = id(model)
    if key in _TREE_EXPLAINER_CACHE:
        return _TREE_EXPLAINER_CACHE[key], base
    try:
        explainer = shap.TreeExplainer(base)
    except Exception:
        explainer = None   # e.g. AdaBoost: structurally unsupported
    _TREE_EXPLAINER_CACHE[key] = explainer
    return explainer, base


def explain_pair(model, feat_names, fd, topn=6):
    """Per-feature breakdown of why the model scored this pair as it did.
    For a linear model: signed contribution = coef * standardized(value).
    For a SHAP-supported tree ensemble (RandomForest, ExtraTrees, GradBoost,
    HistGB, XGBoost, LightGBM): signed per-pair SHAP contribution -- the same
    "why THIS pair" semantics as the linear path, not a global average.
    For everything else (notably AdaBoost, which TreeExplainer doesn't
    support, and where the slow model-agnostic fallback is impractical for
    interactive use): global feature_importances_ x value, same as before --
    this tells you what the model leans on overall, not why this pair scored
    as it did, so treat 'kind == "tree_global"' results with that caveat."""
    x = np.array([fd.get(k, 0.0) for k in feat_names], dtype=float)

    scaler, clf = _linear_parts(model)
    if clf is not None and hasattr(clf, "coef_"):
        xs = (x - scaler.mean_) / scaler.scale_ if scaler is not None else x
        contrib = clf.coef_[0] * xs
        order = np.argsort(contrib)                       # ascending: most negative first
        sinks = [(feat_names[i], x[i], contrib[i]) for i in order[:topn] if contrib[i] < 0]
        lifts = [(feat_names[i], x[i], contrib[i]) for i in order[::-1][:topn] if contrib[i] > 0]
        return "linear", sinks, lifts

    explainer, base = _get_tree_explainer(model)
    if explainer is not None:
        row = pd.DataFrame([fd], columns=feat_names).fillna(0.0)
        try:
            sv = explainer.shap_values(row)
            sv = np.asarray(sv)
            # shape is (1, n_features) for binary single-output, or
            # (1, n_features, n_classes) for multi-output -- take class 1
            vals = sv[0, :, 1] if sv.ndim == 3 else sv[0]
            order = np.argsort(vals)
            sinks = [(feat_names[i], x[i], vals[i]) for i in order[:topn] if vals[i] < 0]
            lifts = [(feat_names[i], x[i], vals[i]) for i in order[::-1][:topn] if vals[i] > 0]
            return "shap", sinks, lifts
        except Exception:
            pass   # fall through to global importance below

    if hasattr(model, "feature_importances_") or (base is not None and hasattr(base, "feature_importances_")):
        imp_model = base if base is not None else model
        imp = imp_model.feature_importances_
        order = np.argsort(-imp)
        return "tree_global", [(feat_names[i], x[i], imp[i]) for i in order[:topn]], []
    return "none", [], []


def _print_explain(kind, sinks, lifts):
    if kind == "linear":
        print("      sinks: " + ", ".join(f"{n}={v:.2f}({c:+.2f})" for n, v, c in sinks))
        if lifts:
            print("      lifts: " + ", ".join(f"{n}={v:.2f}({c:+.2f})" for n, v, c in lifts))
    elif kind == "shap":
        print("      sinks (SHAP): " + ", ".join(f"{n}={v:.2f}({c:+.2f})" for n, v, c in sinks))
        if lifts:
            print("      lifts (SHAP): " + ", ".join(f"{n}={v:.2f}({c:+.2f})" for n, v, c in lifts))
    elif kind == "tree_global":
        print("      top feats (GLOBAL importance x value, not per-pair -- "
              "model type unsupported by SHAP TreeExplainer):")
        print("      " + ", ".join(f"{n}={v:.2f}[{c:.3f}]" for n, v, c in sinks))


def main(export, pool_k, do_explain=False, model_path=None,
         only_smother=False, inject_mate=False, use_rrf=False):
    exports = export if isinstance(export, list) else [export]
    files = []
    for e in exports:
        files += ([e] if e.endswith(".jsonl")
                  else sorted(glob.glob(os.path.join(e, "*.jsonl"))))
    if not files:
        raise SystemExit(f"No export files at {exports}")

    model, feat_names, model_file = load_model(model_path)
    print(f"Model: {os.path.basename(model_file) if model_file else 'NOT FOUND (recall-only)'}"
          f" ({len(feat_names)} features) | pool_k={pool_k} | Lucene={LUCENE_URL}\n")

    # group rows by query
    queries = defaultdict(list)
    for fp in files:
        for line in open(fp, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                queries[r.get("query_id") or fp].append(r)

    if only_smother:
        def _is_smother(rows):
            r0 = rows[0]
            return pf.mate_picture(r0.get("query_fen", ""), r0.get("query_solution_uci"),
                                   r0.get("query_turn")).get("smother")
        queries = {q: rows for q, rows in queries.items() if _is_smother(rows)}
        print(f"[--only-smother] {len(queries)} smother queries (those without "
              f"labelled similars will still be skipped below)")

    import pandas as pd
    grand = {"sim": 0, "in_pool": 0, "top10": 0}
    gen   = {"sim": 0, "in_pool": 0, "top10": 0}   # genuine (excludes the query's own game)

    for qid, rows in queries.items():
        sims = [r for r in rows if str(r.get("label", "")).lower() == "similar"]
        if not sims:
            continue
        r0 = rows[0]
        try:
            if use_rrf:
                pool = lucene_rrf(rrf_query_tokens(r0, inject_mate),
                                  fields=RRF_FIELDS, weights=RRF_WEIGHTS, topk=pool_k)
            else:
                pool = lucene_search(wide_query_tokens(r0, inject_mate), "text_all", pool_k)
        except Exception as e:
            print(f"[{qid}] Lucene unreachable: {e}")
            continue
        pool_ids = [h.get("id") for h in pool]
        rank_of  = {pid: i + 1 for i, pid in enumerate(pool_ids)}

        # rescore the whole pool with the model (needs each doc's features)
        model_rank = {}
        if model is not None:
            qrec = query_rec_from_export(r0)
            rows_feat, ids = [], []
            for h in pool:
                doc = lucene_doc(h.get("id")) or h
                fd = pf.pair_features(qrec, cand_rec_from_doc(doc))
                rows_feat.append([fd.get(k, 0.0) for k in feat_names]); ids.append(h.get("id"))
            probs = model.predict_proba(pd.DataFrame(rows_feat, columns=feat_names))[:, 1]
            order = sorted(range(len(ids)), key=lambda i: -probs[i])
            model_rank = {ids[idx]: rnk + 1 for rnk, idx in enumerate(order)}
            prob_of   = {ids[i]: float(probs[i]) for i in range(len(ids))}
        else:
            prob_of = {}

        print(f"=== query {qid}  ({len(sims)} known-similar, pool={len(pool_ids)}) ===")
        print(f"{'candidate':40s} {'in_pool':8s} {'bm25':6s} {'mProb':7s} {'mRank':6s} {'top10':6s} {'self':5s}")
        for r in sims:
            cid = r.get("candidate_id")
            is_self = (cid == qid)
            in_pool = cid in rank_of
            bm = rank_of.get(cid, None)
            mp = prob_of.get(cid)
            mr = model_rank.get(cid)
            top10 = (mr is not None and mr <= 10)
            grand["sim"] += 1; grand["in_pool"] += int(in_pool); grand["top10"] += int(top10)
            if not is_self:
                gen["sim"] += 1; gen["in_pool"] += int(in_pool); gen["top10"] += int(top10)
            print(f"{str(cid)[:40]:40s} "
                  f"{('YES' if in_pool else 'no'):8s} "
                  f"{(str(bm) if bm else '-'):6s} "
                  f"{(f'{mp:.3f}' if mp is not None else '-'):7s} "
                  f"{(str(mr) if mr else '-'):6s} "
                  f"{('YES' if top10 else 'no'):6s} "
                  f"{('SELF' if is_self else ''):5s}")
            if do_explain and model is not None and not is_self:
                qrec = query_rec_from_export(r0)
                doc  = lucene_doc(cid) or {}
                fd   = pf.pair_features(qrec, cand_rec_from_doc(doc)) if doc else \
                       pf.pair_features(qrec, {
                           "text_static":  r.get("candidate_text_static", ""),
                           "text_dynamic": r.get("candidate_text_dynamic", ""),
                           "solution_uci": r.get("candidate_solution_uci", []),
                           "mate":         r.get("candidate_mate"),
                           "turn":         r.get("candidate_turn"),
                           "rating":       r.get("candidate_rating_avg") or 0,
                           "fen":          r.get("candidate_fen", ""),
                       })
                _print_explain(*explain_pair(model, feat_names, fd))
        print()

    s = grand["sim"] or 1
    g = gen["sim"] or 1
    print("=" * 60)
    print(f"ALL known-similar (incl. each query's own game): {grand['sim']} | "
          f"in pool: {grand['in_pool']} ({100*grand['in_pool']/s:.0f}%) | "
          f"top-10: {grand['top10']} ({100*grand['top10']/s:.0f}%)")
    print(f"GENUINE (excludes self-match the live app filters out): {gen['sim']} | "
          f"in pool: {gen['in_pool']} ({100*gen['in_pool']/g:.0f}%) | "
          f"top-10: {gen['top10']} ({100*gen['top10']/g:.0f}%)   <-- the production number")
    print("If 'in pool' is high but 'top-10' is low -> ranking problem (model/weights).")
    print("If 'in pool' is low -> recall problem (raise pool_k or wire Lever 1).")


if __name__ == "__main__":
    in_ipykernel = "ipykernel" in sys.modules
    if in_ipykernel:
        main(["./exports"], 100, do_explain=True, model_path=None)
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("--export", nargs="+", default=["./exports"],
                        help="one or more ch_export_*.jsonl files or folders")
        ap.add_argument("--pool_k", type=int, default=100)
        ap.add_argument("--model", default=None,
                        help="path to a specific retrieval_model[_tag].joblib "
                             "(default: retrieval_model.joblib, else newest tagged model)")
        ap.add_argument("--explain", action="store_true",
                        help="print per-feature breakdown for each genuine similar miss/hit")
        ap.add_argument("--only-smother", action="store_true",
                        help="restrict to queries whose mate is a smothered mate")
        ap.add_argument("--inject-mate-tokens", action="store_true",
                        help="add query mate-picture tokens live (simulates the new encoder)")
        ap.add_argument("--rrf", action="store_true",
                        help="retrieve via multi-field Reciprocal Rank Fusion (/rrf endpoint) "
                             "instead of single-field text_all BM25")
        a = ap.parse_args()
        main(a.export, a.pool_k, a.explain, a.model, a.only_smother,
             a.inject_mate_tokens, a.rrf)