from pathlib import Path
import json
import random
import chess.variant
import chess.pgn
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[1]
#za synthetic data
#CORPUS = ROOT / "data" / "derived" / "corpus_tactical_with_queryply.jsonl"
#za normal data
CORPUS = ROOT / "data" / "derived" / "corpus_tactical.jsonl"
AUG = ROOT / "data" / "derived" / "tactics_1k_aug.jsonl"
PGN_DB = ROOT / "data" / "derived" / "sample_500_games.pgn"

FIELD = "text_all"
TOPK = 10

# Mentor pack size
N_MATE2 = 5
N_MATE3 = 5
SEED = 42  # reproducible selection

#bizjak
N_QUERIES = 10

def base_game_id(doc_id: str) -> str:
    return doc_id.rsplit("_", 1)[0]


def ply_of(doc_id: str) -> int:
    return int(doc_id.rsplit("_", 1)[1])


def is_syn(doc_id: str) -> bool:
    return "_SYN" in doc_id


def syn_source_site(doc_id: str) -> str:
    return doc_id.split("_SYN", 1)[0]


def load_aug_index():
    idx = {}
    with AUG.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            _id = r.get("event_id") or (r.get("site") + "_" + str(r.get("ply")))
            idx[_id] = r
    return idx


def load_docs():
    docs, texts = [], []
    with CORPUS.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            docs.append(rec)
            texts.append(rec[FIELD].split())
    return docs, texts


def load_game_uci_by_site(site_prefix: str) -> list[str] | None:
    with PGN_DB.open("r", encoding="utf-8", errors="replace") as f:
        while True:
            g = chess.pgn.read_game(f)
            if g is None:
                return None
            site = g.headers.get("Site", "")
            if site.startswith(site_prefix):
                board = chess.variant.CrazyhouseBoard()
                uci = []
                for mv in g.mainline_moves():
                    uci.append(mv.uci())
                    board.push(mv)
                return uci


def to_pgn_from_uci(uci_moves: list[str], ply: int, label: str, site: str, comment: str) -> chess.pgn.Game:
    board = chess.variant.CrazyhouseBoard()
    game = chess.pgn.Game()
    game.headers["Event"] = label
    game.headers["Site"] = site
    game.headers["Variant"] = "Crazyhouse"

    node = game
    for u in uci_moves[: max(0, ply - 1)]:
        mv = board.parse_uci(u)
        node = node.add_variation(mv)
        board.push(mv)

    node.comment = comment
    return game

#prirejeno po bizjaku
def to_pgn_game_from_aug(aug_rec: dict, label: str) -> chess.pgn.Game:
    board = chess.variant.CrazyhouseBoard()

    prefix = aug_rec.get("uci_moves_prefix")
    uci_moves = aug_rec.get("uci_moves", [])
    ply = int(aug_rec["ply"])

    if isinstance(prefix, list):
        moves_to_play = prefix
    else:
        moves_to_play = uci_moves[:max(0, ply - 1)]

    game = chess.pgn.Game()
    game.headers["Event"] = label
    game.headers["Site"] = aug_rec.get("site") or "?"
    game.headers["Variant"] = "Crazyhouse"

    node = game
    for u in moves_to_play:
        mv = board.parse_uci(u)
        node = node.add_variation(mv)
        board.push(mv)

    mate = aug_rec.get("mate_before")
    cp = aug_rec.get("cp_before")
    delta = aug_rec.get("delta")
    best = aug_rec.get("bestmove_before")
    played = aug_rec.get("played_move")
    pv = aug_rec.get("pv_before") or []

    node.comment = (
        f"delta={delta} cp_before={cp} mate_before={mate} "
        f"played={played} best={best} pv={' '.join(pv[:10])}"
    )
    return game
'''def to_pgn_game_from_aug(aug_rec: dict, label: str) -> chess.pgn.Game:
    board = chess.variant.CrazyhouseBoard()
    uci_moves = aug_rec["uci_moves"]
    ply = int(aug_rec["ply"])

    game = chess.pgn.Game()
    game.headers["Event"] = label
    game.headers["Site"] = aug_rec.get("site") or "?"
    game.headers["Variant"] = "Crazyhouse"

    node = game
    for u in uci_moves[: max(0, ply - 1)]:
        mv = board.parse_uci(u)
        node = node.add_variation(mv)
        board.push(mv)

    mate = aug_rec.get("mate_before")
    best = aug_rec.get("bestmove_before")
    pv = aug_rec.get("pv_before") or []
    node.comment = f"mate={mate} best={best} pv={' '.join(pv[:10])}"
    return game'''


'''def pick_queries(aug: dict) -> list[str]:
    # pick from AUG to ensure we can export full PGN lines easily
    mate2 = [k for k, r in aug.items() if r.get("mate_before") == 2]
    mate3 = [k for k, r in aug.items() if r.get("mate_before") == 3]

    random.seed(SEED)
    random.shuffle(mate2)
    random.shuffle(mate3)

    picked = mate2[:N_MATE2] + mate3[:N_MATE3]
    return picked'''

def pick_queries(aug: dict, corpus_doc_ids: set[str]) -> list[str]:
    """
    Bizjak-style selection:
    pick strong tactical events by largest delta, but only if present in corpus.
    """
    rows = []
    for doc_id, rec in aug.items():
        if doc_id not in corpus_doc_ids:
            continue
        delta = rec.get("delta")
        if delta is None:
            continue
        try:
            rows.append((float(delta), doc_id))
        except Exception:
            continue

    rows.sort(reverse=True)

    # take top band, then sample reproducibly if there are many
    top_band = rows[: min(len(rows), 100)]
    random.seed(SEED)
    random.shuffle(top_band)

    picked = [doc_id for _, doc_id in top_band[:N_QUERIES]]
    return picked

#bizjak
def main():
    aug = load_aug_index()
    docs, tokenized = load_docs()
    bm25 = BM25Okapi(tokenized)

    # Map doc_id -> corpus index for fast lookup
    doc_index = {d["id"]: i for i, d in enumerate(docs)}

    query_ids = pick_queries(aug, set(doc_index.keys()))
    if not query_ids:
        raise SystemExit("No suitable queries found that exist in both AUG and CORPUS.")

    out_path = ROOT / "data" / "derived" / f"mentor_pack_{FIELD}_top{TOPK}.pgn"

    with out_path.open("w", encoding="utf-8") as out:
        for qn, q_id in enumerate(query_ids, start=1):
            qi = doc_index[q_id]
            query = docs[qi]
            q_base = base_game_id(q_id)

            scores = bm25.get_scores(query[FIELD].split())
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

            # Write query
            if q_id in aug:
                out.write(str(to_pgn_game_from_aug(aug[q_id], f"QUERY {qn} {q_id}")) + "\n\n")
            else:
                continue

            shown = 0
            for i in ranked:
                cand_id = docs[i]["id"]

                if cand_id == q_id:
                    continue
                if base_game_id(cand_id) == q_base:
                    continue

                label = f'HIT {qn}.{shown + 1} score={scores[i]:.2f} {cand_id}'

                if cand_id in aug:
                    out.write(str(to_pgn_game_from_aug(aug[cand_id], label)) + "\n\n")
                else:
                    # Allow SYN if corpus was changed manually later
                    if is_syn(cand_id):
                        src_site = syn_source_site(cand_id)
                        ply = ply_of(cand_id)
                        uci_full = load_game_uci_by_site(src_site)
                        if uci_full is None:
                            out.write(f'[Event "{label}"]\n')
                            out.write(f'[Site "{src_site}"]\n')
                            out.write('[Variant "Crazyhouse"]\n\n')
                            out.write(f'{{ SYN but source PGN not found in {PGN_DB} }} *\n\n')
                        else:
                            comment = f"SYN from {src_site} ply={ply}"
                            out.write(str(to_pgn_from_uci(uci_full, ply, label, src_site, comment)) + "\n\n")
                    else:
                        continue

                shown += 1
                if shown >= TOPK:
                    break

            out.write("\n\n")

    print("WROTE:", out_path)
    print("Queries used:")
    for q_id in query_ids:
        print(" ", q_id)


if __name__ == "__main__":
    main()
'''def main():
    aug = load_aug_index()
    docs, tokenized = load_docs()
    bm25 = BM25Okapi(tokenized)

    # Map doc_id -> corpus index for fast lookup
    doc_index = {d["id"]: i for i, d in enumerate(docs)}

    query_ids = pick_queries(aug)
    if not query_ids:
        raise SystemExit("No mate_before=2/3 queries found in AUG.")

    out_path = ROOT / "data" / "derived" / f"mentor_pack_{FIELD}_top{TOPK}.pgn"
    with out_path.open("w", encoding="utf-8") as out:
        for qn, q_id in enumerate(query_ids, start=1):
            if q_id not in doc_index:
                # if it happens, skip (corpus/AUG mismatch)
                continue

            qi = doc_index[q_id]
            query = docs[qi]
            q_base = base_game_id(q_id)

            scores = bm25.get_scores(query[FIELD].split())
            ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

            # Write query
            out.write(str(to_pgn_game_from_aug(aug[q_id], f"QUERY {qn} {q_id}")) + "\n\n")

            shown = 0
            for i in ranked:
                cand_id = docs[i]["id"]
                if cand_id == q_id:
                    continue
                if base_game_id(cand_id) == q_base:
                    continue

                label = f'HIT {qn}.{shown+1} score={scores[i]:.2f} {cand_id}'

                if cand_id in aug:
                    out.write(str(to_pgn_game_from_aug(aug[cand_id], label)) + "\n\n")
                else:
                    # allow SYN if present
                    if is_syn(cand_id):
                        src_site = syn_source_site(cand_id)
                        ply = ply_of(cand_id)
                        uci_full = load_game_uci_by_site(src_site)
                        if uci_full is None:
                            out.write(f'[Event "{label}"]\n[Site "{src_site}"]\n[Variant "Crazyhouse"]\n\n')
                            out.write(f'{{ SYN but source PGN not found in {PGN_DB} }} *\n\n')
                        else:
                            comment = f"SYN from {src_site} ply={ply}"
                            out.write(str(to_pgn_from_uci(uci_full, ply, label, src_site, comment)) + "\n\n")
                    else:
                        # skip unknowns
                        continue

                shown += 1
                if shown >= TOPK:
                    break

            # Separator comment between queries
            out.write("\n\n")

    print("WROTE:", out_path)


if __name__ == "__main__":
    main()'''