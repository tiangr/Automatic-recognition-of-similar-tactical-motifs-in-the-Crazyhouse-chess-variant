from pathlib import Path
import json
from encode import encode_static, encode_pockets, encode_dynamic
from encodev2 import encode_dynamic_v2



ROOT = Path(__file__).resolve().parents[1]
#before filtering
#INP = ROOT / "data" / "derived" / "tactics_1k_aug.jsonl"
#OUT = ROOT / "data" / "derived" / "corpus_1k.jsonl"

#after filtering
INP = ROOT / "data" / "derived" / "tactics_1k_tactical.jsonl"
OUT = ROOT / "data" / "derived" / "corpus_tactical.jsonl"


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with INP.open("r", encoding="utf-8", errors="replace") as f, \
         OUT.open("w", encoding="utf-8") as out:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            board_fen = rec.get("board_fen")
            pw = rec.get("pockets_white", {})
            pb = rec.get("pockets_black", {})
            if not board_fen:
                continue

            static = encode_static(board_fen)
            pocket = encode_pockets(pw, pb)
            #encode
            #dynamic = encode_dynamic(rec)

            #encodev2
            dynamic = encode_dynamic_v2(rec)

            rec_out = {
                "id": rec.get("event_id") or (rec.get("site") + "_" + str(rec.get("ply"))),
                "site": rec.get("site"),
                "ply": rec.get("ply"),
                "board_fen": board_fen,
                "pockets_white": pw,
                "pockets_black": pb,
                "text_static": " ".join(static + pocket),
                "text_dynamic": " ".join(dynamic),
                "text_all": " ".join(dynamic + dynamic + dynamic + pocket + pocket + static),

            }
            out.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            n += 1

    print("DONE corpus docs:", n, "->", OUT)

if __name__ == "__main__":
    main()
