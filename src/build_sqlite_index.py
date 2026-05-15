"""
build_sqlite_index.py
---------------------
One-time script: converts corpus_mates.jsonl into a SQLite database
with FTS5 full-text search (BM25 ranking built in).

Also stores all corpus fields as a regular table so app.py can
retrieve any doc by id without loading the full corpus into RAM.

Run once:
    python build_sqlite_index.py

Output:
    data/derived/corpus_mates.db   (~2-4 GB, replaces in-memory corpus+BM25)

After this, app.py starts in ~1 second instead of minutes.
"""

from pathlib import Path
import json
import sqlite3
import time

ROOT        = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "derived" / "corpus_checkmates5.jsonl"
DB_PATH     = ROOT / "data" / "derived" / "corpus_checkmates5.db"

CHUNK_SIZE  = 10_000   # rows per transaction


def main():
    print(f"Input : {CORPUS_PATH}")
    print(f"Output: {DB_PATH}")

    if DB_PATH.exists():
        print(f"\nDB already exists. Delete it first to rebuild:\n  {DB_PATH}")
        return

    con = sqlite3.connect(str(DB_PATH))
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA cache_size=-512000")   # 512 MB page cache during build

    # ── Main document store ────────────────────────────────────────────────
    # Stores all fields needed to serve hits and re-ranking
    con.execute("""
        CREATE TABLE docs (
            id              TEXT PRIMARY KEY,
            site            TEXT,
            ply             INTEGER,
            mate_before     INTEGER,
            board_fen       TEXT,
            turn            TEXT,
            pockets_white   TEXT,   -- JSON
            pockets_black   TEXT,   -- JSON
            text_dynamic    TEXT,
            text_static     TEXT,
            text_all        TEXT,
            text_dynamic_general  TEXT,
            text_dynamic_solution TEXT
        )
    """)

    # ── FTS5 index for BM25 search ─────────────────────────────────────────
    # text_all   → main search field (already weighted in build_corpus)
    # text_static → fallback for static-only search
    # content=''  means FTS stores its own copy of the text
    con.execute("""
        CREATE VIRTUAL TABLE fts_all USING fts5(
            id UNINDEXED,
            text_all,
            content='docs',
            content_rowid='rowid'
        )
    """)

    con.execute("""
        CREATE VIRTUAL TABLE fts_static USING fts5(
            id UNINDEXED,
            text_static,
            content='docs',
            content_rowid='rowid'
        )
    """)

    # ── Board FEN index for exact-match search ─────────────────────────────
    con.execute("CREATE INDEX idx_board_fen ON docs(board_fen)")

    print("\nBuilding index...")
    t0 = time.time()

    n_total = 0
    buf = []

    def flush(buf):
        con.executemany("""
            INSERT OR IGNORE INTO docs
              (id, site, ply, mate_before, board_fen, turn,
               pockets_white, pockets_black,
               text_dynamic, text_static, text_all,
               text_dynamic_general, text_dynamic_solution)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, buf)
        # Populate FTS from the docs table (content= mode)
        con.execute("INSERT INTO fts_all(fts_all) VALUES('rebuild')")
        con.execute("INSERT INTO fts_static(fts_static) VALUES('rebuild')")
        con.commit()

    with CORPUS_PATH.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            buf.append((
                rec.get("id", ""),
                rec.get("site", ""),
                rec.get("ply", 0),
                rec.get("mate_before"),
                rec.get("board_fen", ""),
                rec.get("turn", "white"),
                json.dumps(rec.get("pockets_white", {})),
                json.dumps(rec.get("pockets_black", {})),
                rec.get("text_dynamic", ""),
                rec.get("text_static", ""),
                rec.get("text_all", ""),
                rec.get("text_dynamic_general", ""),
                rec.get("text_dynamic_solution", ""),
            ))
            n_total += 1

            if len(buf) >= CHUNK_SIZE:
                flush(buf)
                buf = []
                elapsed = time.time() - t0
                print(f"  {n_total:,} rows  ({elapsed:.0f}s)")

    if buf:
        flush(buf)

    # Final FTS rebuild to ensure consistency
    con.execute("INSERT INTO fts_all(fts_all) VALUES('rebuild')")
    con.execute("INSERT INTO fts_static(fts_static) VALUES('rebuild')")
    con.execute("PRAGMA optimize")
    con.commit()
    con.close()

    elapsed = time.time() - t0
    size_mb = DB_PATH.stat().st_size / 1024 / 1024
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Rows    : {n_total:,}")
    print(f"  DB size : {size_mb:.0f} MB")
    print(f"  Path    : {DB_PATH}")
    print(f"\nNow restart app.py — it will use the SQLite index with ~0 startup time.")


if __name__ == "__main__":
    main()
