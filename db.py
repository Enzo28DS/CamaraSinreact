# db.py
import sqlite3
import csv
from typing import Optional
from pathlib import Path
from datetime import datetime, timedelta

# ✅ FIX: la DB vive SIEMPRE junto a este archivo db.py
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "inventario.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    model TEXT NOT NULL,
    label TEXT NOT NULL,
    confidence REAL NOT NULL,
    x1 INTEGER NOT NULL,
    y1 INTEGER NOT NULL,
    x2 INTEGER NOT NULL,
    y2 INTEGER NOT NULL,
    image_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_detections_ts ON detections(ts);
CREATE INDEX IF NOT EXISTS idx_detections_label ON detections(label);
"""

def init_db(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()

def insert_detection(
    ts: datetime,
    camera_id: str,
    model: str,
    label: str,
    confidence: float,
    box_xyxy: tuple[int, int, int, int],
    image_path: Optional[str] = None,
    db_path: Path = DB_PATH
) -> None:
    x1, y1, x2, y2 = box_xyxy
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO detections (ts, camera_id, model, label, confidence, x1, y1, x2, y2, image_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ts.isoformat(timespec="seconds"), camera_id, model, label, float(confidence),
             int(x1), int(y1), int(x2), int(y2), image_path)
        )
        conn.commit()
    finally:
        conn.close()

# ---------------- Lecturas simples ----------------
def fetch_detections(limit: int = 200, label: Optional[str] = None, db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        if label:
            rows = conn.execute(
                """
                SELECT *
                FROM detections
                WHERE label LIKE ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (f"%{label}%", int(limit))
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT *
                FROM detections
                ORDER BY id DESC
                LIMIT ?
                """,
                (int(limit),)
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()

def counts_by_label(limit: int = 50, db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT label, COUNT(*) as count
            FROM detections
            GROUP BY label
            ORDER BY count DESC
            LIMIT ?
            """,
            (int(limit),)
        ).fetchall()
        return [{"label": r[0], "count": int(r[1])} for r in rows]
    finally:
        conn.close()

def totals(db_path: Path = DB_PATH):
    conn = sqlite3.connect(db_path)
    try:
        total = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        cats = conn.execute("SELECT COUNT(DISTINCT label) FROM detections").fetchone()[0]
        return {"total_events": int(total), "total_labels": int(cats)}
    finally:
        conn.close()

# ---------------- NUEVO: paginación + filtros ----------------
def fetch_detections_paged(
    page: int = 1,
    page_size: int = 25,
    label: Optional[str] = None,
    since_minutes: Optional[int] = None,
    db_path: Path = DB_PATH
):
    page = max(1, int(page))
    page_size = max(1, min(200, int(page_size)))
    offset = (page - 1) * page_size

    conds = []
    params = []

    if label:
        conds.append("label LIKE ?")
        params.append(f"%{label}%")

    if since_minutes is not None:
        since_dt = datetime.now() - timedelta(minutes=int(since_minutes))
        conds.append("ts >= ?")
        params.append(since_dt.isoformat(timespec="seconds"))

    where = (" WHERE " + " AND ".join(conds)) if conds else ""

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        total_count = conn.execute(
            f"SELECT COUNT(*) FROM detections{where}",
            params
        ).fetchone()[0]

        rows = conn.execute(
            f"""
            SELECT *
            FROM detections
            {where}
            ORDER BY id DESC
            LIMIT ? OFFSET ?
            """,
            (*params, page_size, offset)
        ).fetchall()

        return {
            "page": page,
            "page_size": page_size,
            "total": int(total_count),
            "items": [dict(r) for r in rows],
        }
    finally:
        conn.close()

# ---------------- NUEVO: vaciar inventario ----------------
def clear_inventory(db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM detections")
        conn.commit()
    finally:
        conn.close()

# ---------------- NUEVO: exportar CSV ----------------
def export_inventory_csv(out_path: Path = BASE_DIR / "inventario_export.csv", db_path: Path = DB_PATH) -> Path:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute("SELECT * FROM detections ORDER BY id DESC").fetchall()
        cols = rows[0].keys() if rows else ["id","ts","camera_id","model","label","confidence","x1","y1","x2","y2","image_path"]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(cols)
            for r in rows:
                w.writerow([r[c] for c in cols])
        return out_path
    finally:
        conn.close()

# ✅ opcional: para verificar dónde está la DB de verdad
def db_info():
    return {"db_path": str(DB_PATH.resolve())}
