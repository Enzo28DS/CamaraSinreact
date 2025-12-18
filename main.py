# main.py (PYTHON 3.9 compatible)
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Header
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional
from datetime import datetime, timedelta
import io
import csv
import os
import cv2
import numpy as np

from db import init_db, fetch_detections, counts_by_label, totals, clear_inventory
from learned_db import init_db as init_learn_db, get_label_counts as learned_counts, get_last_learned, DB_PATH as LEARN_DB_PATH
from vision_service import VisionService

app = FastAPI(title="Inventario Inteligente API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

os.makedirs("captures", exist_ok=True)
app.mount("/captures", StaticFiles(directory="captures"), name="captures")

# ✅ Inicializar ambas DBs al arrancar
init_db()
init_learn_db(LEARN_DB_PATH)

VISION = VisionService(
    yolo_model_path="yolov8m.pt",
    device="cuda",
    imgsz=896,
    yolo_conf=0.22,
    min_sharp=45.0,
    min_recog_conf=0.94,  # <- NO guarda en inventario si < 0.94
)

def parse_since(since: Optional[str], last_minutes: Optional[int]):
    if last_minutes is not None:
        return (datetime.now() - timedelta(minutes=int(last_minutes))).isoformat(timespec="seconds")
    if since:
        return since
    return None

@app.get("/stats")
def stats():
    return totals()

@app.get("/counts")
def counts(limit: int = 50):
    return counts_by_label(limit=limit)

@app.get("/detections")
def detections(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=5, le=200),
    label: Optional[str] = None,
    since: Optional[str] = None,
    last_minutes: Optional[int] = None,
):
    since_iso = parse_since(since, last_minutes)
    raw = fetch_detections(limit=5000, label=label)
    if since_iso:
        raw = [r for r in raw if r.get("ts", "") >= since_iso]

    total = len(raw)
    start = (page - 1) * page_size
    end = start + page_size

    return {"page": page, "page_size": page_size, "total": total, "items": raw[start:end]}

@app.get("/export.csv")
def export_csv(
    label: Optional[str] = None,
    since: Optional[str] = None,
    last_minutes: Optional[int] = None,
):
    since_iso = parse_since(since, last_minutes)
    rows = fetch_detections(limit=8000, label=label)
    if since_iso:
        rows = [r for r in rows if r.get("ts", "") >= since_iso]

    def generate():
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["id","ts","camera_id","model","label","confidence","x1","y1","x2","y2","image_path"])
        yield buf.getvalue()
        buf.seek(0); buf.truncate(0)

        for r in rows:
            w.writerow([r.get(k) for k in ["id","ts","camera_id","model","label","confidence","x1","y1","x2","y2","image_path"]])
            yield buf.getvalue()
            buf.seek(0); buf.truncate(0)

    return StreamingResponse(
        generate(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=inventario.csv"},
    )

INVENTORY_TOKEN = os.environ.get("INVENTORY_TOKEN", "1234")

@app.post("/inventory/clear")
def inventory_clear(x_token: Optional[str] = Header(None)):
    if x_token != INVENTORY_TOKEN:
        raise HTTPException(status_code=401, detail="Token inválido")
    clear_inventory()
    return {"ok": True}

# ✅ NUEVO: ver qué se aprendió DE VERDAD
@app.get("/learned/summary")
def learned_summary(limit: int = 50):
    return {
        "db_path": str(LEARN_DB_PATH),
        "top": learned_counts(limit=limit, db_path=LEARN_DB_PATH),
        "last": get_last_learned(10, db_path=LEARN_DB_PATH)
    }

@app.post("/vision/frame")
async def vision_frame(
    file: UploadFile = File(...),
    ignore_person: Optional[bool] = True,
    save: bool = True,
    force_save: bool = False,
):
    try:
        data = await file.read()
        npimg = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Frame inválido")

        result = VISION.process_frame(
            frame_bgr=frame,
            reticle_size=260,
            ignore_person=bool(ignore_person),
            save_to_inventory=bool(save),
            force_save=bool(force_save),
            camera_id="web",
            model_name="yolo+clip",
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/learn/frame")
async def learn_frame(
    file: UploadFile = File(...),
    label: str = Query(...),
    ignore_person: Optional[bool] = True,
    cooldown_ms: int = Query(400),
):
    try:
        data = await file.read()
        npimg = np.frombuffer(data, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Frame inválido")

        result = VISION.learn_frame(
            frame_bgr=frame,
            label=label,
            reticle_size=260,
            ignore_person=bool(ignore_person),
            cooldown_ms=int(cooldown_ms),
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
