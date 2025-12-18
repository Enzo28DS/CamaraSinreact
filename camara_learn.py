import cv2
import numpy as np
import time
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from typing import Optional, Tuple, List, Dict, Any

from learned_db import init_db as init_learn_db, save_object, get_last_learned, get_label_counts
from recognizer import Recognizer
from db import init_db as init_inventory_db, insert_detection

PERSON_CLASS_ID = 0

# ---------------- Utilidades geométricas ----------------
def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)

def blend_box(old, new, alpha=0.65):
    ox1, oy1, ox2, oy2 = old
    nx1, ny1, nx2, ny2 = new
    return (
        int(alpha*ox1 + (1-alpha)*nx1),
        int(alpha*oy1 + (1-alpha)*ny1),
        int(alpha*ox2 + (1-alpha)*nx2),
        int(alpha*oy2 + (1-alpha)*ny2),
    )

def center_distance(box, frame_shape):
    h, w = frame_shape[:2]
    cx, cy = w // 2, h // 2
    x1, y1, x2, y2 = box
    bx, by = (x1 + x2) // 2, (y1 + y2) // 2
    return abs(cx - bx) + abs(cy - by)

def crop_safe(frame, box):
    x1, y1, x2, y2 = box
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]

def laplacian_sharpness(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def box_area(box) -> int:
    x1,y1,x2,y2 = box
    return max(0, x2-x1) * max(0, y2-y1)

def box_aspect(box) -> float:
    x1,y1,x2,y2 = box
    w = max(1, x2-x1)
    h = max(1, y2-y1)
    return float(w) / float(h)

def rect_area(rect) -> int:
    x1,y1,x2,y2 = rect
    return max(0, x2-x1) * max(0, y2-y1)

def center_in_rect(box, rect) -> bool:
    x1,y1,x2,y2 = box
    rx1,ry1,rx2,ry2 = rect
    cx = (x1+x2)//2
    cy = (y1+y2)//2
    return (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)

# ---------------- UI helpers ----------------
def draw_lines(frame, lines, x=10, y=25, color=(255,255,255), scale=0.58):
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
        y += 22

def get_reticle_rect(frame_shape, size=220) -> Tuple[int,int,int,int]:
    h, w = frame_shape[:2]
    cx, cy = w // 2, h // 2
    half = size // 2
    return (cx - half, cy - half, cx + half, cy + half)

def draw_reticle(frame, rect, color=(200, 200, 200)):
    x1, y1, x2, y2 = rect
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx-12, cy), (cx+12, cy), color, 2)
    cv2.line(frame, (cx, cy-12), (cx, cy+12), color, 2)

def draw_traffic_light(frame, state: str, message: str, sharp: float, cd: int):
    x, y = 10, 10
    radius = 10
    spacing = 26
    colors = {
        "red":   (0, 0, 255),
        "yellow":(0, 255, 255),
        "green": (0, 255, 0),
        "off":   (60, 60, 60),
    }
    order = ["red", "yellow", "green"]
    for i, s in enumerate(order):
        c = colors[s] if s == state else colors["off"]
        cv2.circle(frame, (x + radius + 2, y + radius + i*spacing + 2), radius, c, -1)

    cv2.putText(frame, message, (x + 40, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    cv2.putText(frame, f"sharp={sharp:.0f}  centerDist={cd}",
                (x + 40, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

# ---------------- GUIA “tipo FaceID” ----------------
def box_center_offset_to_reticle(box, reticle):
    rx1, ry1, rx2, ry2 = reticle
    rcx, rcy = (rx1+rx2)//2, (ry1+ry2)//2
    x1, y1, x2, y2 = box
    bcx, bcy = (x1+x2)//2, (y1+y2)//2
    return (bcx - rcx, bcy - rcy)

def stage_target_list() -> List[Tuple[str,int]]:
    return [
        ("center", 2),
        ("left", 2),
        ("right", 2),
        ("up", 2),
        ("down", 2),
        ("near", 3),
        ("far", 3),
        ("tilt", 4),
    ]

def stage_instruction(stage: str) -> str:
    return {
        "center": "Coloca el objeto CENTRADO en la retícula",
        "left":   "Mueve el objeto a la IZQUIERDA dentro de la retícula",
        "right":  "Mueve el objeto a la DERECHA dentro de la retícula",
        "up":     "Mueve el objeto ARRIBA dentro de la retícula",
        "down":   "Mueve el objeto ABAJO dentro de la retícula",
        "near":   "ACERCA el objeto (más grande en pantalla)",
        "far":    "ALEJA el objeto (más pequeño, pero visible)",
        "tilt":   "INCLINA/GIRA ligeramente el objeto (cambia el ángulo)",
    }.get(stage, stage)

def stage_satisfied(stage: str, box, reticle, area_now: int, area_ref: int, aspect_now: float, aspect_ref: float) -> bool:
    dx, dy = box_center_offset_to_reticle(box, reticle)

    if stage == "center":
        return abs(dx) < 35 and abs(dy) < 35
    if stage == "left":
        return dx < -45 and abs(dy) < 70
    if stage == "right":
        return dx > 45 and abs(dy) < 70
    if stage == "up":
        return dy < -45 and abs(dx) < 70
    if stage == "down":
        return dy > 45 and abs(dx) < 70

    # ✅ NEAR / FAR robustos
    if stage == "near":
        return area_ref > 0 and area_now >= int(area_ref * 1.25)

    if stage == "far":
        if area_ref <= 0:
            return False
        r_area = rect_area(reticle)
        # 1) bajar vs referencia
        # 2) ser pequeño vs retícula (estabiliza)
        return (area_now <= int(area_ref * 0.85)) and (area_now <= int(r_area * 0.35))

    if stage == "tilt":
        return abs(aspect_now - aspect_ref) >= 0.18

    return False

# ---------------- Reconocimiento robusto ----------------
def _normalize_recognizer_output(out: Any):
    if out is None:
        return None, 0.0, False

    if isinstance(out, (tuple, list)) and len(out) >= 2:
        label = out[0]
        score = float(out[1]) if out[1] is not None else 0.0
        amb = bool(out[2]) if len(out) >= 3 else False
        return label, score, amb

    if isinstance(out, dict):
        label = out.get("label") or out.get("name")
        score = out.get("score") or out.get("confidence") or 0.0
        amb = out.get("ambiguous") or False
        return label, float(score), bool(amb)

    if isinstance(out, str):
        return out, 1.0, False

    return None, 0.0, False

def recognize_crop(rec: Recognizer, crop_bgr: np.ndarray):
    for fn in ["predict_bgr", "classify_bgr", "predict"]:
        if hasattr(rec, fn):
            try:
                return getattr(rec, fn)(crop_bgr)
            except Exception:
                pass

    try:
        emb = rec.embed_bgr(crop_bgr)
    except Exception:
        emb = None

    if emb is not None:
        for fn in ["predict_embedding", "predict_emb", "predict_vec", "classify_embedding", "match_embedding"]:
            if hasattr(rec, fn):
                try:
                    return getattr(rec, fn)(emb)
                except Exception:
                    pass

    return None

# ---------------- Inventario: guardar evento ----------------
def save_inventory_event(
    label: str,
    confidence: float,
    box_xyxy: Tuple[int,int,int,int],
    crop_bgr: np.ndarray,
    captures_dir: Path,
    camera_id: str,
    model_name: str,
):
    ts_dt = datetime.now()
    ts_iso = ts_dt.isoformat(timespec="seconds")
    safe_ts = ts_iso.replace(":", "-").replace(".", "-")

    img_name = f"{safe_ts}_{label}.jpg"
    img_abs = captures_dir / img_name
    img_rel = f"captures/{img_name}"

    cv2.imwrite(str(img_abs), crop_bgr)

    x1, y1, x2, y2 = box_xyxy
    insert_detection(
        ts=ts_dt,
        camera_id=camera_id,
        model=model_name,
        label=label,
        confidence=float(confidence),
        box_xyxy=(int(x1), int(y1), int(x2), int(y2)),
        image_path=img_rel
    )

# ---------------- Main ----------------
def main():
    init_learn_db()
    init_inventory_db()

    # Rendimiento
    yolo_model = "yolov8m.pt"   # menos GPU: yolov8s.pt
    imgsz = 640
    conf = 0.45
    detect_every = 2

    # ✅ Retícula robusta: centro en retícula (mejor en objetos pequeños)
    reticle_size = 220
    require_center_in_reticle = True
    min_reticle_iou = 0.20  # si cambias require_center_in_reticle=False, usa este

    # ✅ Área mínima dinámica
    min_area_normal = 3000
    min_area_far = 900

    # ✅ Tracking TTL (si YOLO pierde unos frames, mantenemos bbox)
    track_ttl_frames = 10
    lost_counter = 0

    # Calidad guardado inventario
    min_sharpness_save = 70.0
    min_recognition_conf = 0.35

    # Cooldown inventario
    inventory_cooldown_sec = 3.0
    last_saved_by_label: Dict[str, float] = {}

    # Scan calidad
    scan_min_sharp = 60.0
    scan_min_distinct = 0.04
    scan_cooldown_frames = 3

    model = YOLO(yolo_model)
    try:
        model.fuse()
    except Exception:
        pass

    rec = Recognizer(device="cuda", threshold=0.32, ambiguous_margin=0.03, use_knn_fallback=True, knn_topk=5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No pude abrir la cámara 0")

    window = "Scan Inventario"
    ignore_person = True

    CAPTURES_DIR = Path("captures")
    CAPTURES_DIR.mkdir(exist_ok=True)

    # UI input
    typing_mode = False
    typed_text = ""
    status = ""

    # bbox cache
    frame_idx = 0
    last_box: Optional[Tuple[int,int,int,int]] = None
    last_crop = None

    # overlays DB
    last5 = get_last_learned(5)
    counts = get_label_counts(15)

    # scan state
    scan_mode = False
    scan_label = ""
    scan_targets = stage_target_list()
    stage_idx = 0
    stage_done = 0
    scan_total_done = 0
    scan_last_saved_frame = -9999
    scan_last_emb = None

    # referencias para near/far/tilt
    area_ref = 0
    aspect_ref = 0.0

    live_label = None
    live_score = 0.0

    print("✅ Listo. Teclas: Q salir | P personas | S escanear guiado (FaceID)")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            reticle = get_reticle_rect(frame.shape, size=reticle_size)

            # ---------- YOLO ----------
            new_box = None
            if frame_idx % detect_every == 0 or last_box is None:
                res = model.predict(frame, conf=conf, imgsz=imgsz, device="cuda", verbose=False)[0]
                boxes = res.boxes

                # min_area dinámico según etapa
                dynamic_min_area = min_area_normal
                if scan_mode and scan_label and stage_idx < len(scan_targets):
                    stage_name, _ = scan_targets[stage_idx]
                    if stage_name == "far":
                        dynamic_min_area = min_area_far

                candidates = []
                if boxes is not None and len(boxes) > 0:
                    for b in boxes:
                        cls_id = int(b.cls[0])
                        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                        area = (x2 - x1) * (y2 - y1)

                        if area < dynamic_min_area:
                            continue
                        if ignore_person and cls_id == PERSON_CLASS_ID:
                            continue

                        candidates.append(((x1,y1,x2,y2), area))

                if candidates:
                    candidates.sort(key=lambda it: center_distance(it[0], frame.shape))
                    top3 = candidates[:3]
                    top3.sort(key=lambda it: it[1], reverse=True)
                    new_box = top3[0][0]

            # actualizar bbox con suavizado + TTL tracking
            if new_box is not None:
                if last_box is None:
                    last_box = new_box
                else:
                    if iou(last_box, new_box) > 0.15:
                        last_box = blend_box(last_box, new_box, alpha=0.65)
                    else:
                        last_box = new_box
                lost_counter = 0
                last_crop = crop_safe(frame, last_box)
            else:
                # sin detección nueva: aguantar last_box por unos frames
                if last_box is not None and lost_counter < track_ttl_frames:
                    lost_counter += 1
                    last_crop = crop_safe(frame, last_box)
                else:
                    last_box = None
                    last_crop = None
                    lost_counter = 0

            # ---------- UI base ----------
            draw_reticle(frame, reticle, color=(200, 200, 200))

            has_obj = (last_box is not None and last_crop is not None and last_crop.size > 0)

            # ✅ SOLO dentro del retículo: criterio robusto para objetos pequeños
            if has_obj:
                if require_center_in_reticle:
                    ok_in_reticle = center_in_rect(last_box, reticle)
                else:
                    ok_in_reticle = (iou(last_box, reticle) >= min_reticle_iou)

                if not ok_in_reticle:
                    last_box = None
                    last_crop = None
                    has_obj = False

            # bbox (solo si válido dentro del cuadro)
            if last_box is not None:
                x1,y1,x2,y2 = last_box
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            sharp = laplacian_sharpness(last_crop) if has_obj else 0.0
            cd = center_distance(last_box, frame.shape) if has_obj else 9999
            sharp_ok_save = has_obj and (sharp >= min_sharpness_save)

            # ---------- Modo normal: reconocer + guardar ----------
            if (not scan_mode) and (not typing_mode):
                if not has_obj:
                    draw_traffic_light(frame, "red", "Pon el objeto dentro de la retícula", sharp=sharp, cd=int(cd))
                    live_label, live_score = None, 0.0
                elif not sharp_ok_save:
                    draw_traffic_light(frame, "red", "Falta enfoque (mejor luz / quieto)", sharp=sharp, cd=int(cd))
                    live_label, live_score = None, 0.0
                else:
                    out = recognize_crop(rec, last_crop)
                    live_label, live_score, _ = _normalize_recognizer_output(out)

                    if live_label is None:
                        draw_traffic_light(frame, "yellow", "No reconocido (usa S para enseñar)", sharp=sharp, cd=int(cd))
                    else:
                        if live_score >= min_recognition_conf:
                            draw_traffic_light(frame, "green", f"Reconocido: {live_label} ({live_score:.2f})", sharp=sharp, cd=int(cd))
                        else:
                            draw_traffic_light(frame, "yellow", f"Baja conf: {live_label} ({live_score:.2f})", sharp=sharp, cd=int(cd))

                    # Guardar inventario (si es confiable + cooldown)
                    if live_label is not None and sharp_ok_save and live_score >= min_recognition_conf:
                        now = time.time()
                        last = last_saved_by_label.get(live_label, 0.0)
                        if (now - last) >= inventory_cooldown_sec:
                            save_inventory_event(
                                label=live_label,
                                confidence=live_score,
                                box_xyxy=last_box,
                                crop_bgr=last_crop,
                                captures_dir=CAPTURES_DIR,
                                camera_id="cam0",
                                model_name="yolo+clip"
                            )
                            last_saved_by_label[live_label] = now
                            status = f"📦 Guardado: {live_label}"

            # ---------- Scan guiado ----------
            if scan_mode and scan_label and stage_idx < len(scan_targets):
                stage_name, stage_quota = scan_targets[stage_idx]

                cooldown_ok = (frame_idx - scan_last_saved_frame) >= scan_cooldown_frames
                sharp_ok = has_obj and (sharp >= scan_min_sharp)

                a_now = box_area(last_box) if has_obj else 0
                asp_now = box_aspect(last_box) if has_obj else 0.0

                stage_ok = False
                if has_obj:
                    stage_ok = stage_satisfied(stage_name, last_box, reticle, a_now, area_ref, asp_now, aspect_ref)

                if not has_obj:
                    light, msg = "red", "Pon el objeto dentro de la retícula"
                elif not sharp_ok:
                    light, msg = "red", "Falta enfoque (mejor luz / quieto)"
                elif not stage_ok:
                    light, msg = "yellow", stage_instruction(stage_name)
                elif not cooldown_ok:
                    light, msg = "yellow", "Esperando (cooldown)"
                else:
                    light, msg = "green", "Listo para capturar"

                draw_traffic_light(frame, light, msg, sharp=sharp, cd=int(cd))

                cv2.putText(
                    frame,
                    f"ESCANEANDO: {scan_label} | ETAPA: {stage_name} ({stage_done}/{stage_quota}) | TOTAL: {scan_total_done}/20",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2
                )

                if light == "green":
                    emb = rec.embed_bgr(last_crop)

                    distinct_ok = True
                    if scan_last_emb is not None:
                        dot = float(np.dot(emb, scan_last_emb))
                        dist = 1.0 - dot
                        distinct_ok = dist >= scan_min_distinct

                    if not distinct_ok:
                        draw_traffic_light(frame, "yellow", "Varía más (gira/inclina)", sharp=sharp, cd=int(cd))
                    else:
                        save_object(scan_label, emb, datetime.now().isoformat(timespec="seconds"))

                        # también guardamos en inventario para demo
                        save_inventory_event(
                            label=scan_label,
                            confidence=1.0,
                            box_xyxy=last_box,
                            crop_bgr=last_crop,
                            captures_dir=CAPTURES_DIR,
                            camera_id="cam0",
                            model_name="learn-scan"
                        )

                        scan_last_emb = emb
                        scan_last_saved_frame = frame_idx
                        stage_done += 1
                        scan_total_done += 1
                        status = f"📌 Captura {scan_total_done}/20 (guardado en DB)"

                        if stage_name == "center":
                            area_ref = a_now if area_ref == 0 else area_ref
                            aspect_ref = asp_now if aspect_ref == 0.0 else aspect_ref

                        if stage_done >= stage_quota:
                            stage_idx += 1
                            stage_done = 0

                        if scan_total_done % 3 == 0 or scan_total_done == 20:
                            rec.reload()
                            last5 = get_last_learned(5)
                            counts = get_label_counts(15)

                if scan_total_done >= 20:
                    rec.reload()
                    last5 = get_last_learned(5)
                    counts = get_label_counts(15)
                    scan_mode = False
                    status = f"✅ Escaneo completado: {scan_label} (20 muestras)"
                    scan_label = ""
                    stage_idx = 0
                    stage_done = 0
                    scan_total_done = 0
                    area_ref = 0
                    aspect_ref = 0.0

            # ---------- Panel info ----------
            lines = [
                "Q: salir | P: toggle personas | S: escanear guiado (tipo FaceID)",
                f"Ignorar personas: {'ON' if ignore_person else 'OFF'}",
                f"Retícula: center-in={require_center_in_reticle} | iou>={min_reticle_iou:.2f}",
                f"Calidad: sharp>={min_sharpness_save:.0f} | conf>={min_recognition_conf:.2f} | TTL={track_ttl_frames}",
            ]
            if typing_mode:
                lines.append(f"Etiqueta scan: {typed_text}_ (Enter confirma / ESC cancela)")
            if status:
                lines.append(status)

            lines.append("Ultimos 5 aprendidos:")
            for lbl, ts in last5:
                tshow = ts.split('T')[-1] if 'T' in ts else ts
                lines.append(f"- {lbl} ({tshow})")

            lines.append("Top muestras (15):")
            for lbl, c in counts:
                lines.append(f"{lbl}: {c}")

            draw_lines(frame, lines, x=10, y=140)

            cv2.imshow(window, frame)

            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break

            key = cv2.waitKey(1) & 0xFF

            # salir
            if key == ord("q"):
                break

            # ESC
            if key == 27:
                if scan_mode:
                    scan_mode = False
                    scan_label = ""
                    status = "✳️ Escaneo cancelado"
                    stage_idx = 0
                    stage_done = 0
                    scan_total_done = 0
                    area_ref = 0
                    aspect_ref = 0.0
                    continue
                if typing_mode:
                    typing_mode = False
                    typed_text = ""
                    status = "✳️ Escritura cancelada"
                    continue
                break

            # typing mode
            if typing_mode:
                if key == 8:
                    typed_text = typed_text[:-1]
                    continue
                if key in (13, 10):
                    label = typed_text.strip()
                    if not label:
                        status = "❌ Escribe una etiqueta"
                        continue
                    scan_label = label
                    typing_mode = False
                    typed_text = ""
                    status = f"🟢 Scan iniciado: {scan_label}"
                    continue
                if 32 <= key <= 126:
                    typed_text += chr(key)
                continue

            # controles normales
            if key == ord("p"):
                ignore_person = not ignore_person
                status = f"Ignorar personas: {'ON' if ignore_person else 'OFF'}"

            if key == ord("s"):
                scan_mode = True
                scan_label = ""
                stage_idx = 0
                stage_done = 0
                scan_total_done = 0
                scan_last_emb = None
                scan_last_saved_frame = -9999
                area_ref = 0
                aspect_ref = 0.0

                typing_mode = True
                typed_text = ""
                status = "🟡 SCAN: escribe etiqueta y Enter para empezar (ESC cancela)"

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Cámara liberada y ventanas cerradas.")

if __name__ == "__main__":
    main()
