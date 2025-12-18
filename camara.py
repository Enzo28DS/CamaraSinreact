# camara.py
import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

from db import init_db, insert_detection

# (Opcional) Clasificación por CLIP sobre el recorte
try:
    from clip_classifier import ClipClassifier
except Exception:
    ClipClassifier = None


def parse_args():
    p = argparse.ArgumentParser(description="YOLO + Inventario SQLite + (opcional) CLIP para mayor variedad")
    p.add_argument("--model", type=str, default="yolov8m.pt", help="Modelo YOLO (recomendado: yolov8m.pt o yolov8l.pt)")
    p.add_argument("--cam", type=int, default=0, help="Índice de cámara")
    p.add_argument("--conf", type=float, default=0.55, help="Umbral de confianza (sube para menos falsos positivos)")
    p.add_argument("--iou", type=float, default=0.50, help="Umbral IoU NMS")
    p.add_argument("--imgsz", type=int, default=960, help="Tamaño de inferencia (más alto = mejor, más lento)")
    p.add_argument("--device", type=str, default="cuda", help="cuda | cpu | 0")
    p.add_argument("--show", action="store_true", help="Mostrar ventana con detecciones")
    p.add_argument("--save-captures", action="store_true", help="Guardar captura cuando se registra en BD")
    p.add_argument("--captures-dir", type=str, default="captures", help="Carpeta de capturas")
    p.add_argument("--cooldown", type=float, default=3.0, help="Segundos mínimos entre registros del mismo label")
    p.add_argument("--min-area", type=int, default=2500, help="Ignorar detecciones muy pequeñas (área bbox)")

    # Filtros por label (para no llenar inventario con personas)
    p.add_argument("--block-labels", type=str, default="person",
                   help="Labels a ignorar (coma-separado). Ej: person,chair")
    p.add_argument("--allow-labels", type=str, default="",
                   help="Si se define, SOLO se permiten estos labels (coma-separado).")

    # CLIP (para “variedad” sin entrenar)
    p.add_argument("--use-clip", action="store_true",
                   help="Usar CLIP para clasificar el recorte del bbox en labels personalizados")
    p.add_argument("--clip-labels", type=str, default="celular,lentes,mouse,caja_audifonos",
                   help="Labels que CLIP puede elegir (coma-separado)")

    return p.parse_args()


def parse_csv_set(s: str):
    s = (s or "").strip()
    if not s:
        return set()
    return {x.strip() for x in s.split(",") if x.strip()}


def crop_safe(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def main():
    args = parse_args()
    init_db()

    block = parse_csv_set(args.block_labels)
    allow = parse_csv_set(args.allow_labels)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir la cámara {args.cam}")

    model = YOLO(args.model)
    names = model.names

    captures_dir = Path(args.captures_dir)
    if args.save_captures:
        captures_dir.mkdir(parents=True, exist_ok=True)

    # anti-duplicados por label (ya sea YOLO o CLIP)
    last_saved_ts = {}

    # FPS
    t_prev = time.time()
    fps = 0.0

    camera_id = f"cam_{args.cam}"
    model_id = Path(args.model).name

    # CLIP classifier (si aplica)
    clip_labels = [x.strip() for x in args.clip_labels.split(",") if x.strip()]
    clipper = None
    if args.use_clip:
        if ClipClassifier is None:
            raise RuntimeError("No se pudo importar clip_classifier.py. Instala dependencias o revisa el archivo.")
        clipper = ClipClassifier(device=("cuda" if args.device != "cpu" else "cpu"), labels=clip_labels)

    window_name = "Inventario - Deteccion"
    print("✅ Controles: [Q] salir | [ESC] salir | cerrar ventana (X) para salir")
    print(f"🔧 YOLO: model={model_id} imgsz={args.imgsz} conf={args.conf} iou={args.iou} device={args.device}")
    if args.use_clip:
        print(f"🧠 CLIP activo. Labels: {clip_labels}")
    if block:
        print(f"🚫 Block labels: {sorted(block)}")
    if allow:
        print(f"✅ Allow labels: {sorted(allow)}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("⚠️ No pude leer frame de la cámara.")
                break

            results = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False
            )

            now = time.time()
            dt = now - t_prev
            t_prev = now
            fps = (0.9 * fps + 0.1 * (1.0 / dt)) if dt > 0 else fps

            annotated = frame.copy()
            r0 = results[0]
            boxes = r0.boxes

            if boxes is not None and len(boxes) > 0:
                for b in boxes:
                    cls_id = int(b.cls[0])
                    yolo_label = names.get(cls_id, str(cls_id))
                    conf = float(b.conf[0])
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

                    area = (x2 - x1) * (y2 - y1)
                    if area < args.min_area:
                        continue

                    # Filtros por YOLO label (antes de CLIP)
                    if yolo_label in block:
                        continue
                    if allow and (yolo_label not in allow):
                        continue

                    final_label = yolo_label
                    final_conf = conf

                    # CLIP: re-etiquetar según tus categorías (celular/lentes/mouse/etc.)
                    if clipper is not None:
                        crop = crop_safe(frame, x1, y1, x2, y2)
                        if crop is not None:
                            clip_label, clip_score = clipper.classify_bgr(crop)
                            # Puedes ajustar esta lógica:
                            # - siempre usar CLIP
                            # - o usar CLIP solo si YOLO dice "person" / etiqueta rara
                            final_label = clip_label
                            final_conf = float(clip_score)

                    # cooldown por FINAL label
                    last = last_saved_ts.get(final_label, 0.0)
                    can_save = (now - last) >= args.cooldown

                    # Dibujar bbox + etiqueta final
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    txt = f"{final_label} {final_conf:.2f}"
                    cv2.putText(
                        annotated, txt, (x1, max(20, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )

                    if can_save:
                        ts = datetime.now()
                        image_path = None

                        if args.save_captures:
                            fname = f"{ts.strftime('%Y%m%d_%H%M%S')}_{final_label}.jpg".replace(" ", "_")
                            out = captures_dir / fname
                            cv2.imwrite(str(out), frame)
                            image_path = str(out)

                        insert_detection(
                            ts=ts,
                            camera_id=camera_id,
                            model=(model_id + ("+CLIP" if args.use_clip else "")),
                            label=final_label,
                            confidence=final_conf,
                            box_xyxy=(x1, y1, x2, y2),
                            image_path=image_path
                        )
                        last_saved_ts[final_label] = now
                        print(f"[DB] {ts.strftime('%Y-%m-%d %H:%M:%S')}  {final_label}  conf={final_conf:.2f}  img={image_path}")

            # Overlay
            cv2.putText(
                annotated, f"FPS: {fps:.1f} | model={model_id} | device={args.device}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
            if args.use_clip:
                cv2.putText(
                    annotated, f"CLIP labels: {', '.join(clip_labels[:4])}" + ("..." if len(clip_labels) > 4 else ""),
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
                )
            cv2.putText(
                annotated, "Q/ESC para salir | X para cerrar",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
            )

            if args.show:
                cv2.imshow(window_name, annotated)

                # Cerrar con X
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("🧹 Ventana cerrada (X). Saliendo...")
                    break

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("🧹 Salida por teclado (Q/ESC).")
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Cámara liberada y ventanas cerradas.")


if __name__ == "__main__":
    main()
