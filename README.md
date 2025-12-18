# Inventario inteligente (solo Python)

Sistema de visión en tiempo real para inventario automático con YOLOv8 + CLIP. Toda la lógica central está en Python (FastAPI para la API, OpenCV para cámara y CLIP para reconocimiento). El frontend React no es necesario para la hackatón; esta es una mejora futura, pero la entrega se basa en los servicios y scripts de Python.

## Arquitectura Python
- **`vision_service.py`**: Servicio reutilizable que toma frames BGR, detecta/recorta con YOLOv8, filtra por calidad (enfoque/retícula) y reconoce con `Recognizer` (CLIP). Guarda eventos en SQLite (`inventario.db`) y soporta modo aprendizaje incremental (`learn_frame`).
- **`main.py` (FastAPI)**: Expone endpoints REST para stats/export, para enviar frames de inventario (`/vision/frame`) y para aprender nuevas etiquetas (`/learn/frame`). Monta `/captures/` para ver las imágenes guardadas.
- **`db.py`**: Manejo de SQLite de inventario (`detections`), exportación CSV y limpieza.
- **`learned_db.py`**: Almacena embeddings CLIP por etiqueta y devuelve resúmenes de muestras aprendidas.
- **`recognizer.py`**: CLIP + prototipos + fallback kNN para reconocer objetos ya aprendidos.
- **`clip_classifier.py`**: Clasificador zero-shot con CLIP para re-etiquetar recortes cuando se usa en `camara.py`.
- **`camara.py`**: Script de cámara en vivo con YOLOv8 (opcional CLIP zero-shot). Inserta detecciones en `inventario.db` y puede guardar capturas.
- **`camara_learn.py`**: Flujo “FaceID” guiado para escanear 20 vistas de un objeto, generar embeddings CLIP y dejarlos listos para reconocimiento inmediato.

## Requisitos
- Python 3.9+
- GPU CUDA opcional pero recomendada (torch + YOLO + CLIP). En CPU funciona más lento.
- Cámara accesible en el sistema (índice 0 por defecto).

## Instalación rápida
```bash
python -m venv .venv
source .venv/bin/activate      # en Windows: .venv\\Scripts\\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Descarga los pesos YOLOv8 (colócalos en la raíz del proyecto):
- `yolov8m.pt` (por defecto en scripts)
- `yolov8l.pt` / `yolov8n.pt` / `yolov8l.pt` (opcional según rendimiento)

## Cómo correr la API (FastAPI + CLIP + YOLO)
1) Inicializa las bases (se crean solas al arrancar):
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

2) Endpoints clave:
- `GET /stats` y `GET /counts`: resumen y top de labels en inventario.
- `GET /detections?page=1&page_size=25&label=...&last_minutes=60`: lecturas paginadas/filtradas.
- `GET /export.csv`: exporta las detecciones a CSV.
- `POST /vision/frame` (multipart `file`): procesa un frame BGR, aplica retícula, valida enfoque y guarda si la confianza es suficiente.
- `POST /learn/frame` (multipart `file`, query `label`): guarda embeddings CLIP de un objeto y recarga el reconocedor al instante.
- `GET /learned/summary`: muestra últimos aprendidos y top etiquetas en `learned_objects.db`.
- `POST /inventory/clear` con header `X-Token`: limpia `inventario.db` (token por env `INVENTORY_TOKEN`, default `1234`).

3) Capturas:
- Se guardan en `/captures` y quedan servidas como estáticos. `main.py` monta esa carpeta automáticamente.

## Scripts de cámara (solo Python)
### Detección + inventario (`camara.py`)
```bash
python camara.py --show                 # ventana con FPS + controles
python camara.py --model yolov8l.pt --conf 0.5 --imgsz 1280 --device cuda --save-captures
python camara.py --use-clip --clip-labels "celular,lentes,mouse,caja_audifonos"
```
Controles: `q` o `ESC` para salir. Usa cooldown por etiqueta, filtrado de áreas pequeñas y bloqueo de personas por defecto.

### Escaneo guiado tipo FaceID (`camara_learn.py`)
```bash
python camara_learn.py
```
- Tecla `S`: inicia escaneo guiado (20 vistas: center/left/right/up/down/near/far/tilt).
- Ingresa la etiqueta y el script guarda embeddings en `learned_objects.db`, recarga `Recognizer` y opcionalmente inserta capturas en inventario.
- Tecla `P`: alterna ignorar personas. `Q` o `ESC`: salir/cancelar.

## Consideraciones de rendimiento
- **Precisión**: sube `--conf` y `--imgsz` (ej. 1280) en `camara.py`, o usa `yolov8l.pt`.
- **Velocidad**: baja `--imgsz`, usa `yolov8n.pt`, o GPU (`--device cuda`). En API, ajusta `VisionService` (`imgsz`, `yolo_conf`, `min_sharp`, `min_recog_conf`).
- **Calidad de enfoque**: `VisionService` y `camara_learn.py` miden `Laplacian` para asegurar nitidez antes de guardar.

## Bases de datos
- `inventario.db`: detecciones (ts, cámara, modelo, label, confianza, bbox, ruta de imagen). Se crea en la raíz junto a `db.py`.
- `learned_objects.db`: embeddings CLIP por etiqueta para reconocimiento incremental.

## Próximos pasos / mejoras opcionales
- Integrar el frontend React existente como panel (no necesario para la competencia, solo mencionarlo como mejora).
- Añadir autenticación real en la API y tokens por cámara.
- Empaquetar en contenedor con GPU y health checks.
-Crear una base de datos mas robusta
-mayor mejoras en la presicion de deteccion y reconocimiento de las imagenes


