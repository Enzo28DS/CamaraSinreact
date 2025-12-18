# Detección de objetos (YOLOv8)

Este proyecto contiene un script `camara.py` que realiza detección de objetos en tiempo real usando YOLOv8 (`ultralytics`). Está orientado a precisión y permite ajustar parámetros para mejorar exactitud.

Requisitos
- Python 3.8+
- Cámara conectada (índice 0 por defecto)

Instalación
1. Crear un entorno virtual (recomendado):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Instalar dependencias:

```powershell
pip install -r requirements.txt
```


Uso
```powershell
python camara.py                          # inicia con el modelo por defecto (yolov8m)
python camara.py --model yolov8l.pt --conf 0.5 --imgsz 1280 --device cuda
```

Controles dentro de la ventana
- Presiona `q` para cerrar la ventana y terminar el script.

Notas sobre detección de objetos

- El script usa `ultralytics` (YOLOv8). Si `ultralytics` no se puede instalar o hay problemas con la descarga del modelo, el script mostrará un mensaje en pantalla.
- Por precisión, puedes usar modelos más grandes: `yolov8m.pt`, `yolov8l.pt` o `yolov8x.pt`. Ten en cuenta que modelos más grandes consumen más memoria y CPU/GPU.

Consejos para más precisión
- Aumenta `--conf` (ej. 0.5) para reducir falsos positivos.
- Aumenta `--imgsz` (ej. 1280) para mejorar detección de objetos pequeños/precisos (a costa de velocidad).
- Si tienes GPU CUDA, pásala con `--device cuda` para acelerar y poder usar modelos más grandes.

Consejos para mejorar rendimiento (si la cámara va lenta)
- Usa el modo rápido: `--fast` (usa `yolov8n` y `imgsz=640`).
- Reduce `--imgsz` (ej. 640 o 512) para acelerar la inferencia.
- Incrementa `--skip` para procesar menos frames (ej. `--skip 2` procesa 1 de cada 3 frames).
- Usa `--device cuda` si tienes GPU para acelerar (es la forma más efectiva de mejorar FPS).

Notas
Si tienes problemas
- Revisa la salida de `pip install -r requirements.txt` para detectar errores de compilación.
- En Windows, la instalación de algunas dependencias puede requerir Microsoft Build Tools o ruedas binarias compatibles.

