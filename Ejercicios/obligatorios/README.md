# Ejercicios Obligatorios

Compilación de los ejercicios obligatorios de la asignatura de Visión por Computador (UMU).

## 1. CALIBRACIÓN

**Ubicación:** `code/calibrate/`

Calibración de cámara usando imágenes de un tablero de ajedrez (chessboard).

### Scripts principales:
- **calibrate.py** - Realiza la calibración usando múltiples imágenes
- **verify.py** - Verifica la calidad de la calibración
- **undistort1.py** - Elimina distorsión mediante matriz de calibración
- **undistort2.py** - Método alternativo de rectificación
- **medida.py** - **Cuadrícula de medida sobre plano perpendicular** (resuelve requisito de overlay con slider para distancia Z)
- **triangulate.py** - Triangulación de puntos para estéreo
- **calib.txt** - Archivo con matriz de calibración (parámetro cx, cy, fx, fy)

### Ejecución:
```bash
# Calibrar cámara
python code/calibrate/calibrate.py

# Mostrar cuadrícula de medida con slider de distancia Z
python code/calibrate/medida.py
```

---

## 2. ANÁLISIS DE TRÁFICO

**Ubicación:** `code/color/`

Análisis automático de secuencias de vídeo de tráfico vehicular.

### Scripts principales:
- **trafico.py** ⭐ - Detector y contador de vehículos por sentido (implementación principal)
- **backsub.py** / **backsub0.py** - Sustracción de fondo para detección de movimiento
- **camshift.py** - Tracking de objetos en movimiento (vehículos)
- **mean-shift.py** - Segmentación de color para clasificar vehículos
- **histogram.py** - Análisis de histogramas de color

### Ejecución:
```bash
# Análisis de tráfico con contador vehicular
python code/color/trafico.py --video <url_o_video.mp4>

# Ejemplo con URL del alias "carretera"
python code/color/trafico.py --video <carretera_url>
```

### Salida:
- Gráficas de vehículos por hora
- Contadores por sentido (izquierda/derecha)
- Detección de horas punta

---

## 3. ACTIVIDAD

**Ubicación:** `code/bot/` y `code/LK/`

Detector de movimiento en región de interés (ROI).

### Scripts principales:
- **code/bot/actividad.py** ⭐ - Detector de actividad con guardado de video y notificación Telegram
- **code/LK/lk_track.py** - Tracking con Lucas-Kanade base
- **code/opcionales/lk_cam_motion.py** - Extensión con detección de dirección y velocidad angular

### Características:
✅ Detección de movimiento en región manual  
✅ Guardado de 2-3 segundos de vídeo  
✅ Filtrado por categoría (usando clasificador)  
✅ Anonimización de personas  
✅ Envío de notificación via bot Telegram  

### Ejecución:
```bash
# Detector de actividad
python code/bot/actividad.py

# Con token de Telegram (configurar en code/bot/token.env)
python code/bot/actividad.py --telegram
```

---

## 4. CLASIFICADOR

**Ubicación:** `code/clasificador/`

Sistema modular de reconocimiento de imágenes con multiples métodos de comparación.

### Scripts principales:
- **clasificador.py** ⭐ - Interfaz principal (argumento `--method=` y `--models=`)
- **metodos/embedder.py** - Método de embedding (MediaPipe)
- **metodos/sift.py** - Reconocedor SIFT (objetos texturados)
- **metodos/gestos.py** - Clasificador de gestos de manos (Procrustes)
- **embedder.tflite** - Modelo preentrenado MediaPipe
- **modelos_embedding/** - Carpeta con imágenes de modelos
- **modelos_gestos/** - Modelos de gestos de manos
- **modelos_sift/** - Imágenes para matching SIFT

### Métodos disponibles:
- `embedding` - MediaPipe embeddings + distancia euclidea
- `sift` - Matching keypoints SIFT + ratio test
- `gestos` - Gesturas de manos (Procrustes distance)

### Ejecución:
```bash
# Clasificar con embedding
python code/clasificador/clasificador.py --models=clasificador/modelos_embedding --method=embedding

# Clasificar con SIFT
python code/clasificador/clasificador.py --models=clasificador/modelos_sift --method=sift

# Gesturas de mano
python code/clasificador/clasificador.py --method=gestos
```

---

## 5. DL (Deep Learning)

**Ubicación:** `code/DL/`

Entrenamiento y uso de modelos de deep learning.

### Subcarpetas principales:

#### CNN (Clasificación)
- `code/DL/CNN/` - Redes convolucionales con Keras/TensorFlow
- Ejemplos: clasificación de dígitos, imágenes personalizadas

#### UNET (Segmentación)
- `code/DL/UNET/train.py` - Entrenamiento de red UNET
- `code/DL/UNET/rununet.py` - Ejecución

#### YOLO (Detección de Objetos)
- `code/DL/yolo/yolo.py` - Inferencia YOLO
- `code/DL/yolotrain/yolo_run.py` - Entrenamiento con datos personalizados
- Soporte: detección, tracking, segmentación

#### MediaPipe Models
- `code/DL/mp_hands/` - Detección de manos
- `code/DL/mp_humanpose/` - Estimación de pose corporal
- `code/DL/mp_face/` - Detección facial
- `code/DL/mp_embedder/` - Embeddings para similitud

#### Vision Transformers
- `code/DL/DINO/dino*.py` - DINO (Vision Transformer)
- `code/DL/CLIP/runclip.py` - CLIP (text-image matching)

### Ejecución (ejemplo YOLO):
```bash
# Entrenar YOLO con imágenes propias
python code/DL/yolotrain/yolo_run.py --dataset /ruta/al/dataset

# Inferencia YOLO
python code/DL/yolo/yolo.py --video <video.mp4>
```

---

## 6. RECTIFICACIÓN

**Ubicación:** `code/rectifica/`

Corrección de perspectiva para mediciones en planos.

### Scripts principales:
- **rectifica.py** ⭐ - Rectificación interactiva de planos

### Funcionalidad:
✅ Seleccionar puntos de referencia manual en región de interés  
✅ Proporcionar coordenadas reales (archivo .txt)  
✅ Calcular homografía de destorsionamiento  
✅ Medir distancias en plano rectificado  
✅ Mostrar resultado sobre imagen original  

### Ejemplos:
- Medir distancia entre monedas (`images/coins.png`)
- Distancia de disparo en imágenes de gol

### Ejecución:
```bash
# Rectificar con archivo de referencia
python code/rectifica/rectifica.py --image <img.jpg> --points <ref.txt>
```

### Formato archivo referencia (ref.txt):
```
x1 y1 real_x1 real_y1
x2 y2 real_x2 real_y2
...
```

---

## Resumen Rápido

| Ejercicio | Carpeta | Script Principal | Comando |
|-----------|---------|------------------|---------|
| Calibración | `code/calibrate/` | `medida.py` | `python code/calibrate/medida.py` |
| Tráfico | `code/color/` | `trafico.py` | `python code/color/trafico.py` |
| Actividad | `code/bot/` | `actividad.py` | `python code/bot/actividad.py` |
| Clasificador | `code/clasificador/` | `clasificador.py` | `python code/clasificador/clasificador.py` |
| DL | `code/DL/` | `yolotrain/yolo_run.py` | `python code/DL/yolotrain/yolo_run.py` |
| Rectificación | `code/rectifica/` | `rectifica.py` | `python code/rectifica/rectifica.py` |

---

## Notas

- Todos los scripts utilizan **OpenCV** y librerías estándar de Python
- Los ejercicios DL requieren **TensorFlow/Keras** o **PyTorch**
- MediaPipe está incluido en varios ejercicios (clasificador, detección, actividad)
- Para Telegram bot: configurar token en `code/bot/token.env`
