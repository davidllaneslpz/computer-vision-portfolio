import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from umucv.util import check_and_download

# Descargar modelo si no existe
check_and_download("embedder.tflite", "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/1/mobilenet_v3_small.tflite")

# Inicializar embedder
base_options = python.BaseOptions(model_asset_path='embedder.tflite')
options = vision.ImageEmbedderOptions(base_options=base_options, l2_normalize=True, quantize=False)
embedder = vision.ImageEmbedder.create_from_options(options)

def precomputar(imagen):
    """Convierte la imagen en un vector matemático (embedding)"""
    rgb = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    resultado = embedder.embed(mp_image)
    # Devolvemos el vector (embedding)
    return resultado.embeddings[0]

def comparar(frame, caracteristicas_modelo):
    """Calcula la similitud del coseno entre el frame y el modelo.
    Devuelve un valor de 0 a 1 (1 es idéntico)"""
    embedding_modelo = caracteristicas_modelo
    
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    resultado_frame = embedder.embed(mp_image)
    embedding_frame = resultado_frame.embeddings[0]
    
    # La API de MediaPipe tiene una función nativa para comparar
    similitud = vision.ImageEmbedder.cosine_similarity(embedding_modelo, embedding_frame)
    
    return similitud