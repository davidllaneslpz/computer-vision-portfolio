import cv2 as cv
import numpy as np
import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands
from scipy.spatial import procrustes

# Inicializar detector de manos de MediaPipe
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def extraer_landmarks(imagen):
    """Extrae las coordenadas X,Y de los 21 puntos de la mano"""
    rgb = cv.cvtColor(imagen, cv.COLOR_BGR2RGB)
    resultados = hands.process(rgb)
    
    if not resultados.multi_hand_landmarks:
        return None
        
    puntos = []
    for landmark in resultados.multi_hand_landmarks[0].landmark:
        puntos.append([landmark.x, landmark.y])
    return np.array(puntos)

def precomputar(imagen):
    """El modelo es la forma geométrica de la mano (21 puntos)"""
    return extraer_landmarks(imagen)

def comparar(frame, caracteristicas_modelo):
    """Usa Procrustes para comparar formas.
    Devuelve un valor de 'confianza' (1/disparidad)"""
    puntos_modelo = caracteristicas_modelo
    if puntos_modelo is None:
        return 0.0
        
    puntos_frame = extraer_landmarks(frame)
    if puntos_frame is None:
        return 0.0
        
    # Procrustes devuelve (modelo_alineado, frame_alineado, disparidad)
    # Disparidad cercana a 0 significa que son casi iguales
    try:
        _, _, disparidad = procrustes(puntos_modelo, puntos_frame)
        # Convertimos la disparidad en una "confianza" para que sea más alta cuanto más se parezca
        # Evitamos división por cero sumando un epsilon
        confianza = 1.0 / (disparidad + 0.001)
        return confianza
    except ValueError:
        # A veces procrustes falla si los puntos están muy deformados
        return 0.0