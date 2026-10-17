#!/usr/bin/env python
import cv2 as cv
import numpy as np
import mediapipe as mp
import os
from umucv.stream import autoStream

# 1. Creamos las carpetas exactas que pide YOLO (si no existen)
os.makedirs("datasets/mano/images/train", exist_ok=True)
os.makedirs("datasets/mano/labels/train", exist_ok=True)

# 2. Inicializamos MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

N = 0
print("--- AUTO ETIQUETADOR INICIADO ---")
print("Mueve la mano por la pantalla (cerca, lejos, a los lados).")
print("Se guardará 1 foto por segundo automáticamente...")

for k, (key, frame) in enumerate(autoStream()):
    # Para YOLO, la imagen base no debe tener dibujos, así que hacemos una copia limpia
    frame_limpio = frame.copy()
    h, w = frame.shape[:2]
    
    image_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    resultados = hands.process(image_rgb)

    if resultados.multi_hand_landmarks:
        # Cogemos los 21 puntos de la mano
        hand_landmarks = resultados.multi_hand_landmarks[0]
        
        # Calculamos los límites (el cuadradito que la envuelve)
        x_min = min([landmark.x for landmark in hand_landmarks.landmark])
        x_max = max([landmark.x for landmark in hand_landmarks.landmark])
        y_min = min([landmark.y for landmark in hand_landmarks.landmark])
        y_max = max([landmark.y for landmark in hand_landmarks.landmark])

        # Dejamos un poco de margen para que no corte los dedos
        margen = 0.05
        x_min = max(0, x_min - margen)
        x_max = min(1, x_max + margen)
        y_min = max(0, y_min - margen)
        y_max = min(1, y_max + margen)

        # Dibujamos el rectángulo en la pantalla para que veas qué está capturando
        x0, y0 = int(x_min * w), int(y_min * h)
        x1, y1 = int(x_max * w), int(y_max * h)
        cv.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv.putText(frame, f"Capturas: {N}/40", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        # Guardamos 1 de cada 30 fotogramas (1 por segundo aprox)
        if k % 30 == 29 and N < 40:
            N += 1
            # Formato YOLO: Clase X_centro Y_centro Ancho Alto (todo de 0 a 1)
            xc = (x_min + x_max) / 2
            yc = (y_min + y_max) / 2
            bw = x_max - x_min
            bh = y_max - y_min

            # Guardamos la foto LIMPIA y el archivo de texto
            cv.imwrite(f"datasets/mano/images/train/{N:03}.jpg", frame_limpio)
            with open(f"datasets/mano/labels/train/{N:03}.txt", "w") as f:
                f.write(f"0 {xc:.5f} {yc:.5f} {bw:.5f} {bh:.5f}")
            print(f"-> Foto y etiqueta {N}/40 guardadas.")

        if N >= 40:
            cv.putText(frame, "¡DATASET COMPLETADO! Pulsa ESC", (20, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv.imshow('Auto-Etiquetador YOLO', frame)

cv.destroyAllWindows()