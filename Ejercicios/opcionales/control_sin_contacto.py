#!/usr/bin/env python3
import cv2 as cv
import mediapipe as mp
import numpy as np
import argparse
import time
import os

parser = argparse.ArgumentParser(description='Controlador sin contacto: distancia y orientacion')
parser.add_argument('--hand-width', type=float, default=8.5, help='Ancho real mano cm (default 8.5)')
parser.add_argument('--fx', type=float, default=None, help='Focal (px)')
parser.add_argument('--camera', type=int, default=0, help='Índice de cámara')
args = parser.parse_args()

def read_fx_from_calib():
    candidates = ['calib.txt', '../calibrate/calib.txt', '../../code/calibrate/calib.txt']
    for p in candidates:
        try:
            if os.path.exists(p):
                vals = np.loadtxt(p)
                return vals[:9].reshape(3,3)[0,0]
        except: continue
    return None

fx = args.fx if args.fx else read_fx_from_calib()
if fx is None:
    fx = 800.0
    print('[WARN] Usando fx = 800 px por defecto.')
else:
    print(f'[INFO] Usando fx = {fx:.2f} px')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5)

cap = cv.VideoCapture(args.camera)

print('\n[INFO] Controlador activo. Muestra la mano para mover el Dron.')
print('[INFO] Pulsa "g" para capturar pantalla, o ESC para salir.')

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv.flip(frame, 1) # Efecto espejo para control intuitivo
    h, w = frame.shape[:2]
    
    results = hands.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))

    if results.multi_hand_landmarks:
        lm = results.multi_hand_landmarks[0]
        pts = np.array([[p.x * w, p.y * h, p.z] for p in lm.landmark])

        x_min, x_max = int(np.min(pts[:,0])), int(np.max(pts[:,0]))
        y_min, y_max = int(np.min(pts[:,1])), int(np.max(pts[:,1]))
        pw = x_max - x_min
        cx, cy = int((x_min + x_max)/2), int((y_min + y_max)/2)

        # Distancia Pinhole
        D_cm = (args.hand_width * fx) / float(pw) if pw > 0 else 0.0

        v_w, v_m = pts[0][:2], pts[9][:2]
        roll_deg = np.degrees(np.arctan2(v_m[1]-v_w[1], v_m[0]-v_w[0]))

        # Dibujar objeto controlado por rotación y escala según distancia
        escala_dron = max(20, int(3000 / (D_cm + 1))) 
        rect = ((cx, cy), (escala_dron, escala_dron), roll_deg)
        box = np.int0(cv.boxPoints(rect))
        cv.drawContours(frame, [box], 0, (0, 0, 255), 4)
        cv.putText(frame, "OBJETIVO FIJADO", (cx - 70, cy - escala_dron//2 - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Textos informativos
        cv.rectangle(frame, (x_min,y_min), (x_max,y_max), (0,255,0), 2)
        cv.putText(frame, f'Dist: {D_cm:.1f} cm', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv.putText(frame, f'Roll: {roll_deg:.1f} deg', (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (255,200,0), 2)

    cv.imshow('Control Sin Contacto', frame)

    key = cv.waitKey(1) & 0xFF
    if key == 27: break
    elif key == ord('g'):
        cv.imwrite("controlador_mano.png", frame)
        print("[OK] Captura guardada.")

cap.release()
cv.destroyAllWindows()