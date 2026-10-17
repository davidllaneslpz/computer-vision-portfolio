import cv2 as cv
import numpy as np
from umucv.stream import autoStream

# 1. Cargar los datos de calibración
try:
    calib_data = np.loadtxt('calib.txt')
    K = calib_data[:9].reshape(3,3)
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
except Exception as e:
    print("Error leyendo calib.txt.")
    exit()

def nothing(x): pass

cv.namedWindow("Medida")
# Slider para la distancia Z (profundidad)
cv.createTrackbar("Z (cm)", "Medida", 50, 500, nothing)
# NUEVO: Slider para la Altura de la cámara (A) respecto al suelo
cv.createTrackbar("Altura (cm)", "Medida", 80, 200, nothing) 

fov_calculado = False

for key, frame in autoStream():
    h, w = frame.shape[:2]
    
    if not fov_calculado:
        fov_x = 2 * np.arctan(w / (2 * fx)) * 180 / np.pi
        fov_y = 2 * np.arctan(h / (2 * fy)) * 180 / np.pi
        print(f"FOV Horizontal: {fov_x:.2f}° | FOV Vertical: {fov_y:.2f}°")
        fov_calculado = True

    Z = cv.getTrackbarPos("Z (cm)", "Medida")
    if Z == 0: Z = 1
    
    # Leer la altura de la cámara
    A = cv.getTrackbarPos("Altura (cm)", "Medida")

    # Línea del horizonte (donde apunta el centro de la cámara)
    cv.line(frame, (0, int(cy)), (w, int(cy)), (150, 150, 150), 2)

    paso_cm = 10
    limite_x = 150 
    
    # La parte superior del plano (por ejemplo, 100 cm por encima de la cámara)
    y_min = -100 
    # La base del plano es exactamente la altura a la que está la cámara (A)
    y_max = A 

    # Líneas verticales
    for x in range(-limite_x, limite_x + paso_cm, paso_cm):
        u = int(fx * (x / Z) + cx)
        # Dibujamos desde arriba (y_min) hasta el suelo (y_max)
        v_top = int(fy * (y_min / Z) + cy)
        v_bottom = int(fy * (y_max / Z) + cy)
        cv.line(frame, (u, v_top), (u, v_bottom), (0, 255, 0), 1)
        
    # Líneas horizontales
    for y in range(y_min, y_max + paso_cm, paso_cm):
        v = int(fy * (y / Z) + cy)
        # Si es la línea del suelo (y == A), la hacemos más gruesa
        grosor = 3 if y == A else 1
        cv.line(frame, (0, v), (w, v), (0, 255, 0), grosor)

    cv.imshow("Medida", frame)