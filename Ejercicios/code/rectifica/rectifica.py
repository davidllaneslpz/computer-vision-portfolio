#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import argparse
import math

parser = argparse.ArgumentParser(description="Rectificacion de perspectiva y medicion")
parser.add_argument('--img', type=str, required=True, help='Imagen a rectificar')
parser.add_argument('--ref', type=str, required=True, help='Archivo de puntos de referencia')
parser.add_argument('--out', type=str, default='rectificada.png', help='Archivo de salida')
parser.add_argument('--scale', type=float, default=100, help='Escala: píxeles por unidad real (por defecto 100)')
args = parser.parse_args()

# Leer imagen
img = cv.imread(args.img)
if img is None:
    print(f"No se pudo abrir la imagen {args.img}")
    exit(1)

# Leer puntos de referencia
def leer_puntos(archivo):
    pts_img, pts_real = [], []
    with open(archivo) as f:
        for line in f:
            if line.strip() == '' or line.strip().startswith('#'):
                continue
            tokens = line.strip().split()
            if len(tokens) < 4:
                continue
            x_img, y_img, x_real, y_real = map(float, tokens[:4])
            pts_img.append([x_img, y_img])
            pts_real.append([x_real, y_real])
    return np.array(pts_img, dtype=np.float32), np.array(pts_real, dtype=np.float32)

pts_img, pts_real = leer_puntos(args.ref)

# --- PARTE 1: CREAR IMAGEN RECTIFICADA ---
SCALE = args.scale
pts_real_scaled = pts_real * SCALE

H_scaled, _ = cv.findHomography(pts_img, pts_real_scaled)

min_xy = np.min(pts_real_scaled, axis=0)
max_xy = np.max(pts_real_scaled, axis=0)
size = (int(np.ceil(max_xy[0] - min_xy[0])), int(np.ceil(max_xy[1] - min_xy[1])))

T = np.array([[1, 0, -min_xy[0]], [0, 1, -min_xy[1]], [0, 0, 1]], dtype=np.float32)
H_shifted = T @ H_scaled

rectificada = cv.warpPerspective(img, H_shifted, size)
cv.imwrite(args.out, rectificada)

# --- PARTE 2: DIBUJAR PUNTOS ESTÁTICOS ---
img_estatica = img.copy()
for (x, y) in pts_img:
    cv.circle(img_estatica, (int(x), int(y)), 15, (0, 0, 255), -1)

for (x, y) in pts_real_scaled - min_xy:
    cv.circle(rectificada, (int(x), int(y)), 15, (0, 255, 0), -1)

def mostrar_redimensionada(nombre, imagen, max_w=800, max_h=600):
    h, w = imagen.shape[:2]
    escala = min(max_w / w, max_h / h, 1.0)
    img_show = cv.resize(imagen, (int(w*escala), int(h*escala))) if escala < 1.0 else imagen
    cv.imshow(nombre, img_show)

mostrar_redimensionada('Original con puntos', img_estatica)
mostrar_redimensionada('Rectificada', rectificada)

# --- PARTE 3: MEDICIÓN INTERACTIVA CON RATÓN ---
H_real_cm, _ = cv.findHomography(pts_img, pts_real)

img_interactiva = img.copy()
# Grosor ampliado para pantallas/fotos gigantes
cv.polylines(img_interactiva, [np.int32(pts_img)], True, (0, 0, 255), 6)
cv.putText(img_interactiva, "Referencia (A4)", (int(pts_img[0][0]), int(pts_img[0][1])-30), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

puntos_medicion = []

def click_raton(event, x, y, flags, param):
    global puntos_medicion, img_interactiva
    if event == cv.EVENT_LBUTTONDOWN:
        puntos_medicion.append((x, y))
        cv.circle(img_interactiva, (x, y), 15, (255, 0, 0), -1)
        
        if len(puntos_medicion) == 2:
            pt1_img = np.array([[[puntos_medicion[0][0], puntos_medicion[0][1]]]], dtype=np.float32)
            pt2_img = np.array([[[puntos_medicion[1][0], puntos_medicion[1][1]]]], dtype=np.float32)
            
            # Proyectar a la realidad (en cm)
            pt1_real = cv.perspectiveTransform(pt1_img, H_real_cm)[0][0]
            pt2_real = cv.perspectiveTransform(pt2_img, H_real_cm)[0][0]
            
            # Calcular distancia
            distancia = math.hypot(pt2_real[0] - pt1_real[0], pt2_real[1] - pt1_real[1])
            
            # Dibujar línea y texto gruesos
            cv.line(img_interactiva, puntos_medicion[0], puntos_medicion[1], (0, 255, 255), 8)
            mid_x = (puntos_medicion[0][0] + puntos_medicion[1][0]) // 2
            mid_y = (puntos_medicion[0][1] + puntos_medicion[1][1]) // 2
            cv.putText(img_interactiva, f"{distancia:.2f} cm", (mid_x, mid_y - 20), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 8)
            
            puntos_medicion = [] # Reset
            
        cv.imshow('Medicion Interactiva', img_interactiva)

# La clave está aquí: dejamos que OpenCV escale la ventana, para que el ratón mapee bien
cv.namedWindow('Medicion Interactiva', cv.WINDOW_NORMAL)
cv.resizeWindow('Medicion Interactiva', 1000, 800)
cv.setMouseCallback('Medicion Interactiva', click_raton)
cv.imshow('Medicion Interactiva', img_interactiva)

print("\n[INFO] Ventanas abiertas. Ve a 'Medicion Interactiva'.")
print("[INFO] Haz clic en dos puntos para medir la distancia real.")
print("[INFO] Pulsa 'g' para guardar la foto con las medidas, o 'ESC' para salir.")

while True:
    key = cv.waitKey(1) & 0xFF
    if key == 27: # ESC
        break
    elif key == ord('g'):
        cv.imwrite("medicion_resultado.png", img_interactiva)
        print("[OK] Captura guardada como medicion_resultado.png")

cv.destroyAllWindows()