#!/usr/bin/env python3
"""
Comparativa: Filtro de media propio vs. OpenCV

Uso:
    python filtro_media_vs_opencv.py --img imagen.jpg

- Aplica un filtro de media 3x3 implementado a mano.
- Aplica el filtro de media de OpenCV (cv.blur).
- Muestra ambas imágenes y la diferencia.
- Imprime el tiempo de ejecución de cada método.
"""
import cv2 as cv
import numpy as np
import argparse
import time

parser = argparse.ArgumentParser(description="Filtro de media propio vs OpenCV")
parser.add_argument('--img', type=str, required=True, help='Imagen de entrada')
args = parser.parse_args()

img = cv.imread(args.img)
if img is None:
    print(f"No se pudo abrir la imagen {args.img}")
    exit(1)

# Convertir a escala de grises para simplificar
if len(img.shape) == 3:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
else:
    gray = img.copy()

def filtro_media_manual(im):
    """Filtro de media manual 3x3."""
    h, w = im.shape
    out = np.zeros_like(im, dtype=np.uint8)
    for y in range(1, h-1):
        for x in range(1, w-1):
            region = im[y-1:y+2, x-1:x+2]
            out[y, x] = np.mean(region, dtype=np.float32)
    return out

start = time.time()
manual = filtro_media_manual(gray)
t_manual = time.time() - start

start = time.time()
opencv = cv.blur(gray, (3,3))
t_opencv = time.time() - start

diff = cv.absdiff(manual, opencv)
cv.imshow('Original', gray)
cv.imshow('Filtro Media Manual', manual)
cv.imshow('Filtro Media OpenCV', opencv)
cv.imshow('Diferencia Absoluta', diff)
print(f"Tiempo filtro manual: {t_manual*1000:.2f} ms")
print(f"Tiempo OpenCV: {t_opencv*1000:.2f} ms")
print(f"Diferencia media por píxel: {np.mean(diff):.4f}")
print("Pulsa cualquier tecla para cerrar las ventanas.")
cv.waitKey(0)
cv.destroyAllWindows()
