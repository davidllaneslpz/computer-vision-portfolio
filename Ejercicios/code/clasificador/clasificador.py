#!/usr/bin/env python

import cv2 as cv
import os
import glob
import importlib
import argparse
from umucv.stream import autoStream
from umucv.util import putText

# ==========================================
# 1. PARSEO DE ARGUMENTOS (--models, --method)
# ==========================================
parser = argparse.ArgumentParser(description='Clasificador de Imagenes Modular')
parser.add_argument('--models', type=str, required=True, help='Carpeta con las imagenes modelo')
parser.add_argument('--method', type=str, required=True, help='Metodo a usar: sift, embedder, gestos')
args = parser.parse_args()

# ==========================================
# 2. CARGA DINÁMICA DEL MÉTODO
# ==========================================
print(f"[*] Iniciando sistema con el método: '{args.method}'")
try:
    modulo = importlib.import_module(f"metodos.{args.method}")
except Exception as e:
    print(f"[!] ERROR interno cargando el módulo '{args.method}': {e}")
    exit()

# ==========================================
# 3. PRECOMPUTACIÓN DE MODELOS
# ==========================================
modelos_db = {} # Diccionario para guardar Nombre -> Caracteristicas
archivos = glob.glob(os.path.join(args.models, "*.*"))

print(f"[*] Precomputando {len(archivos)} modelos desde '{args.models}'...")
for ruta in archivos:
    img = cv.imread(ruta)
    if img is not None:
        nombre_base = os.path.basename(ruta).split('.')[0]
        # Llamamos a la funcion precomputar() del metodo que hayamos elegido
        caracteristicas = modulo.precomputar(img)
        
        if caracteristicas is not None:
            modelos_db[nombre_base] = caracteristicas
            print(f"  -> {nombre_base} OK")
        else:
            print(f"  -> {nombre_base} FALLO (No se encontraron caracteristicas)")

if not modelos_db:
    print("[!] AVISO: No hay modelos precomputados. Usa la tecla 'g' para guardar modelos en vivo.")

print("\n--- SISTEMA LISTO ---")
print("Pulsa 'g' para añadir lo que ve la cámara como un nuevo modelo.")
print("Pulsa ESC para salir.")

# ==========================================
# 4. BUCLE DE VÍDEO EN TIEMPO REAL
# ==========================================
for key, frame in autoStream():
    
    resultados = []
    
    # Comparamos el frame actual contra todos los modelos en la base de datos
    for nombre, caracteristicas in modelos_db.items():
        # Llamamos a la funcion comparar() del metodo elegido
        confianza = modulo.comparar(frame, caracteristicas)
        resultados.append((confianza, nombre))
    
    # Ordenamos de mayor confianza a menor
    resultados.sort(reverse=True)
    
    # ==========================================
    # 5. MOSTRAR RESULTADOS EN PANTALLA
    # ==========================================
    frame_display = frame.copy()
    y_offset = 30
    
    if resultados:
        mejor_confianza, mejor_nombre = resultados[0]
        
        # Filtrar un poco para que no muestre basura si no hay nada parecido
        if mejor_confianza > 0.3: # Ajusta este umbral si es necesario
            putText(frame_display, f"RECONOCIDO: {mejor_nombre}", (20, y_offset), color=(0,255,0), scale=1)
            y_offset += 30
            putText(frame_display, f"Confianza: {mejor_confianza:.2f}", (20, y_offset), color=(0,255,0), scale=0.7)
        else:
            putText(frame_display, "No se reconoce objeto", (20, y_offset), color=(0,0,255), scale=1)
            
        y_offset += 40
        # Mostrar las distancias/confianzas del resto de modelos
        putText(frame_display, "Top 3 similitudes:", (20, y_offset), scale=0.6)
        y_offset += 20
        for conf, nom in resultados[:3]:
            putText(frame_display, f"- {nom}: {conf:.2f}", (20, y_offset), scale=0.6)
            y_offset += 20

    # ==========================================
    # 6. AÑADIR MODELOS EN CALIENTE (ON THE FLY)
    # ==========================================
    if key == ord('g'):
        nuevo_nombre = f"modelo_nuevo_{len(modelos_db) + 1}"
        caracteristicas_nuevas = modulo.precomputar(frame)
        
        if caracteristicas_nuevas is not None:
            modelos_db[nuevo_nombre] = caracteristicas_nuevas
            # Guardamos la foto fisicamente en la carpeta
            if not os.path.exists(args.models):
                os.makedirs(args.models)
            ruta_guardado = os.path.join(args.models, f"{nuevo_nombre}.jpg")
            cv.imwrite(ruta_guardado, frame)
            print(f"[*] Nuevo modelo '{nuevo_nombre}' guardado y precomputado con exito.")

    cv.imshow("Clasificador Modular", frame_display)

cv.destroyAllWindows()