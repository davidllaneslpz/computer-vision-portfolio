import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from umucv.stream import autoStream

# 1. MOG2 Base 
bgsub = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

# Contadores
coches_derecha = 0
coches_izquierda = 0
centros_previos = []

# Gráficas
historial_tiempo = []
historial_trafico_der = []
historial_trafico_izq = []
frame_count = 0
intervalo_grafica = 30 

for key, frame in autoStream():
    H, W = frame.shape[:2]
    frame_count += 1
    
    # 2. Extraer el movimiento
    fgmask = bgsub.apply(frame)
    
    # --- LA ESTRATEGIA DE LA FRANJA (ROI) ---
    # Tapamos de negro absoluto todo lo que NO sea carretera
    limite_superior = int(H * 0.35) # Del 35% hacia arriba (Cielo) será ignorado
    limite_inferior = int(H * 0.65) # Del 65% hacia abajo (Río) será ignorado
    
    fgmask[:limite_superior, :] = 0  # Borrar cielo
    fgmask[limite_inferior:, :] = 0  # Borrar río
    
    # 3. Limpieza de ruido 
    kernel = np.ones((3,3), np.uint8)
    fgmask = cv.erode(fgmask, kernel, iterations=1)
    fgmask = cv.medianBlur(fgmask, 3)
    fgmask = cv.dilate(fgmask, kernel, iterations=2) # Engordar manchas
    
    # 4. Encontrar coches
    contornos, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    centros_actuales = []
    
    for c in contornos:
        area = cv.contourArea(c)
        # Filtro muy básico
        if 300 < area < 6000: 
            x, y, bw, bh = cv.boundingRect(c)
            cx, cy = x + bw // 2, y + bh // 2
            centros_actuales.append((cx, cy))
            
            # Dibujar detecciones
            cv.rectangle(frame, (x, y), (x+bw, y+bh), (0, 255, 255), 2)
            cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    # 5. Lógica de conteo de peaje (Línea vertical)
    linea_x = W // 2
    
    for cx, cy in centros_actuales:
        for px, py in centros_previos:
            if abs(cx - px) < 100 and abs(cy - py) < 50: 
                # De izquierda a derecha
                if px < linea_x and cx >= linea_x:
                    coches_derecha += 1
                    cv.line(frame, (linea_x, limite_superior), (linea_x, limite_inferior), (0, 0, 255), 5) 
                # De derecha a izquierda
                elif px > linea_x and cx <= linea_x:
                    coches_izquierda += 1
                    cv.line(frame, (linea_x, limite_superior), (linea_x, limite_inferior), (0, 255, 0), 5) 

    centros_previos = centros_actuales.copy()

    # Guardar datos
    if frame_count % intervalo_grafica == 0:
        historial_tiempo.append(frame_count)
        historial_trafico_der.append(coches_derecha)
        historial_trafico_izq.append(coches_izquierda)

    # --- DIBUJOS EN PANTALLA ---
    cv.line(frame, (0, limite_superior), (W, limite_superior), (0, 255, 255), 1)
    cv.line(frame, (0, limite_inferior), (W, limite_inferior), (0, 255, 255), 1)
    # Línea de conteo vertical
    cv.line(frame, (linea_x, limite_superior), (linea_x, limite_inferior), (255, 0, 0), 2) 
    
    cv.putText(frame, f"Hacia Dcha: {coches_derecha}", (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.putText(frame, f"Hacia Izq: {coches_izquierda}", (10, 80), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv.imshow('Trafico (ESC para Grafica)', frame)
    cv.imshow('Mascara Estricta', fgmask)

cv.destroyAllWindows()

plt.figure(figsize=(10, 5))
plt.plot(historial_tiempo, historial_trafico_der, label='Hacia Derecha', color='red', marker='o')
plt.plot(historial_tiempo, historial_trafico_izq, label='Hacia Izquierda', color='green', marker='o')
plt.title('Análisis de Tráfico por Unidad de Tiempo')
plt.xlabel('Unidad de tiempo (Fotogramas)')
plt.ylabel('Vehículos Acumulados')
plt.legend()
plt.grid(True)
plt.show()