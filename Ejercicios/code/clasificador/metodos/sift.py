import cv2 as cv

# Inicializamos SIFT una sola vez
sift = cv.SIFT_create()
# Usamos FLANN (Fast Library for Approximate Nearest Neighbors) que es más rápido que BFMatcher
index_params = dict(algorithm=1, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)

def precomputar(imagen):
    """Extrae los puntos clave y descriptores SIFT de una imagen modelo"""
    gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gris, None)
    return (kp, des)

def comparar(frame, caracteristicas_modelo):
    """Compara el frame actual con los descriptores del modelo. 
    Devuelve la confianza normalizada (de 0.0 a 1.0)"""
    
    _, des_modelo = caracteristicas_modelo
    if des_modelo is None or len(des_modelo) == 0:
        return 0.0

    gris_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    kp_frame, des_frame = sift.detectAndCompute(gris_frame, None)
    
    if des_frame is None or len(des_frame) < 2:
        return 0.0

    # KNN Match
    matches = flann.knnMatch(des_modelo, des_frame, k=2)
    
    # Filtro de Lowe (Ratio test)
    buenos_matches = 0
    for match in matches:
        if len(match) == 2:
            m, n = match
            if m.distance < 0.7 * n.distance:
                buenos_matches += 1
                
    # Calculamos qué porcentaje de los puntos del modelo original hemos encontrado
    confianza = buenos_matches / len(des_modelo)
    
    # Nos aseguramos de que no pase de 1.0 por si hay falsos positivos extra
    return min(confianza, 1.0)