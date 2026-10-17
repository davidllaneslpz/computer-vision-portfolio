#!/usr/bin/env python3
"""
Extensión del tracker Lucas-Kanade para estimar movimiento de cámara
(UP, DOWN, LEFT, RIGHT, FORWARD, BACKWARD) y velocidad angular.
"""
import argparse
import time
from collections import deque
import cv2 as cv
import numpy as np

parser = argparse.ArgumentParser(description='Lucas-Kanade camera motion estimation')
parser.add_argument('--camera', type=str, default='0', help='Fuente de vídeo')
parser.add_argument('--min-points', type=int, default=20, help='Mínimo de puntos')
parser.add_argument('--detect-interval', type=int, default=5, help='Intervalo de frames')
parser.add_argument('--warp-threshold', type=float, default=1.0, help='Umbral de traslación')
args = parser.parse_args()

tracks = []
track_len = 20

corners_params = dict(maxCorners=500, qualityLevel=0.01, minDistance=10, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

last_time = time.time()

def direction_from_translation(tx, ty, threshold=1.0):
    if abs(tx) > abs(ty):
        return 'LEFT' if tx > threshold else 'RIGHT' if tx < -threshold else None
    else:
        return 'UP' if ty > threshold else 'DOWN' if ty < -threshold else None

cap = cv.VideoCapture(int(args.camera) if args.camera.isdigit() else args.camera)
if not cap.isOpened():
    raise RuntimeError(f'No se pudo abrir la cámara: {args.camera}')

n = 0
prevgray = None

print("[INFO] Lucas-Kanade Tracker iniciado.")
print("[INFO] Mueve la camara para ver las direcciones. Pulsa 'g' para capturar o ESC para salir.")

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if n == 0:
        prevgray = gray
        n += 1
        continue

    if tracks:
        p0 = np.float32([t[-1] for t in tracks])
        p1, st, err = cv.calcOpticalFlowPyrLK(prevgray, gray, p0, None, **lk_params)
        p0r, st2, err2 = cv.calcOpticalFlowPyrLK(gray, prevgray, p1, None, **lk_params)
        d = abs(p0 - p0r).reshape(-1, 2).max(axis=1)
        good = d < 1

        new_tracks = []
        for t, (x, y), ok in zip(tracks, p1.reshape(-1, 2), good):
            if not ok: continue
            t.append((x, y))
            new_tracks.append(t)
        tracks = new_tracks

        if len(tracks) >= args.min_points:
            src_pts = np.float32([t[-2] for t in tracks]).reshape(-1, 1, 2)
            dst_pts = np.float32([t[-1] for t in tracks]).reshape(-1, 1, 2)
            
            M, inliers = cv.estimateAffinePartial2D(src_pts, dst_pts, method=cv.RANSAC, ransacReprojThreshold=3)
            
            if M is not None:
                tx, ty = M[0, 2], M[1, 2]
                angle = np.degrees(np.arctan2(M[1, 0], M[0, 0])) 
                scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
                
                now = time.time()
                dt = max(now - last_time, 1e-3)
                last_time = now
                angular_speed = abs(angle) / dt

                move_dir = direction_from_translation(tx, ty, args.warp_threshold)
                zoom_dir = 'FORWARD' if scale > 1.02 else 'BACKWARD' if scale < 0.98 else ''
                parts = [d for d in (move_dir, zoom_dir) if d]
                camera_motion = ' + '.join(parts) if parts else 'STABLE'

                cv.putText(frame, f'Motion: {camera_motion}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv.putText(frame, f'w Speed: {angular_speed:.1f} deg/s', (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv.putText(frame, f'Zoom Scale: {scale:.3f}', (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv.putText(frame, 'Motion: RANSAC Fail', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv.polylines(frame, [np.int32(t) for t in tracks], isClosed=False, color=(0, 0, 255), thickness=2)
        for t in tracks:
            cv.circle(frame, tuple(np.int32(t[-1])), 3, (0, 255, 255), -1)

    if n % args.detect_interval == 0:
        mask = np.zeros_like(gray)
        mask[:] = 255
        for x, y in [np.int32(t[-1]) for t in tracks]:
            cv.circle(mask, (x, y), 5, 0, -1)
        corners = cv.goodFeaturesToTrack(gray, mask=mask, **corners_params)
        if corners is not None:
            for [pt] in np.float32(corners):
                tracks.append(deque([pt], maxlen=track_len))

    cv.putText(frame, f'Active Tracks: {len(tracks)}', (10, frame.shape[0] - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv.imshow('LK Camera Motion', frame)
    prevgray = gray

    key = cv.waitKey(1) & 0xFF
    if key == 27: # ESC
        break
    elif key == ord('g'):
        cv.imwrite("lk_tracker.png", frame)
        print("[OK] Captura guardada como lk_tracker.png")
    
    n += 1

cap.release()
cv.destroyAllWindows()