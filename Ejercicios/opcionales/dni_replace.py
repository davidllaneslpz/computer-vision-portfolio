#!/usr/bin/env python3
"""Sustitución de fotografía en Carnet UMU en vivo.
Mejora la robustez de detección y utiliza un detector de caras sobre
la tarjeta rectificada para ubicar con precisión la zona de la foto.
"""

import argparse
import os
import sys
import cv2 as cv
import numpy as np

# Coordenadas relativas de la foto en el carnet [0,1]
PHOTO_RECT = np.array([[0.08, 0.65], [0.48, 0.65], [0.48, 0.95], [0.08, 0.95]], dtype=np.float32)


def order_quad(pts):
    """Order quad points in counter-clockwise starting from top-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    sorted_pts = pts[np.argsort(angles)]
    top_left_idx = np.argmin(sorted_pts[:, 1] + sorted_pts[:, 0] * 1e-3)
    ordered = np.roll(sorted_pts, -top_left_idx, axis=0)
    if np.cross(ordered[1] - ordered[0], ordered[2] - ordered[1]) < 0:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered


def load_replacement_photo(photo_path):
    if photo_path is None:
        # placeholder neutral photo
        img = np.full((420, 320, 3), (240, 240, 240), dtype=np.uint8)
        cv.putText(img, 'PHOTO', (30, 220), cv.FONT_HERSHEY_SIMPLEX, 1.2, (60, 60, 60), 2)
        return img
    if not os.path.exists(photo_path):
        raise FileNotFoundError(photo_path)
    img = cv.imread(photo_path, cv.IMREAD_COLOR)
    if img is None:
        raise ValueError('Cannot read replacement photo')
    return img


def find_card_quad(frame):
    """Detect card boundary using multiple preprocessing strategies."""
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    height, width = frame.shape[:2]
    methods = []
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    methods.append(otsu)
    adapt = cv.adaptiveThreshold(blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 51, -10)
    methods.append(adapt)
    edges = cv.Canny(blur, 50, 150)
    edges = cv.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    methods.append(edges)

    min_area = width * height * 0.02
    best_quad = None
    best_score = 0

    for m in methods:
        proc = cv.morphologyEx(m, cv.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        contours, _ = cv.findContours(proc, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = abs(cv.contourArea(cnt))
            if area < min_area:
                continue
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) != 4:
                continue
            if not cv.isContourConvex(approx):
                continue

            quad = approx.reshape(4, 2).astype(np.float32)
            w1 = np.linalg.norm(quad[0] - quad[1])
            w2 = np.linalg.norm(quad[2] - quad[3])
            h1 = np.linalg.norm(quad[0] - quad[3])
            h2 = np.linalg.norm(quad[1] - quad[2])
            w = (w1 + w2) / 2
            h = (h1 + h2) / 2
            if h <= 0 or w <= 0:
                continue
            aspect = w / h

            aspect_pref = 1.6
            aspect_pen = 1.0 / (1.0 + abs(aspect - aspect_pref))
            score = area * aspect_pen

            if score > best_score:
                best_score = score
                best_quad = quad

    return best_quad


def warp_photo_onto_frame(frame, photo, card_quad):
    """Warp photo onto card, with face-only overlay if available."""
    card_quad = order_quad(card_quad)

    TARGET_W = 600
    aspect_pref = 1.6
    TARGET_H = int(TARGET_W / aspect_pref)
    dst_rect = np.array([[0, 0], [TARGET_W - 1, 0], [TARGET_W - 1, TARGET_H - 1], [0, TARGET_H - 1]], dtype=np.float32)

    H_img2rect = cv.getPerspectiveTransform(card_quad.astype(np.float32), dst_rect)
    rectified = cv.warpPerspective(frame, H_img2rect, (TARGET_W, TARGET_H))

    # Detect face regions on both card and replacement
    face_bbox = None
    try:
        cascade_path = cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv.CascadeClassifier(cascade_path)
        gray = cv.cvtColor(rectified, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        if len(faces) > 0:
            faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
            face_bbox = faces[0]
    except Exception:
        face_bbox = None

    repl_face = None
    try:
        cascade_path = cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
        repl_cascade = cv.CascadeClassifier(cascade_path)
        gray_repl = cv.cvtColor(photo, cv.COLOR_BGR2GRAY)
        faces_repl = repl_cascade.detectMultiScale(gray_repl, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
        if len(faces_repl) > 0:
            faces_repl = sorted(faces_repl, key=lambda r: r[2] * r[3], reverse=True)
            repl_face = faces_repl[0]
    except Exception:
        repl_face = None

    H_rect2img = cv.getPerspectiveTransform(dst_rect, card_quad.astype(np.float32))

    if face_bbox is not None:
        x, y, fw, fh = face_bbox
        M = 0.25
        nx = max(0, int(x - fw * M))
        ny = max(0, int(y - fh * M))
        nw = min(TARGET_W - 1 - nx, int(fw * (1 + 2 * M)))
        nh = min(TARGET_H - 1 - ny, int(fh * (1 + 2 * M)))

        bbox_rect = np.array([[nx, ny], [nx + nw - 1, ny], [nx + nw - 1, ny + nh - 1], [nx, ny + nh - 1]], dtype=np.float32)

        # If replacement contains face, warp only face region into card area
        if repl_face is not None:
            rx, ry, rfw, rfh = repl_face
            RM = 0.3
            rnx = max(0, int(rx - rfw * RM))
            rny = max(0, int(ry - rfh * RM))
            rnw = min(photo.shape[1] - 1 - rnx, int(rfw * (1 + 2 * RM)))
            rnh = min(photo.shape[0] - 1 - rny, int(rfh * (1 + 2 * RM)))

            bbox_src = np.array([[rnx, rny], [rnx + rnw - 1, rny], [rnx + rnw - 1, rny + rnh - 1], [rnx, rny + rnh - 1]], dtype=np.float32)

            overlay_rect = np.zeros_like(rectified)
            H_src2rect = cv.getPerspectiveTransform(bbox_src, bbox_rect)
            cv.warpPerspective(photo, H_src2rect, (TARGET_W, TARGET_H), overlay_rect, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_TRANSPARENT)

            mask_rect = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
            cv.fillConvexPoly(mask_rect, bbox_rect.astype(int), 255)
            alpha = cv.GaussianBlur(mask_rect.astype(np.float32) / 255.0, (31, 31), 0)
            alpha_3 = np.stack([alpha] * 3, axis=2)

            rectified = (rectified.astype(np.float32) * (1 - alpha_3) + overlay_rect.astype(np.float32) * alpha_3).astype(np.uint8)

            overlay_frame = cv.warpPerspective(rectified, H_rect2img, (frame.shape[1], frame.shape[0]), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_TRANSPARENT)
            mask_frame = cv.warpPerspective((mask_rect).astype(np.uint8), H_rect2img, (frame.shape[1], frame.shape[0]))
            mask_bool = mask_frame[:, :] > 0

            output = frame.copy()
            alpha_frame = cv.warpPerspective(alpha, H_rect2img, (frame.shape[1], frame.shape[0]))
            alpha_frame_3 = np.stack([alpha_frame] * 3, axis=2)
            output = (output.astype(np.float32) * (1 - alpha_frame_3) + overlay_frame.astype(np.float32) * alpha_frame_3).astype(np.uint8)

            photo_dst = cv.perspectiveTransform(bbox_rect.reshape(-1, 1, 2), H_rect2img).reshape(-1, 2)
            return output, photo_dst
        else:
            # No face in replacement: use fixed card area
            photo_norm = PHOTO_RECT.reshape(-1, 1, 2)
            card_source = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
            H_card = cv.getPerspectiveTransform(card_source, card_quad.astype(np.float32))
            photo_dst = cv.perspectiveTransform(photo_norm, H_card).reshape(-1, 2)
    else:
        # No face detected on card: use fixed area
        photo_norm = PHOTO_RECT.reshape(-1, 1, 2)
        card_source = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        H_card = cv.getPerspectiveTransform(card_source, card_quad.astype(np.float32))
        photo_dst = cv.perspectiveTransform(photo_norm, H_card).reshape(-1, 2)

    ph, pw = photo.shape[:2]
    photo_src = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], dtype=np.float32)
    H_photo = cv.getPerspectiveTransform(photo_src, photo_dst.astype(np.float32))

    overlay = np.zeros_like(frame)
    cv.warpPerspective(photo, H_photo, (frame.shape[1], frame.shape[0]), overlay, flags=cv.INTER_LINEAR, borderMode=cv.BORDER_TRANSPARENT)

    mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    cv.fillConvexPoly(mask, photo_dst.astype(int), 255)

    output = frame.copy()
    mask_bool = mask.astype(bool)
    output[mask_bool] = overlay[mask_bool]
    return output, photo_dst


def main():
    parser = argparse.ArgumentParser(description='DNI Photo Replace')
    parser.add_argument('--dev', type=str, default='0', help='Video source (0 for webcam or path to video)')
    parser.add_argument('--photo', type=str, default=None, help='Ruta a tu foto de reemplazo')
    parser.add_argument('--debug', action='store_true', help='Mostrar imagen rectificada y máscara')
    args = parser.parse_args()

    photo = load_replacement_photo(args.photo)

    dev = args.dev
    try:
        dev_i = int(dev)
        cap = cv.VideoCapture(dev_i)
    except Exception:
        cap = cv.VideoCapture(dev)

    if not cap.isOpened():
        print('Cannot open source', dev)
        return

    print("[INFO] Buscando DNI UMU... Pulsa 'g' para guardar captura, ESC para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        card_quad = find_card_quad(frame)
        if card_quad is not None:
            try:
                out, pdst = warp_photo_onto_frame(frame, photo, card_quad)
                frame = out
                cv.polylines(frame, [order_quad(card_quad).astype(int)], True, (0, 255, 0), 2, cv.LINE_AA)
                cv.polylines(frame, [pdst.astype(int)], True, (255, 0, 0), 2, cv.LINE_AA)
                cv.putText(frame, 'UMU ID photo replaced', (5, 16), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            except Exception as e:
                cv.putText(frame, 'Error al sustituir foto', (5, 16), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                print('warp error', e)
        else:
            cv.putText(frame, 'UMU ID card not found', (5, 16), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv.imshow('UMU ID Replacement', frame)

        if args.debug and card_quad is not None:
            # show rectified view for debugging
            H_img2rect = cv.getPerspectiveTransform(order_quad(card_quad).astype(np.float32), np.array([[0, 0], [599, 0], [599, 374], [0, 374]], dtype=np.float32))
            rect = cv.warpPerspective(frame, H_img2rect, (600, 375))
            cv.imshow('rectified', rect)

        k = cv.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break
        if k == ord('g'):
            fname = 'dni_replace_capture.png'
            cv.imwrite(fname, frame)
            print('Saved', fname)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
