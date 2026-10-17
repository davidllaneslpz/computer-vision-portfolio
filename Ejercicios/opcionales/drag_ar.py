#!/usr/bin/env python3
"""Opcional 7: Efecto AR Realista (Pseudo-3D Drag & Drop).

Simula objetos cilíndricos 3D con perspectiva y sombra dinámica.
La profundidad se controla por la posición Y en pantalla.

Controles:
  - Click IZQ: Arrastrar objeto o crear target
  - Click DER: Crear nuevo objeto
  - n: Nuevo objeto en el centro
  - s: Guardar captura
  - ESC: Salir
"""
import argparse
import cv2 as cv
import numpy as np

class VirtualObject:
    def __init__(self, oid, pos, color=(0, 180, 255)):
        self.id = oid
        self.pos_2d = np.array(pos, dtype=float)
        self.base_radius = 45
        self.color = color
        self.is_lifted = False

class DragARRealista:
    def __init__(self, src=0):
        self.cap = cv.VideoCapture(int(src) if str(src).isdigit() else src)
        if not self.cap.isOpened():
            raise RuntimeError(f'Cannot open source {src}')

        self.window = 'AR Realista'
        cv.namedWindow(self.window, cv.WINDOW_NORMAL)

        # Dimensiones para los cálculos de perspectiva
        self.W = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH) or 640)
        self.H = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT) or 480)  # Aspect para perspectiva

        self.objects = []
        self.targets = []
        self.dragging = None
        self.oid = 1

        cv.setMouseCallback(self.window, self.mouse_cb)

    def add_object(self, pos=None):
        if pos is None:
            pos = (self.W // 2, int(self.H * 0.8))
        obj = VirtualObject(self.oid, pos)
        self.oid += 1
        self.objects.append(obj)
        return obj

    def add_target(self, pos):
        self.targets.append(tuple(pos))

    def get_perspective_scale(self, y):
        """Scale factor based on Y position (depth). Larger at bottom, smaller at top."""
        scale = 0.4 + 0.6 * (y / self.H)
        return max(0.2, scale)

    def draw_realistic_object(self, frame, obj):
        """Draw 3D cylinder with perspective-based shadow."""
        cx, cy = int(obj.pos_2d[0]), int(obj.pos_2d[1])
        scale = self.get_perspective_scale(cy)
        
        rx = int(obj.base_radius * scale)
        ry = int(rx * 0.45)

        shadow_offset_y = int(15 * scale) if not obj.is_lifted else int(35 * scale)
        shadow_rx = rx if not obj.is_lifted else int(rx * 0.8)
        shadow_ry = ry if not obj.is_lifted else int(ry * 0.8)
        shadow_alpha = 0.6 if not obj.is_lifted else 0.3

        overlay = frame.copy()
        cv.ellipse(overlay, (cx, cy + shadow_offset_y), (shadow_rx, shadow_ry), 0, 0, 360, (20, 20, 20), -1)
        cv.addWeighted(overlay, shadow_alpha, frame, 1 - shadow_alpha, 0, frame)

        obj_overlay = frame.copy()
        h = int(10 * scale) if not obj.is_lifted else int(15 * scale)
        
        body_color = tuple(int(c * 0.8) for c in obj.color)
        cv.ellipse(obj_overlay, (cx, cy - h), (rx, ry), 0, 0, 180, body_color, -1)
        cv.rectangle(obj_overlay, (cx - rx, cy - h), (cx + rx, cy), body_color, -1)
        cv.ellipse(obj_overlay, (cx, cy - h), (rx, ry), 0, 0, 360, obj.color, -1)
        cv.ellipse(obj_overlay, (cx, cy - h), (rx, ry), 0, 0, 360, (255, 255, 255), max(1, int(2*scale)))

        font_scale = 0.7 * scale
        text = f'#{obj.id}'
        (tw, th), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv.putText(obj_overlay, text, (cx - tw//2, cy - h + th//2), cv.FONT_HERSHEY_SIMPLEX, 
                   font_scale, (255, 255, 255), max(1, int(2*scale)), cv.LINE_AA)

        cv.addWeighted(obj_overlay, 0.9, frame, 0.1, 0, frame)

    def mouse_cb(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            for obj in reversed(self.objects):
                scale = self.get_perspective_scale(obj.pos_2d[1])
                rx = obj.base_radius * scale
                ry = rx * 0.45
                
                dx = x - obj.pos_2d[0]
                dy = y - obj.pos_2d[1]
                if (dx**2 / rx**2) + (dy**2 / ry**2) <= 1:
                    obj.is_lifted = True
                    self.dragging = (obj, obj.pos_2d - np.array([x, y]))
                    self.objects.remove(obj)
                    self.objects.append(obj)
                    return
            self.add_target((x, y))

        elif event == cv.EVENT_MOUSEMOVE:
            if self.dragging is not None:
                obj, off = self.dragging
                obj.pos_2d = np.array([x, y]) + off

        elif event == cv.EVENT_LBUTTONUP:
            if self.dragging is not None:
                obj, off = self.dragging
                obj.is_lifted = False
                self.dragging = None
                
                if self.targets:
                    pts = np.array(self.targets)
                    diffs = pts - obj.pos_2d
                    dists = np.sqrt(diffs[:, 0]**2 + (diffs[:, 1]*2)**2) 
                    
                    idx = int(np.argmin(dists))
                    scale = self.get_perspective_scale(obj.pos_2d[1])
                    if dists[idx] < max(40, obj.base_radius * scale * 2):
                        obj.pos_2d = pts[idx].astype(float)

        elif event == cv.EVENT_RBUTTONDOWN:
            self.add_object((x, y))

    def draw(self, frame):
        for p in self.targets:
            x, y = int(p[0]), int(p[1])
            scale = self.get_perspective_scale(y)
            rx, ry = int(15 * scale), int(7 * scale)
            cv.ellipse(frame, (x, y), (rx, ry), 0, 0, 360, (0, 0, 200), max(1, int(2*scale)), cv.LINE_AA)
            cv.drawMarker(frame, (x, y), (0, 0, 255), cv.MARKER_CROSS, int(20*scale), max(1, int(2*scale)), cv.LINE_AA)

        for obj in self.objects:
            self.draw_realistic_object(frame, obj)

    def run(self):
        print('[INFO] Point camera at a table or floor. Objects appear larger at bottom (closer).')
        print('[INFO] Left-click: drag/target | Right-click: new object | n: new | s: save | ESC: exit')
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            
            frame = cv.flip(frame, 1)
            self.draw(frame)
            cv.imshow(self.window, frame)
            
            k = cv.waitKey(10) & 0xFF
            if k == 27 or k == ord('q'): break
            if k == ord('n'): self.add_object()
            if k == ord('s'):
                cv.imwrite('drag_ar_realista.jpg', frame)
                print('[OK] Saved to drag_ar_realista.jpg')

        self.cap.release()
        cv.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Realistic 3D AR drag-and-drop on table/floor')
    parser.add_argument('--dev', default='0', help='Video source (0=webcam or path)')
    args = parser.parse_args()
    ar = DragARRealista(args.dev)
    ar.add_object()
    ar.run()