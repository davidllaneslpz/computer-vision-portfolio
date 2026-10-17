from io import BytesIO

import cv2 as cv
import numpy as np
from PIL import Image

from umucv.util import putText


def send_image(bot, user_id, frame):
    """Send an OpenCV BGR frame to Telegram without creating a temporary file."""
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb, mode="RGB")
    image_buffer = BytesIO()
    image.save(image_buffer, "PNG")
    image_buffer.seek(0)

    try:
        bot.sendPhoto(
            chat_id=user_id,
            photo=image_buffer,
            caption="ALERTA: Intruso detectado en la zona vigilada!",
        )
        print("-> [TELEGRAM] Alerta enviada con exito al movil.")
    except Exception as error:
        print(f"-> [ERROR TELEGRAM]: {error}")


def detect_motion(frame, roi, background_subtractor, threshold=300):
    """Annotate the ROI and return the frame plus the movement flag."""
    annotated_frame = frame.copy()
    movement_detected = False

    if not roi:
        return annotated_frame, movement_detected

    x1, y1, x2, y2 = roi
    cv.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    putText(annotated_frame, "ZONA VIGILADA", orig=(x1, y1 - 8))

    roi_frame = frame[y1:y2 + 1, x1:x2 + 1]
    if roi_frame.size == 0:
        return annotated_frame, movement_detected

    foreground_mask = background_subtractor.apply(roi_frame)
    foreground_mask = cv.dilate(
        foreground_mask,
        np.ones((5, 5), np.uint8),
        iterations=1,
    )
    movement_detected = cv.countNonZero(foreground_mask) > threshold

    if movement_detected:
        cv.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return annotated_frame, movement_detected


def anonymize_people(frame, result, roi, confidence_threshold=0.50):
    """Blur detected people and report whether one overlaps the configured ROI."""
    person_in_roi = False
    roi_x1, roi_y1, roi_x2, roi_y2 = roi if roi else (None, None, None, None)

    for box in result.boxes:
        class_id = int(box.cls.cpu().numpy()[0])
        confidence = float(box.conf.cpu().numpy()[0])
        if class_id != 0 or confidence <= confidence_threshold:
            continue

        [[box_x1, box_y1, box_x2, box_y2]] = box.xyxy.cpu().numpy().astype(int)
        box_x1 = max(0, box_x1)
        box_y1 = max(0, box_y1)
        box_x2 = min(frame.shape[1], box_x2)
        box_y2 = min(frame.shape[0], box_y2)

        person_crop = frame[box_y1:box_y2, box_x1:box_x2]
        if person_crop.size > 0:
            frame[box_y1:box_y2, box_x1:box_x2] = cv.blur(person_crop, (40, 40))
            cv.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 165, 255), 3)

        if roi and not (
            box_x2 < roi_x1
            or box_x1 > roi_x2
            or box_y2 < roi_y1
            or box_y1 > roi_y2
        ):
            person_in_roi = True

    return person_in_roi


def draw_recording_indicator(frame, video):
    """Draw and persist the recording indicator for the current frame."""
    cv.circle(frame, (30, 30), 10, (0, 0, 255), -1)
    putText(frame, "REC", orig=(50, 35))
    video.write(frame)