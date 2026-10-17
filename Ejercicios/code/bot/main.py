#!/usr/bin/env python

import os
from pathlib import Path

import cv2 as cv
from dotenv import load_dotenv
from telegram.ext import Updater
from ultralytics import YOLO
from umucv.stream import autoStream
from umucv.util import ROI, Video

from utils import anonymize_people, detect_motion, draw_recording_indicator, send_image


def main():
    load_dotenv(Path(__file__).with_name("token.env"))
    token = os.environ.get("TOKEN")
    user_id = os.environ.get("USER_ID")
    if not token or not user_id:
        raise RuntimeError("Faltan TOKEN y USER_ID en token.env")

    print("Cargando modelo neuronal YOLO11...")
    model = YOLO(Path(__file__).with_name("yolo11n.pt"))
    background_subtractor = cv.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=False,
    )

    bot = Updater(token).bot
    cv.namedWindow("CCTV")
    region = ROI("CCTV")
    video = Video(fps=20, codec="MJPG", ext="avi")

    recording = False
    frames_to_record = 0
    cooldown = 0

    print("\n--- SISTEMA DE ALARMA ACTIVADO ---")
    print("Dibuja un ROI en CCTV. Pulsa ESC para salir.")

    try:
        for _, frame in autoStream():
            cooldown = max(0, cooldown - 1)
            processed_frame, movement_detected = detect_motion(
                frame,
                region.roi,
                background_subtractor,
            )

            if movement_detected or recording:
                [result] = model(frame, verbose=False)
                person_in_roi = anonymize_people(processed_frame, result, region.roi)

                if movement_detected and person_in_roi and not recording and cooldown == 0:
                    print("\n[!] ALERTA CRITICA: Intruso en el perimetro.")
                    send_image(bot, user_id, processed_frame)
                    recording = True
                    frames_to_record = 60
                    cooldown = 150

            if recording:
                draw_recording_indicator(processed_frame, video)
                frames_to_record -= 1
                if frames_to_record <= 0:
                    recording = False
                    print("-> Grabacion del evento finalizada.")

            cv.imshow("CCTV", processed_frame)
    finally:
        video.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()