# Computer Vision Portfolio

A Python computer vision portfolio covering classical image processing, camera geometry, real-time video analysis, deep learning, and interactive vision systems.

The project combines OpenCV, YOLO11, MediaPipe, PyTorch, TensorFlow/Keras, and UMU computer-vision utilities across a collection of practical experiments and reusable pipelines.

## Highlights

- Camera calibration, undistortion, triangulation, and planar rectification
- Motion detection, background subtraction, optical flow, and object tracking
- Real-time object detection, tracking, and segmentation with YOLO11
- Hand, face, and human-pose analysis with MediaPipe
- Image classification with SIFT, embeddings, CNNs, CLIP, and DINO
- UNET segmentation and custom YOLO dataset labelling
- Activity monitoring with ROI filtering, person anonymization, video recording, and Telegram alerts

## Repository Structure

```text
.
├── Ejercicios/
│   ├── code/
│   │   ├── bot/          # Activity monitoring and Telegram alerts
│   │   ├── calibrate/    # Camera calibration and 3D measurement
│   │   ├── clasificador/ # SIFT, embeddings, and gesture classification
│   │   ├── color/        # Colour segmentation and traffic analysis
│   │   ├── DL/           # CNN, YOLO, MediaPipe, DINO, CLIP, and UNET
│   │   ├── LK/           # Lucas-Kanade tracking and optical flow
│   │   ├── rectifica/    # Perspective correction and measurement
│   │   ├── SIFT/         # Local feature experiments
│   │   └── util/         # Reusable course utilities
│   ├── obligatorios/     # Assignment documentation and requirements
│   └── opcionales/       # Extended experiments
├── requirements.txt
└── .gitignore
```

The detailed assignment documentation is available in [Ejercicios/obligatorios/README.md](Ejercicios/obligatorios/README.md).

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/davidllaneslpz/computer-vision-portfolio.git
cd computer-vision-portfolio
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Linux or macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

Most real-time examples require a webcam or a video source. Some deep-learning examples also require downloading the corresponding model assets before execution.

## Selected Examples

### YOLO11 object detection

```bash
python Ejercicios/code/DL/yolo/yolo.py
```

### MediaPipe hands

```bash
python Ejercicios/code/DL/mp_hands/hands.py
```

### Activity monitoring

Configure `TOKEN` and `USER_ID` in a local `Ejercicios/code/bot/token.env` file, then run:

```bash
python Ejercicios/code/bot/main.py
```

The activity pipeline detects movement inside a manually selected ROI, runs YOLO only when needed, anonymizes detected people, records a short clip, and sends Telegram alerts when an authorized event is detected.

## Data, Models, and Secrets

Large model files, generated media, datasets, logs, Python caches, and environment files are intentionally excluded from Git. Download model assets locally or use an external model registry. Never commit Telegram tokens, API keys, or other credentials.

## Documentation

- [Mandatory exercises](Ejercicios/obligatorios/README.md)
- [Dependency list](requirements.txt)
- [Repository ignore policy](.gitignore)

## License

This repository is an academic and professional portfolio project. Add a project-specific license before distributing the code as a reusable library.
