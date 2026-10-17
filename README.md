# Computer Vision Experiments

Python implementations of classical computer vision, camera geometry, real-time video processing, and deep-learning inference.

The code uses OpenCV, YOLO11, MediaPipe, PyTorch, TensorFlow/Keras, and UMU computer-vision utilities. Each directory contains focused scripts for a specific technique or processing pipeline.

## Contents

- Camera calibration, undistortion, triangulation, and planar rectification
- Motion detection, background subtraction, optical flow, and object tracking
- Object detection, tracking, and segmentation with YOLO11
- Hand, face, and human-pose analysis with MediaPipe
- Image classification with SIFT, embeddings, CNNs, CLIP, and DINO
- UNET segmentation and custom YOLO dataset labelling
- ROI-based activity monitoring with person anonymization, video recording, and Telegram alerts

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

Detailed descriptions and commands for the mandatory exercises are available in [Ejercicios/obligatorios/README.md](Ejercicios/obligatorios/README.md).

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

Most real-time examples require a webcam or a video source. Deep-learning examples may also require downloading the model asset referenced by the script.

## Examples

### YOLO11 object detection

```bash
python Ejercicios/code/DL/yolo/yolo.py
```

### MediaPipe hands

```bash
python Ejercicios/code/DL/mp_hands/hands.py
```

### Activity monitoring

Create a local `Ejercicios/code/bot/token.env` file containing `TOKEN` and `USER_ID`, then run:

```bash
python Ejercicios/code/bot/main.py
```

This pipeline detects movement inside a manually selected ROI, runs YOLO when movement is present, anonymizes detected people, records a short clip, and sends Telegram alerts when a person overlaps the monitored region.

## Data, Models, and Secrets

Model weights, generated media, datasets, logs, Python caches, and environment files are excluded from Git. Download model assets locally or use an external model registry. Never commit Telegram tokens, API keys, or other credentials.

## Documentation

- [Mandatory exercises](Ejercicios/obligatorios/README.md)
- [Dependency list](requirements.txt)
- [Repository ignore policy](.gitignore)

## License

No license has been selected for this codebase yet. Add a project-specific license before distributing the code as a reusable library.
