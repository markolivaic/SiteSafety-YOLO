"""Configuration module for SiteSafety-YOLO HSE monitoring system.

This module centralizes all hyperparameters, model configuration, and file
system paths required for object detection, video processing, and visualization.
Using a centralized configuration approach ensures consistency across all
components of the system and simplifies deployment across environments.
"""

from pathlib import Path
from typing import Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / 'data' / 'input'
OUTPUT_DIR = BASE_DIR / 'data' / 'output'

MODEL_WEIGHTS = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45

CLASS_ID_PERSON = 0

COLOR_SAFE: Tuple[int, int, int] = (0, 255, 0)
COLOR_WARNING: Tuple[int, int, int] = (0, 0, 255)
FONT = 0
