"""Configuration module for SiteSafety-YOLO HSE monitoring system.

This module centralizes all hyperparameters, model configuration, and file
system paths required for object detection, video processing, and visualization.
Using a centralized configuration approach ensures consistency across all
components of the system and simplifies deployment across different environments.
"""

from pathlib import Path
from typing import Tuple

# File System Paths
# Using pathlib for cross-platform compatibility (Windows/Linux/Docker)
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / 'data' / 'input'
OUTPUT_DIR = BASE_DIR / 'data' / 'output'

# Model Configuration
# YOLOv8n (Nano) selected for Edge AI deployment readiness.
# While larger models offer higher accuracy, Nano allows for >30 FPS
# on constrained hardware (Jetson/Raspberry Pi), critical for real-time safety.
MODEL_WEIGHTS = 'yolov8n.pt'

# Confidence Threshold: 0.45
# Selected to balance Recall (detecting all workers) vs Precision (reducing false alarms).
# In safety contexts, missing a worker (False Negative) is worse than a false alarm,
# but 0.45 filters out most background noise in complex industrial scenes.
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45

# COCO Class ID for 'Person'. 
# In a production HSE model, we would fine-tune for 'Hardhat'/'Vest'.
CLASS_ID_PERSON = 0

# Visualization Colors (BGR)
# Green for Safe/Compliant, Red for Warning/Violation
COLOR_SAFE: Tuple[int, int, int] = (0, 255, 0)
COLOR_WARNING: Tuple[int, int, int] = (0, 0, 255)
FONT = 0