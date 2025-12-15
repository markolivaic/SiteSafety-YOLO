"""Object detection module using YOLOv8n for real-time edge inference.

This module handles YOLOv8n model initialization with CUDA acceleration support
and provides inference capabilities for person detection in video frames. The
detector is optimized for edge devices by using the nano model variant (YOLOv8n)
which balances speed and accuracy for real-time HSE compliance monitoring.
"""

import logging

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results


class ObjectDetector:
    """Handles YOLOv8n model initialization and frame inference.

    The ObjectDetector class encapsulates YOLOv8n model loading and provides
    inference methods optimized for edge deployment. It automatically selects
    CUDA acceleration when available, falling back to CPU for compatibility.
    The detector is configured to filter only person class detections (class
    ID 0) to optimize inference speed and reduce false positives in industrial
    safety monitoring scenarios.
    """

    def __init__(self, model_path: str, confidence_threshold: float) -> None:
        """Initialize ObjectDetector with YOLOv8n model.

        Loads the YOLOv8n model weights and configures device selection
        (CUDA or CPU) based on hardware availability. CUDA acceleration is
        critical for real-time inference on edge devices, providing 5-10x
        speedup compared to CPU-only execution. The confidence threshold
        filters low-confidence detections to reduce false positives in
        industrial environments with varying lighting and occlusion conditions.

        Args:
            model_path (str): Path to YOLOv8n model weights file (.pt format).
            confidence_threshold (float): Minimum confidence score (0.0-1.0)
                for detections to be considered valid. Lower values increase
                recall but may introduce false positives.

        Raises:
            Exception: If model file cannot be loaded or is invalid. The
                exception is logged and re-raised to allow calling code to
                handle initialization failures gracefully.
        """
        # Device selection: CUDA provides 5-10x inference speedup for
        # real-time edge deployment. Automatic fallback to CPU ensures
        # compatibility with systems without GPU acceleration.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.confidence_threshold = confidence_threshold
        self.logger = logging.getLogger(__name__)

        try:
            self.model = YOLO(model_path)
            self.logger.info(
                f"Model loaded successfully on {self.device.upper()}"
            )
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise e

    def predict(self, frame: np.ndarray) -> Results:
        """Perform inference on a single video frame.

        Executes YOLOv8n inference on the input frame with confidence
        threshold filtering and person class filtering. The class filtering
        (classes=[0]) restricts detection to person objects only, which is
        critical for HSE compliance monitoring. This optimization reduces
        inference time by eliminating unnecessary class predictions and
        reduces false positives from non-person objects in industrial
        environments.

        The confidence threshold filtering occurs at the model level,
        ensuring only detections above the configured threshold are returned.
        This reduces post-processing overhead and ensures consistent
        performance characteristics across varying scene complexity.

        Args:
            frame (np.ndarray): Input video frame as numpy array with shape
                (height, width, 3) in BGR format (OpenCV standard).

        Returns:
            Results: Ultralytics Results object containing detection boxes,
                confidence scores, and class IDs. The results object provides
                access to .boxes attribute containing bounding box coordinates
                (xyxy format), confidence scores, and optional tracking IDs.
        """
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
            # Person class filtering: Restricts detection to class ID 0 (person)
            # from COCO dataset. This optimization reduces inference time by
            # eliminating predictions for 79 other COCO classes, which is
            # critical for real-time edge deployment where every millisecond
            # of latency impacts system responsiveness.
            classes=[0]
        )
        return results[0]
