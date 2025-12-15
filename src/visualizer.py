"""Frame annotation and visualization module for HSE compliance monitoring.

This module handles bounding box annotation, confidence-based color coding,
and operational dashboard rendering on video frames. The visualization
pipeline transforms raw YOLO detection results into actionable visual
feedback for safety operators, with real-time zone status indicators
based on personnel count thresholds.
"""

import cv2
import numpy as np
from ultralytics.engine.results import Results

from src.config import COLOR_SAFE, COLOR_WARNING, FONT


class FrameAnnotator:
    """Handles frame annotation and dashboard overlay rendering.

    The FrameAnnotator class processes YOLO detection results and renders
    bounding boxes, confidence labels, and operational metrics on video
    frames. It implements confidence-based color coding to visually
    distinguish high-confidence detections (green) from lower-confidence
    detections (red), enabling operators to quickly assess detection
    reliability. The dashboard overlay provides real-time compliance
    status based on personnel count thresholds.
    """

    @staticmethod
    def annotate(frame: np.ndarray, results: Results) -> np.ndarray:
        """Annotate frame with detection boxes and operational dashboard.

        Processes YOLO detection results to draw bounding boxes, confidence
        labels, and tracking IDs (when available) on the input frame. The
        method implements confidence-based color assignment: detections with
        confidence > 0.8 are rendered in green (COLOR_SAFE), indicating
        high reliability suitable for automated compliance decisions. Lower
        confidence detections are rendered in red (COLOR_WARNING), alerting
        operators to potential false positives requiring manual verification.

        The confidence threshold of 0.8 was selected based on operational
        requirements: high-confidence detections (>0.8) typically correspond
        to clear, unobstructed views of personnel, enabling automated
        compliance decisions. Lower confidence detections may result from
        partial occlusion, poor lighting, or distance, requiring operator
        review to prevent false compliance violations.

        Args:
            frame (np.ndarray): Input video frame as numpy array with shape
                (height, width, 3) in BGR format.
            results (Results): Ultralytics Results object containing detection
                boxes, confidence scores, and optional tracking IDs from
                YOLO inference.

        Returns:
            np.ndarray: Annotated frame with bounding boxes, labels, and
                dashboard overlay. The frame maintains the same shape and
                format as the input.
        """
        annotated_frame = frame.copy()
        person_count = 0

        if results.boxes:
            for box in results.boxes:
                person_count += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                # Confidence-based color assignment: High confidence (>0.8)
                # implies clearer visibility and reduced occlusion, enabling
                # safer automated compliance decisions. Lower confidence
                # detections require operator verification to prevent false
                # violations in challenging visual conditions.
                color = COLOR_SAFE if confidence > 0.8 else COLOR_WARNING

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # Tracking ID extraction: Ultralytics provides optional
                # tracking IDs when tracking is enabled. The 'N/A' fallback
                # handles detection-only mode where tracking is not active.
                track_id = int(box.id[0]) if box.id is not None else 'N/A'
                label = f"ID:{track_id} Conf:{confidence:.2f}"

                cv2.putText(
                    annotated_frame, label, (x1, y1 - 10), FONT, 0.5, color, 1
                )

        FrameAnnotator._render_dashboard(annotated_frame, person_count)

        return annotated_frame

    @staticmethod
    def _render_dashboard(frame: np.ndarray, count: int) -> None:
        """Render semi-transparent operational dashboard overlay.

        Creates a semi-transparent header overlay displaying real-time HSE
        monitoring metrics including active personnel count and zone compliance
        status. The overlay uses alpha blending (60% overlay, 40% original
        frame) to maintain visibility of underlying video content while
        providing clear operational feedback.

        Zone status logic implements count-based compliance checking: zones
        with fewer than 5 personnel are marked as COMPLIANT (green), while
        zones with 5 or more personnel are marked as OVERCROWDED (red). This
        threshold-based approach provides immediate visual feedback for
        safety operators, though future enhancements may implement geometric
        geofencing using cv2.pointPolygonTest for polygon-based intrusion
        detection.

        Args:
            frame (np.ndarray): Frame to render dashboard on (modified in-place).
            count (int): Current personnel count from detection results.
        """
        # Semi-transparent overlay: Creates a black rectangle overlay and
        # blends it with the original frame using weighted addition. The
        # 0.6 alpha for overlay and 0.4 for original frame ensures the
        # dashboard is clearly visible while maintaining context from the
        # underlying video feed.
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (400, 80), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(
            frame, "REAL-TIME HSE MONITORING", (10, 25), FONT, 0.6,
            (255, 255, 255), 1
        )
        cv2.putText(
            frame, f"Active Personnel: {count}", (10, 50), FONT, 0.6,
            (200, 200, 200), 1
        )

        # Zone compliance logic: Count-based threshold (5 personnel) provides
        # immediate compliance feedback. The threshold of 5 was selected
        # based on typical industrial safety requirements for restricted zones.
        # Future enhancements will implement polygon-based geofencing using
        # cv2.pointPolygonTest for geometric intrusion detection, enabling
        # more sophisticated zone definitions beyond simple count thresholds.
        status = "COMPLIANT" if count < 5 else "OVERCROWDED"
        status_color = COLOR_SAFE if status == "COMPLIANT" else COLOR_WARNING
        cv2.putText(
            frame, f"Zone Status: {status}", (10, 70), FONT, 0.6,
            status_color, 2
        )
