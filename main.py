"""Main video processing pipeline for SiteSafety-YOLO HSE monitoring.

This module orchestrates the complete computer vision pipeline from video
ingestion through YOLO inference to annotated output generation. It serves
as the entry point for video processing and handles resource management,
error handling, and progress tracking for batch video processing operations.
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

from src.config import (
    CONFIDENCE_THRESHOLD, INPUT_DIR, MODEL_WEIGHTS, OUTPUT_DIR
)
from src.detector import ObjectDetector
from src.visualizer import FrameAnnotator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def process_video(filename: str) -> None:
    """Execute the complete video processing pipeline.

    Processes a video file through the HSE monitoring pipeline: video
    ingestion, frame-by-frame YOLO inference, annotation with bounding boxes
    and compliance metrics, and output video generation. The pipeline
    handles resource management through try-finally blocks to ensure video
    capture and writer objects are properly released even in error scenarios.

    The video codec selection (XVID) is optimized for Windows compatibility,
    avoiding OpenH264 dependency issues that can occur with H.264 codecs
    on Windows systems. This ensures consistent deployment across different
    Windows environments without requiring additional codec installations.

    Resource management is critical for long-running video processing: the
    finally block guarantees that video capture and writer resources are
    released even if processing is interrupted (KeyboardInterrupt) or
    encounters errors. This prevents file handle leaks and ensures output
    videos are properly finalized with correct headers.

    Args:
        filename (str): Name of the video file in the data/input/ directory.
            The file must exist in INPUT_DIR or processing will be aborted
            with an error log message.

    Returns:
        None: Function returns None after processing completes or encounters
            an error. All status information is logged rather than returned.
    """
    input_path = INPUT_DIR / filename
    output_path = OUTPUT_DIR / f"processed_{filename}"

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return

    detector = ObjectDetector(MODEL_WEIGHTS, CONFIDENCE_THRESHOLD)
    annotator = FrameAnnotator()

    video_capture = cv2.VideoCapture(str(input_path))
    width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # XVID codec selection: Provides Windows compatibility without requiring
    # OpenH264 codec installation. XVID is widely supported across Windows
    # systems and avoids dependency issues that can occur with H.264 codecs
    # in containerized or minimal Windows environments.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter(
        str(output_path), fourcc, fps, (width, height)
    )

    logger.info(
        f"Processing started: {filename} [{width}x{height} @ {fps:.1f} FPS]"
    )

    try:
        for _ in tqdm(range(total_frames), desc="Inference Progress", unit="frames"):
            ret, frame = video_capture.read()
            if not ret:
                break

            results = detector.predict(frame)
            processed_frame = annotator.annotate(frame, results)
            video_writer.write(processed_frame)

    except KeyboardInterrupt:
        logger.warning("Process interrupted by user.")
    finally:
        # Resource cleanup: Ensures video capture and writer objects are
        # properly released even if processing is interrupted or encounters
        # errors. This prevents file handle leaks and ensures output video
        # files are properly finalized with correct headers, enabling
        # playback in standard video players.
        video_capture.release()
        video_writer.release()
        logger.info(f"Processing complete. Output saved to: {output_path}")


if __name__ == "__main__":
    """Execute the video processing pipeline via command-line interface.

    The pipeline consists of four stages:
    1. Argument parsing: Validates command-line input and extracts video filename
    2. Directory initialization: Ensures input and output directories exist
    3. Video processing: Executes the complete inference and annotation pipeline
    4. Resource cleanup: Automatically handled by process_video function
    """
    parser = argparse.ArgumentParser(
        description="SiteSafety AI Video Processing Pipeline"
    )
    parser.add_argument(
        "--video", type=str, required=True,
        help="Filename of the video in data/input/"
    )

    args = parser.parse_args()

    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    process_video(args.video)
