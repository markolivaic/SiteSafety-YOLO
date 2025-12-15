# SiteSafety-YOLO: Automated HSE Compliance Monitoring with Edge AI

## Project Overview

The SiteSafety-YOLO system is a Proof-of-Concept designed to address Health, Safety, and Environment (HSE) compliance challenges in industrial environments through automated computer vision monitoring. The system employs YOLOv8n (You Only Look Once version 8 nano) neural networks for real-time person detection combined with geometric analysis to enable proactive safety violation detection and automated compliance reporting.

Traditional manual HSE monitoring approaches introduce latency in identifying safety violations and require constant human supervision. This system provides real-time object detection with confidence-based filtering and zone status monitoring, allowing safety operators to respond immediately to compliance violations such as overcrowded restricted zones or unauthorized personnel presence, thereby reducing accident risk and ensuring regulatory compliance.

## System Demonstration

Real-time inference output showing personnel detection and operational dashboard overlay.

![SiteSafety Demo](assets/demo_preview.gif)

Figure 1: The system operating on standard CCTV footage, visualizing bounding boxes and safety metrics.

## System Architecture

The system implements a complete computer vision pipeline from video ingestion through annotated output generation:

```
Video Ingestion → Preprocessing → YOLOv8 Inference → Analysis Logic → Business Logic → Visualization
```

### Pipeline Components

1. **Video Ingestion**: Reads video files from the input directory and extracts frame-by-frame data for processing. Supports standard video formats (MP4, AVI) with automatic codec detection.

2. **Preprocessing**: Frames are prepared for YOLO inference by maintaining original resolution and BGR color format (OpenCV standard). No additional preprocessing is required as YOLOv8n accepts raw video frames directly.

3. **YOLOv8 Inference**: YOLOv8n model performs real-time object detection optimized for edge deployment. The nano variant (YOLOv8n) provides the optimal speed/accuracy trade-off for edge devices, achieving 5-10x inference speedup with CUDA acceleration compared to CPU-only execution. Detection is filtered to person class (COCO class ID 0) to optimize inference speed and reduce false positives.

4. **Geometric Analysis**: Detection results are analyzed to extract bounding box coordinates, confidence scores, and optional tracking IDs. Current implementation uses count-based zone status monitoring, with future enhancements planned for polygon-based geofencing using computational geometry (cv2.pointPolygonTest) for sophisticated intrusion detection.

5. **Business Logic**: Confidence-based filtering and zone compliance checking determine visual annotation colors and operational status. High-confidence detections (>0.8) are marked as safe (green), while lower-confidence detections require operator verification (red). Zone status is determined by personnel count thresholds (currently 5 personnel for compliance).

6. **Visualization**: Annotated frames include bounding boxes, confidence labels, tracking IDs, and a semi-transparent operational dashboard overlay displaying real-time metrics and compliance status.

## Key Features

- **Real-time Object Detection**: YOLOv8n inference optimized for edge devices with CUDA acceleration support, achieving sub-100ms inference latency on GPU-enabled systems
- **Count-based Zone Monitoring**: Personnel count thresholds provide immediate compliance feedback for restricted zones
- **Confidence-based Filtering**: Visual distinction between high-confidence (automated decision-ready) and low-confidence (operator verification required) detections
- **Edge Optimization**: YOLOv8n nano model variant balances speed and accuracy for real-time deployment on resource-constrained edge devices
- **Automated Compliance Reporting**: Real-time zone status indicators enable proactive safety management
- **Resource Management**: Proper video capture and writer resource cleanup ensures reliable long-running operation

## Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended for real-time inference, optional for CPU-only operation)
- Docker (optional, for containerized deployment)
- 4GB RAM minimum (8GB recommended for video processing)
- Internet connection for dependency installation and model weight download

## Installation & Setup

### Virtual Environment Setup

Execute the following commands to establish an isolated Python environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Dependency Installation

Install required packages using pip:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The first execution will automatically download YOLOv8n model weights (yolov8n.pt) from Ultralytics repository if not present in the project directory.

### Verify Installation

Verify CUDA availability (if using GPU acceleration):

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Usage

### Command-Line Execution

Initiate the video processing pipeline via command-line interface:

```bash
python main.py --video worker-zone-detection.mp4
```

This command executes the following operations:

1. Loads YOLOv8n model weights and initializes detector with CUDA acceleration (if available)
2. Opens input video file from `data/input/` directory
3. Processes each frame through YOLO inference and annotation pipeline
4. Writes annotated output video to `data/output/processed_<filename>`
5. Displays progress bar and processing statistics

The pipeline automatically handles resource cleanup and provides logging output for monitoring processing status.

### Docker Deployment

#### Build Container Image

Construct the Docker image with the following command:

```bash
docker build -t sitesafety-yolo:latest .
```

The build process installs all dependencies and prepares the container for video processing operations.

#### Run Container

Execute the containerized processing pipeline:

```bash
docker run -v $(pwd)/data:/app/data sitesafety-yolo:latest --video worker-zone-detection.mp4
```

The `-v` flag mounts the local `data/` directory into the container, enabling access to input videos and output generation.

#### Container Configuration

The Dockerfile configuration:

- Base image: `python:3.9-slim`
- Working directory: `/app`
- System dependencies: OpenCV libraries (libgl1-mesa-glx, libglib2.0-0)
- Entry point: `python main.py` with command-line argument forwarding

## Methodology

### YOLOv8n Selection Rationale

YOLOv8n (nano variant) was selected for this application based on the speed/accuracy trade-off required for edge deployment in industrial environments. The nano model provides:

- **Inference Speed**: 5-10x faster than larger YOLOv8 variants (s, m, l, x) on edge devices
- **Model Size**: 6.2MB model weights enable deployment on resource-constrained devices
- **Accuracy**: Maintains sufficient detection accuracy (mAP@0.5:0.95 ≈ 0.37) for person detection in industrial scenarios
- **Edge Optimization**: Designed specifically for real-time inference on edge devices with limited computational resources

The speed advantage is critical for real-time HSE monitoring where sub-100ms inference latency enables immediate compliance violation detection and operator response. Larger YOLOv8 variants (s, m, l, x) provide higher accuracy but introduce unacceptable latency (200-500ms) for real-time video processing at standard frame rates (25-30 FPS).

### Confidence Threshold Selection

The confidence threshold of 0.45 was selected to balance recall (detecting all personnel) and precision (reducing false positives). Lower thresholds (0.3-0.4) increase recall but introduce false positives from background objects, while higher thresholds (0.5-0.6) reduce false positives but may miss partially occluded personnel. The 0.45 threshold provides optimal balance for industrial safety monitoring where missing personnel detections poses greater risk than occasional false positives.

### Zone Compliance Logic

Current implementation uses count-based zone status monitoring with a threshold of 5 personnel. Zones with fewer than 5 detected personnel are marked as COMPLIANT (green), while zones with 5 or more personnel are marked as OVERCROWDED (red). This threshold-based approach provides immediate visual feedback, though future enhancements will implement polygon-based geofencing using computational geometry (cv2.pointPolygonTest) for sophisticated zone definitions and intrusion detection.

## Technical Highlights

### Latency Optimization

Real-time inference optimization through YOLOv8n model selection and CUDA acceleration enables sub-100ms frame processing latency. This performance characteristic ensures the system can process standard video frame rates (25-30 FPS) in real-time, providing immediate compliance violation detection without introducing processing delays that could impact safety response times.

### Automated Compliance

Confidence-based detection filtering and zone status monitoring enable automated compliance decision-making. High-confidence detections (>0.8) can trigger automated alerts or compliance logging without operator intervention, while lower-confidence detections require operator verification to prevent false violation reports. This hybrid approach balances automation benefits with safety-critical verification requirements.

### Computational Geometry

The system architecture supports future enhancements for polygon-based geofencing using computational geometry algorithms (cv2.pointPolygonTest). This will enable sophisticated zone definitions beyond simple count thresholds, allowing safety operators to define complex polygonal restricted zones and detect geometric intrusion events with precise coordinate-based analysis.

### Resource Management

Proper video capture and writer resource management through try-finally blocks ensures reliable long-running operation. The resource cleanup mechanism prevents file handle leaks and ensures output videos are properly finalized with correct headers, enabling playback in standard video players even if processing is interrupted.

## Project Structure

```
SiteSafety-YOLO/
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration parameters and paths
│   ├── detector.py             # YOLOv8n model initialization and inference
│   └── visualizer.py           # Frame annotation and dashboard rendering
├── data/
│   ├── input/                  # Input video files
│   └── output/                 # Processed annotated videos
├── notebooks/
│   └── exploration.ipynb       # Research and analysis notebooks
├── main.py                     # Video processing pipeline entry point
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container configuration
└── README.md                   # This file
```

## Configuration

Key hyperparameters can be adjusted in `src/config.py`:

- `MODEL_WEIGHTS`: YOLOv8n model weights filename (default: 'yolov8n.pt')
- `CONFIDENCE_THRESHOLD`: Minimum confidence score for detections (default: 0.45)
- `IOU_THRESHOLD`: Intersection over Union threshold for non-maximum suppression (default: 0.45)
- `CLASS_ID_PERSON`: COCO dataset class ID for person detection (default: 0)
- `COLOR_SAFE`: BGR color tuple for high-confidence detections (default: (0, 255, 0) - green)
- `COLOR_WARNING`: BGR color tuple for low-confidence detections (default: (0, 0, 255) - red)

Visual configuration parameters (colors, font) can be modified to match organizational branding or operator interface requirements.

## License

This project is provided as a Proof-of-Concept for research and evaluation purposes.

## Contact

For technical inquiries regarding architecture, implementation, or deployment, please refer to the inline documentation in source files or the module docstrings for detailed component descriptions.
