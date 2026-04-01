"""Visual insect detection using OpenCV and YOLO.

Two detection modes:
1. Motion detection (Background Subtraction) - fast, no ML needed
2. ML detection (YOLOv8) - accurate classification of insect type

Both can run independently or combined.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """A single visual detection result."""

    x: int  # Center x
    y: int  # Center y
    w: int  # Bounding box width
    h: int  # Bounding box height
    confidence: float
    class_name: str  # mosquito, gnat, fly, unknown
    class_id: int = -1

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Return (x1, y1, x2, y2) bounding box."""
        return (
            self.x - self.w // 2,
            self.y - self.h // 2,
            self.x + self.w // 2,
            self.y + self.h // 2,
        )

    @property
    def center(self) -> tuple[int, int]:
        return (self.x, self.y)

    @property
    def area(self) -> int:
        return self.w * self.h


class MotionDetector:
    """Detect moving objects using background subtraction.

    Good for initial detection of any flying object.
    Fast and works without ML model.
    """

    def __init__(
        self,
        min_area: int = 5,
        max_area: int = 500,
        learning_rate: float = 0.005,
        history: int = 200,
    ) -> None:
        self.min_area = min_area
        self.max_area = max_area
        self.learning_rate = learning_rate
        self._prev_gray: np.ndarray | None = None

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect moving objects using frame differencing.

        More robust than BG subtraction for small fast-moving objects.

        Args:
            frame: BGR image (H, W, 3).

        Returns:
            List of Detection objects for moving objects.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray.copy()
            return []

        # Frame difference
        diff = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray.copy()

        # Threshold
        _, fg_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)

        # Minimal morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                cx = x + w // 2
                cy = y + h // 2
                # Confidence based on how well-defined the contour is
                # Confidence based on how well the contour matches expected insect size
                # Scale so typical insect sizes (5-50px area) give reasonable confidence
                conf = min(1.0, max(0.3, area / 50.0))
                detections.append(Detection(
                    x=cx, y=cy, w=w, h=h,
                    confidence=conf,
                    class_name="unknown",
                ))

        return detections


class InsectClassifier:
    """Classify detected objects as specific insect types.

    Uses a simple CNN that classifies small image patches.
    For the simulation, we use size-based heuristics.
    When a trained model is available, it uses the ML model.
    """

    def __init__(self, model_path: str | None = None) -> None:
        self.model = None
        self.model_path = model_path
        self._use_ml = False

        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str) -> None:
        """Load YOLO or custom classification model."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(path)
            self._use_ml = True
            logger.info(f"Loaded ML model from {path}")
        except Exception as e:
            logger.warning(f"Could not load ML model: {e}. Using heuristic classifier.")

    def classify(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> list[Detection]:
        """Classify each detection as an insect type.

        Args:
            frame: Full BGR image.
            detections: List of detections from MotionDetector.

        Returns:
            Updated detections with class_name set.
        """
        if self._use_ml and self.model is not None:
            return self._classify_ml(frame, detections)
        return self._classify_heuristic(detections)

    def _classify_heuristic(self, detections: list[Detection]) -> list[Detection]:
        """Simple size-based classification (for simulation/testing)."""
        for det in detections:
            area = det.area
            if area < 20:
                det.class_name = "gnat"
                det.class_id = 1
            elif area < 60:
                det.class_name = "mosquito"
                det.class_id = 0
            else:
                det.class_name = "fly"
                det.class_id = 2
        return detections

    def _classify_ml(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> list[Detection]:
        """ML-based classification using YOLO."""
        results = self.model(frame, verbose=False)
        ml_detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = result.names.get(cls_id, "unknown")
                w = int(x2 - x1)
                h = int(y2 - y1)
                ml_detections.append(Detection(
                    x=int(x1 + w // 2),
                    y=int(y1 + h // 2),
                    w=w, h=h,
                    confidence=conf,
                    class_name=cls_name,
                    class_id=cls_id,
                ))
        return ml_detections


class VisualDetectionPipeline:
    """Complete visual detection pipeline combining motion + classification."""

    def __init__(
        self,
        model_path: str | None = None,
        min_area: int = 5,
        max_area: int = 500,
        confidence_threshold: float = 0.3,
    ) -> None:
        self.motion_detector = MotionDetector(min_area=min_area, max_area=max_area)
        self.classifier = InsectClassifier(model_path=model_path)
        self.confidence_threshold = confidence_threshold
        self.frame_count = 0

    def process_frame(self, frame: np.ndarray) -> list[Detection]:
        """Process a single frame through the full detection pipeline.

        Args:
            frame: BGR image.

        Returns:
            List of classified detections.
        """
        self.frame_count += 1

        # Step 1: Detect motion
        detections = self.motion_detector.detect(frame)

        # Step 2: Classify detected objects
        if detections:
            detections = self.classifier.classify(frame, detections)

        # Step 3: Filter by confidence
        detections = [d for d in detections if d.confidence >= self.confidence_threshold]

        return detections

    def draw_detections(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> np.ndarray:
        """Draw bounding boxes and labels on frame."""
        output = frame.copy()
        colors = {
            "mosquito": (0, 255, 0),
            "gnat": (0, 255, 255),
            "fly": (255, 0, 0),
            "unknown": (128, 128, 128),
        }

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = colors.get(det.class_name, (255, 255, 255))
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)
            label = f"{det.class_name} {det.confidence:.2f}"
            cv2.putText(
                output, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )
            # Draw crosshair at center
            cv2.drawMarker(
                output, det.center, color,
                cv2.MARKER_CROSS, 10, 1,
            )

        return output
