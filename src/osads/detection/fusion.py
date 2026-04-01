"""Sensor Fusion: Combine visual and acoustic detections.

Merges information from camera and microphone to produce
higher-confidence detections than either sensor alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from osads.detection.acoustic import AcousticDetection
from osads.detection.visual import Detection

logger = logging.getLogger(__name__)


@dataclass
class FusedDetection:
    """Combined detection from visual + acoustic sensors."""

    visual: Detection | None
    acoustic: AcousticDetection | None
    fused_confidence: float
    insect_type: str
    x: int = 0
    y: int = 0

    @property
    def has_visual(self) -> bool:
        return self.visual is not None

    @property
    def has_acoustic(self) -> bool:
        return self.acoustic is not None and self.acoustic.detected


class SensorFusion:
    """Fuses visual and acoustic detections.

    Strategy:
    - If both sensors agree on type → high confidence
    - If only visual → use visual confidence
    - If only acoustic → use acoustic confidence (no position info)
    - If sensors disagree → lower confidence, prefer visual for type
    """

    def __init__(
        self,
        visual_weight: float = 0.6,
        acoustic_weight: float = 0.4,
        min_confidence: float = 0.5,
    ) -> None:
        self.visual_weight = visual_weight
        self.acoustic_weight = acoustic_weight
        self.min_confidence = min_confidence

    def fuse(
        self,
        visual_detections: list[Detection],
        acoustic: AcousticDetection | None,
    ) -> list[FusedDetection]:
        """Fuse visual and acoustic detections.

        Args:
            visual_detections: List of visual detections from camera.
            acoustic: Acoustic detection result (one per audio chunk).

        Returns:
            List of fused detections.
        """
        results: list[FusedDetection] = []

        if not visual_detections and (acoustic is None or not acoustic.detected):
            return results

        # If we have visual detections, enhance them with acoustic info
        if visual_detections:
            for vis_det in visual_detections:
                fused_conf = vis_det.confidence * self.visual_weight

                acoustic_match = None
                if acoustic and acoustic.detected:
                    # Check if acoustic type matches visual type
                    if acoustic.insect_type == vis_det.class_name:
                        # Agreement → boost confidence
                        fused_conf += acoustic.confidence * self.acoustic_weight
                        acoustic_match = acoustic
                    else:
                        # Disagreement → still add some acoustic confidence
                        fused_conf += acoustic.confidence * self.acoustic_weight * 0.3
                        acoustic_match = acoustic

                if fused_conf >= self.min_confidence:
                    results.append(FusedDetection(
                        visual=vis_det,
                        acoustic=acoustic_match,
                        fused_confidence=min(1.0, fused_conf),
                        insect_type=vis_det.class_name,
                        x=vis_det.x,
                        y=vis_det.y,
                    ))

        # Acoustic-only detection (no visual)
        elif acoustic and acoustic.detected:
            if acoustic.confidence * self.acoustic_weight >= self.min_confidence:
                results.append(FusedDetection(
                    visual=None,
                    acoustic=acoustic,
                    fused_confidence=acoustic.confidence * self.acoustic_weight,
                    insect_type=acoustic.insect_type or "unknown",
                ))

        return results
