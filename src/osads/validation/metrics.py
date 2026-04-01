"""Performance metrics for OSADS - tracks hit rates, latency, accuracy."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class FrameMetrics:
    """Metrics for a single frame."""

    timestamp: float
    num_detections: int
    num_tracks: int
    laser_on: bool
    tracking_error_px: float  # Distance laser-to-target in pixels
    is_hit: bool              # Laser within hit threshold
    latency_ms: float         # Processing time


class PerformanceTracker:
    """Tracks overall system performance over time."""

    def __init__(self, hit_threshold_px: float = 10.0) -> None:
        self.hit_threshold = hit_threshold_px
        self.frames: list[FrameMetrics] = []
        self._start_time = time.time()

    def record_frame(
        self,
        num_detections: int,
        num_tracks: int,
        laser_on: bool,
        tracking_error: float,
        latency_ms: float,
    ) -> FrameMetrics:
        metrics = FrameMetrics(
            timestamp=time.time() - self._start_time,
            num_detections=num_detections,
            num_tracks=num_tracks,
            laser_on=laser_on,
            tracking_error_px=tracking_error,
            is_hit=laser_on and tracking_error <= self.hit_threshold,
            latency_ms=latency_ms,
        )
        self.frames.append(metrics)
        return metrics

    def summary(self) -> dict[str, float]:
        if not self.frames:
            return {}

        total = len(self.frames)
        laser_frames = [f for f in self.frames if f.laser_on]
        hits = [f for f in self.frames if f.is_hit]

        return {
            "total_frames": total,
            "avg_detections": sum(f.num_detections for f in self.frames) / total,
            "avg_tracks": sum(f.num_tracks for f in self.frames) / total,
            "laser_active_pct": len(laser_frames) / total * 100 if total else 0,
            "hit_rate_pct": len(hits) / len(laser_frames) * 100 if laser_frames else 0,
            "avg_tracking_error_px": (
                sum(f.tracking_error_px for f in laser_frames) / len(laser_frames)
                if laser_frames else 0
            ),
            "avg_latency_ms": sum(f.latency_ms for f in self.frames) / total,
            "max_latency_ms": max(f.latency_ms for f in self.frames),
            "runtime_sec": self.frames[-1].timestamp if self.frames else 0,
        }
