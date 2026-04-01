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
    """Tracks overall system performance over time.

    Keeps a rolling window of recent frames to prevent unbounded memory growth.
    """

    MAX_FRAMES = 9000  # ~5 minutes at 30fps

    def __init__(self, hit_threshold_px: float = 10.0) -> None:
        self.hit_threshold = hit_threshold_px
        self.frames: list[FrameMetrics] = []
        self._start_time = time.time()
        # Cumulative counters preserved when rolling window drops old frames
        self._total_frame_count = 0
        self._total_hits = 0
        self._total_laser_frames = 0

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
        self._total_frame_count += 1
        if laser_on:
            self._total_laser_frames += 1
            if metrics.is_hit:
                self._total_hits += 1
        # Rolling window: drop oldest frames to bound memory usage
        if len(self.frames) > self.MAX_FRAMES:
            self.frames.pop(0)
        return metrics

    def summary(self) -> dict[str, float]:
        if not self.frames:
            return {}

        # Window stats (recent frames in rolling buffer)
        window = self.frames
        n = len(window)
        laser_frames = [f for f in window if f.laser_on]

        return {
            # Use cumulative counters for total_frames / hit_rate (survive rolling window)
            "total_frames": float(self._total_frame_count),
            "avg_detections": sum(f.num_detections for f in window) / n,
            "avg_tracks": sum(f.num_tracks for f in window) / n,
            "laser_active_pct": self._total_laser_frames / self._total_frame_count * 100,
            "hit_rate_pct": (
                self._total_hits / self._total_laser_frames * 100
                if self._total_laser_frames else 0.0
            ),
            "avg_tracking_error_px": (
                sum(f.tracking_error_px for f in laser_frames) / len(laser_frames)
                if laser_frames else 0.0
            ),
            "avg_latency_ms": sum(f.latency_ms for f in window) / n,
            "max_latency_ms": max(f.latency_ms for f in window),
            "runtime_sec": window[-1].timestamp,
        }
