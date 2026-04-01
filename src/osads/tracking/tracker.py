"""Multi-target tracker using Kalman filters.

Manages multiple tracked objects, handles assignment of detections
to existing tracks, creates new tracks, and removes lost tracks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from osads.detection.visual import Detection
from osads.tracking.kalman import KalmanTracker

logger = logging.getLogger(__name__)


@dataclass
class Track:
    """A tracked object with history."""

    track_id: int
    kalman: KalmanTracker
    class_name: str
    hits: int = 1           # Number of frames with detection
    misses: int = 0         # Consecutive frames without detection
    age: int = 0            # Total frames alive
    is_confirmed: bool = False
    history: list[tuple[float, float]] = field(default_factory=list)

    @property
    def position(self) -> tuple[float, float]:
        return self.kalman.position

    @property
    def velocity(self) -> tuple[float, float]:
        return self.kalman.velocity


class MultiTracker:
    """Tracks multiple objects across frames.

    Uses Hungarian algorithm (simplified) for detection-to-track assignment.
    """

    def __init__(
        self,
        max_lost_frames: int = 15,
        min_hits_to_confirm: int = 3,
        max_distance: float = 50.0,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
    ) -> None:
        self.max_lost = max_lost_frames
        self.min_hits = min_hits_to_confirm
        self.max_distance = max_distance
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.tracks: list[Track] = []
        self._next_id = 0

    def update(self, detections: list[Detection]) -> list[Track]:
        """Update tracks with new detections.

        Args:
            detections: Current frame detections.

        Returns:
            List of active (confirmed) tracks.
        """
        # Step 1: Predict all tracks
        for track in self.tracks:
            track.kalman.predict()
            track.age += 1

        # Step 2: Compute distance matrix
        if self.tracks and detections:
            distances = self._compute_distances(detections)
            matched, unmatched_dets, unmatched_tracks = self._assign(
                distances, detections
            )
        else:
            matched = []
            unmatched_dets = list(range(len(detections)))
            unmatched_tracks = list(range(len(self.tracks)))

        # Step 3: Update matched tracks
        for det_idx, track_idx in matched:
            det = detections[det_idx]
            track = self.tracks[track_idx]
            track.kalman.update(det.x, det.y)
            track.hits += 1
            track.misses = 0
            track.class_name = det.class_name
            track.history.append(track.position)
            if track.hits >= self.min_hits:
                track.is_confirmed = True

        # Step 4: Handle unmatched tracks (lost)
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].misses += 1

        # Step 5: Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            self._create_track(det)

        # Step 6: Remove dead tracks
        self.tracks = [t for t in self.tracks if t.misses <= self.max_lost]

        return [t for t in self.tracks if t.is_confirmed]

    def _create_track(self, detection: Detection) -> Track:
        kalman = KalmanTracker(
            initial_x=detection.x,
            initial_y=detection.y,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
        )
        track = Track(
            track_id=self._next_id,
            kalman=kalman,
            class_name=detection.class_name,
        )
        track.history.append(track.position)
        self._next_id += 1
        self.tracks.append(track)
        return track

    def _compute_distances(self, detections: list[Detection]) -> np.ndarray:
        """Compute distance matrix between detections and tracks."""
        n_dets = len(detections)
        n_tracks = len(self.tracks)
        distances = np.full((n_dets, n_tracks), float("inf"))

        for i, det in enumerate(detections):
            for j, track in enumerate(self.tracks):
                tx, ty = track.position
                dx = det.x - tx
                dy = det.y - ty
                distances[i, j] = np.sqrt(dx * dx + dy * dy)

        return distances

    def _assign(
        self,
        distances: np.ndarray,
        detections: list[Detection],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Greedy assignment of detections to tracks."""
        n_dets, n_tracks = distances.shape
        matched: list[tuple[int, int]] = []
        used_dets: set[int] = set()
        used_tracks: set[int] = set()

        # Sort all pairs by distance
        pairs = []
        for i in range(n_dets):
            for j in range(n_tracks):
                if distances[i, j] <= self.max_distance:
                    pairs.append((distances[i, j], i, j))
        pairs.sort()

        for _, det_idx, track_idx in pairs:
            if det_idx not in used_dets and track_idx not in used_tracks:
                matched.append((det_idx, track_idx))
                used_dets.add(det_idx)
                used_tracks.add(track_idx)

        unmatched_dets = [i for i in range(n_dets) if i not in used_dets]
        unmatched_tracks = [j for j in range(n_tracks) if j not in used_tracks]

        return matched, unmatched_dets, unmatched_tracks

    def get_target(self) -> Track | None:
        """Get the highest-priority target for the laser.

        Priority: longest-tracked confirmed target.
        """
        confirmed = [t for t in self.tracks if t.is_confirmed]
        if not confirmed:
            return None
        return max(confirmed, key=lambda t: t.hits)
