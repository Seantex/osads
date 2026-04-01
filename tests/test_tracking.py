"""Tests for tracking modules."""

import numpy as np
import pytest

from osads.tracking.kalman import KalmanTracker
from osads.tracking.tracker import MultiTracker
from osads.detection.visual import Detection


class TestKalmanTracker:
    """Tests for Kalman filter."""

    def test_constant_velocity_prediction(self) -> None:
        """Object moving at constant velocity should be predicted correctly."""
        # dt=1.0 so velocity in state matches px/frame directly
        tracker = KalmanTracker(initial_x=0, initial_y=0, dt=1.0)
        # Simulate constant velocity: move right 5px per frame
        for i in range(1, 20):
            tracker.predict()
            tracker.update(i * 5.0, 0.0)

        # Predict next position
        state = tracker.predict()
        # Should predict ~100 (20 * 5)
        assert abs(state[0] - 100.0) < 15.0
        assert abs(state[2] - 5.0) < 2.0  # velocity ~5 px/frame

    def test_predict_future(self) -> None:
        """Future predictions should be ordered."""
        tracker = KalmanTracker(initial_x=50, initial_y=50)
        for i in range(10):
            tracker.predict()
            tracker.update(50 + i * 3.0, 50 + i * 2.0)

        future = tracker.predict_future(steps=5)
        assert len(future) == 5
        # Each prediction should advance in roughly the same direction
        for i in range(1, len(future)):
            assert future[i][0] >= future[i - 1][0] - 5  # roughly rightward

    def test_speed_property(self) -> None:
        tracker = KalmanTracker(initial_x=0, initial_y=0)
        for i in range(10):
            tracker.predict()
            tracker.update(i * 3.0, i * 4.0)
        assert tracker.speed > 0


class TestMultiTracker:
    """Tests for multi-target tracking."""

    def test_creates_tracks_from_detections(self) -> None:
        tracker = MultiTracker(min_hits_to_confirm=2)
        dets = [
            Detection(x=100, y=100, w=10, h=10, confidence=0.9, class_name="mosquito"),
            Detection(x=300, y=200, w=8, h=8, confidence=0.8, class_name="fly"),
        ]
        # First frame: tracks created but not confirmed
        tracks = tracker.update(dets)
        assert len(tracks) == 0  # Need min_hits to confirm

        # Second frame: same positions → confirmed
        tracks = tracker.update(dets)
        assert len(tracks) == 2

    def test_tracks_moving_object(self) -> None:
        tracker = MultiTracker(min_hits_to_confirm=2, max_distance=50)
        for i in range(10):
            dets = [Detection(x=100 + i * 5, y=100, w=10, h=10,
                             confidence=0.9, class_name="mosquito")]
            tracks = tracker.update(dets)

        assert len(tracks) == 1
        assert tracks[0].class_name == "mosquito"

    def test_removes_lost_tracks(self) -> None:
        tracker = MultiTracker(min_hits_to_confirm=2, max_lost_frames=3)
        dets = [Detection(x=100, y=100, w=10, h=10, confidence=0.9, class_name="mosquito")]

        # Build up track
        for _ in range(5):
            tracker.update(dets)

        # Object disappears
        for _ in range(5):
            tracker.update([])

        tracks = tracker.update([])
        # Track should be removed after max_lost_frames
        assert len(tracks) == 0

    def test_get_target_returns_best(self) -> None:
        tracker = MultiTracker(min_hits_to_confirm=2)
        # Two objects
        for i in range(5):
            dets = [
                Detection(x=100, y=100, w=10, h=10, confidence=0.9, class_name="mosquito"),
                Detection(x=300, y=200, w=8, h=8, confidence=0.5, class_name="fly"),
            ]
            tracker.update(dets)

        target = tracker.get_target()
        assert target is not None
