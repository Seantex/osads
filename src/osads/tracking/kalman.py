"""Kalman Filter for insect/drone tracking and trajectory prediction.

Tracks position and velocity, predicts future position for laser lead.
State vector: [x, y, vx, vy]
Measurement: [x, y]
"""

from __future__ import annotations

import numpy as np


class KalmanTracker:
    """Kalman filter tracker for a single target.

    Tracks 2D position and velocity, provides trajectory prediction.
    """

    def __init__(
        self,
        initial_x: float = 0,
        initial_y: float = 0,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
        dt: float = 1.0 / 30,  # 30 FPS
    ) -> None:
        self.dt = dt

        # State: [x, y, vx, vy]
        self.state = np.array([initial_x, initial_y, 0.0, 0.0], dtype=np.float64)

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Measurement matrix (we observe x, y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Process noise covariance
        self.Q = np.eye(4, dtype=np.float64) * process_noise

        # Measurement noise covariance
        self.R = np.eye(2, dtype=np.float64) * measurement_noise

        # State covariance
        self.P = np.eye(4, dtype=np.float64) * 1.0

    def predict(self) -> np.ndarray:
        """Predict next state.

        Returns:
            Predicted [x, y, vx, vy].
        """
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state.copy()

    def update(self, measurement_x: float, measurement_y: float) -> np.ndarray:
        """Update state with new measurement.

        Args:
            measurement_x: Observed x position.
            measurement_y: Observed y position.

        Returns:
            Updated [x, y, vx, vy].
        """
        z = np.array([measurement_x, measurement_y], dtype=np.float64)
        y = z - self.H @ self.state  # Innovation
        S = self.H @ self.P @ self.H.T + self.R  # Innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.state = self.state + K @ y
        # Joseph form: numerically stable, keeps P symmetric positive-definite
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        return self.state.copy()

    def predict_future(self, steps: int = 5) -> list[tuple[float, float]]:
        """Predict future positions without updating internal state.

        Args:
            steps: Number of frames to predict ahead.

        Returns:
            List of (x, y) predicted positions.
        """
        state = self.state.copy()
        positions = []
        for _ in range(steps):
            state = self.F @ state
            positions.append((state[0], state[1]))
        return positions

    @property
    def position(self) -> tuple[float, float]:
        return (self.state[0], self.state[1])

    @property
    def velocity(self) -> tuple[float, float]:
        return (self.state[2], self.state[3])

    @property
    def speed(self) -> float:
        return float(np.sqrt(self.state[2] ** 2 + self.state[3] ** 2))
