"""Gimbal control - simulated or hardware Pan/Tilt with PID controller."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class GimbalState:
    """Current gimbal orientation."""

    pan: float = 0.0   # degrees, 0 = center
    tilt: float = 0.0  # degrees, 0 = center


class PIDController:
    """Discrete PID controller (frame-based, not time-based).

    Uses fixed dt=1 per call, making it deterministic and
    consistent regardless of actual frame rate.
    """

    def __init__(self, kp: float = 0.5, ki: float = 0.01, kd: float = 0.1) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._integral = 0.0
        self._prev_error = 0.0

    def compute(self, error: float) -> float:
        self._integral += error
        self._integral = max(-500, min(500, self._integral))  # Anti-windup
        derivative = error - self._prev_error
        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        self._prev_error = error
        return output

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = 0.0


class SimulatedGimbal:
    """Simulated gimbal for laptop testing.

    Directly tracks pixel coordinates with PID-smoothed movement.
    In hardware mode, this would convert pixels to servo angles.
    """

    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        fov_h: float = 60.0,
        fov_v: float = 45.0,
        max_speed: float = 300.0,
        kp: float = 0.7,
        ki: float = 0.02,
        kd: float = 0.15,
    ) -> None:
        self.frame_w = frame_width
        self.frame_h = frame_height
        self.fov_h = fov_h
        self.fov_v = fov_v
        self.max_speed = max_speed
        self.state = GimbalState()
        self.pid_x = PIDController(kp=kp, ki=ki, kd=kd)
        self.pid_y = PIDController(kp=kp, ki=ki, kd=kd)
        self.laser_on = False
        self._laser_x = float(frame_width / 2)
        self._laser_y = float(frame_height / 2)

    def aim_at_pixel(self, target_x: float, target_y: float) -> GimbalState:
        """Move laser toward target pixel. Called once per frame."""
        error_x = target_x - self._laser_x
        error_y = target_y - self._laser_y

        correction_x = self.pid_x.compute(error_x)
        correction_y = self.pid_y.compute(error_y)

        # Clamp movement per frame
        max_step = 80.0
        correction_x = max(-max_step, min(max_step, correction_x))
        correction_y = max(-max_step, min(max_step, correction_y))

        self._laser_x += correction_x
        self._laser_y += correction_y

        self._laser_x = max(0, min(self.frame_w - 1, self._laser_x))
        self._laser_y = max(0, min(self.frame_h - 1, self._laser_y))

        return self.state

    def get_laser_point(self) -> tuple[float, float]:
        return (self._laser_x, self._laser_y)

    def set_laser(self, on: bool) -> None:
        self.laser_on = on

    def tracking_error(self, target_x: float, target_y: float) -> float:
        """Distance in pixels between laser point and target."""
        return math.sqrt(
            (self._laser_x - target_x) ** 2 + (self._laser_y - target_y) ** 2
        )
