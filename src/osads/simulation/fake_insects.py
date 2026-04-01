"""Simulated insects for testing without real camera/microphone.

Generates realistic insect behavior:
- Random flight paths with direction changes
- Correct wing-beat frequencies for audio simulation
- Visual rendering on a virtual camera frame
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SimulatedInsect:
    """A single simulated insect with position, velocity, and type."""

    insect_type: str  # mosquito, gnat, fly
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    size: int = 5
    wing_freq: float = 550.0  # Hz
    alive: bool = True
    _id: int = 0

    # Type-specific defaults
    # Scientifically verified profiles
    # Freq sources: Arthur et al. PMC3985972, UF Insect Records, Rockstein 1966
    PROFILES: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        "mosquito": {
            "speed_range": (1.0, 5.0),
            "size_range": (3, 8),
            # Arthur et al. PMC3985972: Aedes aegypti female 421-578 Hz lab, ~664 Hz field
            # Male 571-832 Hz lab, ~982 Hz field. Full range for realism.
            "freq_range": (421.0, 700.0),
            "freq_peak": 511.0,   # Female lab mean ± 46 Hz (n=11)
            "direction_change_prob": 0.3,
        },
        "gnat": {
            "speed_range": (1.0, 3.0),
            "size_range": (2, 5),
            # Chironomidae / Bradysia: primary literature sparse; 650-900 Hz estimated
            "freq_range": (600.0, 900.0),
            "freq_peak": 700.0,
            "direction_change_prob": 0.4,
        },
        "fly": {
            "speed_range": (3.0, 10.0),
            "size_range": (5, 12),
            # Musca domestica: 190 Hz (Wikipedia Insect Flight), Rockstein 1966 ~200 Hz
            "freq_range": (160.0, 250.0),
            "freq_peak": 190.0,
            "direction_change_prob": 0.2,
        },
    })

    def __post_init__(self) -> None:
        profile = self.PROFILES.get(self.insect_type, self.PROFILES["mosquito"])
        if self.size == 5:  # default, randomize
            self.size = random.randint(*profile["size_range"])
        # Use full biological frequency range (not just ±30Hz from peak)
        # e.g. Aedes aegypti females span 421-578Hz (Arthur et al. PMC3985972)
        freq_min, freq_max = profile["freq_range"]
        self.wing_freq = random.uniform(freq_min, freq_max)
        speed_min, speed_max = profile["speed_range"]
        speed = random.uniform(speed_min, speed_max)
        angle = random.uniform(0, 2 * math.pi)
        self.vx = speed * math.cos(angle)
        self.vy = speed * math.sin(angle)

    def update(self, arena_w: int, arena_h: int) -> None:
        """Update position for one frame."""
        if not self.alive:
            return

        profile = self.PROFILES.get(self.insect_type, self.PROFILES["mosquito"])

        # Random direction change
        if random.random() < profile["direction_change_prob"]:
            speed_min, speed_max = profile["speed_range"]
            speed = random.uniform(speed_min, speed_max)
            angle = random.uniform(0, 2 * math.pi)
            self.vx = speed * math.cos(angle)
            self.vy = speed * math.sin(angle)

        # Slight random jitter (realistic flight)
        self.vx += random.gauss(0, 0.3)
        self.vy += random.gauss(0, 0.3)

        # Move
        self.x += self.vx
        self.y += self.vy

        # Bounce off walls
        if self.x < 0 or self.x >= arena_w:
            self.vx *= -1
            self.x = max(0, min(arena_w - 1, self.x))
        if self.y < 0 or self.y >= arena_h:
            self.vy *= -1
            self.y = max(0, min(arena_h - 1, self.y))

    def generate_audio_signal(
        self, sample_rate: int = 44100, duration: float = 0.05
    ) -> np.ndarray:
        """Generate audio signal of this insect's wingbeat."""
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        # Fundamental frequency + harmonics (realistic wing sound)
        signal = np.sin(2 * np.pi * self.wing_freq * t)
        signal += 0.3 * np.sin(2 * np.pi * self.wing_freq * 2 * t)  # 2nd harmonic
        signal += 0.1 * np.sin(2 * np.pi * self.wing_freq * 3 * t)  # 3rd harmonic
        # Amplitude modulation (wing beat envelope)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 15 * t)  # ~15Hz modulation
        signal *= envelope
        # Distance-based amplitude (closer = louder)
        amplitude = 0.3 * random.uniform(0.5, 1.0)
        return (signal * amplitude).astype(np.float32)


class InsectSwarm:
    """Manages a swarm of simulated insects."""

    def __init__(
        self,
        arena_width: int = 640,
        arena_height: int = 480,
        num_mosquitoes: int = 2,
        num_gnats: int = 1,
        num_flies: int = 1,
    ) -> None:
        self.arena_w = arena_width
        self.arena_h = arena_height
        self.insects: list[SimulatedInsect] = []
        self._next_id = 0

        for _ in range(num_mosquitoes):
            self._spawn("mosquito")
        for _ in range(num_gnats):
            self._spawn("gnat")
        for _ in range(num_flies):
            self._spawn("fly")

    def _spawn(self, insect_type: str) -> SimulatedInsect:
        insect = SimulatedInsect(
            insect_type=insect_type,
            x=random.uniform(50, self.arena_w - 50),
            y=random.uniform(50, self.arena_h - 50),
            _id=self._next_id,
        )
        self._next_id += 1
        self.insects.append(insect)
        return insect

    def update(self) -> None:
        """Update all insects for one frame."""
        for insect in self.insects:
            insect.update(self.arena_w, self.arena_h)

    def render_frame(self) -> np.ndarray:
        """Render insects onto a camera frame (BGR image)."""
        import cv2

        # Static background (consistent for background subtraction)
        if not hasattr(self, "_bg_frame"):
            # Create background once with fixed noise pattern
            self._bg_frame = np.zeros((self.arena_h, self.arena_w, 3), dtype=np.uint8)
            self._bg_frame[:] = (30, 30, 30)
            rng = np.random.RandomState(42)
            noise = rng.randint(0, 8, self._bg_frame.shape, dtype=np.uint8)
            self._bg_frame = cv2.add(self._bg_frame, noise)

        frame = self._bg_frame.copy()

        colors = {
            "mosquito": (0, 200, 0),
            "gnat": (0, 200, 200),
            "fly": (200, 0, 0),
        }

        for insect in self.insects:
            if not insect.alive:
                continue
            color = colors.get(insect.insect_type, (255, 255, 255))
            cx, cy = int(insect.x), int(insect.y)
            r = insect.size // 2

            # Draw insect body (small ellipse)
            cv2.ellipse(frame, (cx, cy), (r, r // 2), 0, 0, 360, color, -1)
            # Wings (tiny lines)
            wing_len = r + 2
            cv2.line(frame, (cx - wing_len, cy - 2), (cx, cy), color, 1)
            cv2.line(frame, (cx + wing_len, cy - 2), (cx, cy), color, 1)

        return frame

    def generate_mixed_audio(
        self, sample_rate: int = 44100, duration: float = 0.05
    ) -> np.ndarray:
        """Generate combined audio from all insects + background noise."""
        n_samples = int(sample_rate * duration)
        mixed = np.zeros(n_samples, dtype=np.float32)

        for insect in self.insects:
            if insect.alive:
                sig = insect.generate_audio_signal(sample_rate, duration)
                if len(sig) == n_samples:
                    mixed += sig

        # Add background noise
        noise = np.random.normal(0, 0.02, n_samples).astype(np.float32)
        mixed += noise

        # Clip to [-1, 1]
        mixed = np.clip(mixed, -1.0, 1.0)
        return mixed

    def get_ground_truth(self) -> list[dict[str, Any]]:
        """Return ground truth positions for validation."""
        return [
            {
                "id": insect._id,
                "type": insect.insect_type,
                "x": insect.x,
                "y": insect.y,
                "size": insect.size,
                "freq": insect.wing_freq,
                "alive": insect.alive,
            }
            for insect in self.insects
        ]
