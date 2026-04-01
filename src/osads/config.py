"""Configuration loader for OSADS."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "config/default.yaml") -> dict[str, Any]:
    """Load YAML configuration file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


@dataclass
class FrequencyRange:
    """Frequency range for an insect type."""

    min: float
    max: float
    peak: float

    def contains(self, freq: float) -> bool:
        return self.min <= freq <= self.max


@dataclass
class InsectProfile:
    """Profile defining an insect type's characteristics."""

    name: str
    frequency: FrequencyRange
    speed_range: tuple[float, float] = (1.0, 5.0)
    size_range: tuple[int, int] = (3, 8)

    @classmethod
    def from_config(cls, name: str, config: dict[str, Any]) -> InsectProfile:
        freq_cfg = config["detection"]["acoustic"]["frequency_ranges"][name]
        sim_cfg = config.get("simulation", {}).get("insect_types", {}).get(name, {})
        return cls(
            name=name,
            frequency=FrequencyRange(
                min=freq_cfg["min"],
                max=freq_cfg["max"],
                peak=freq_cfg["peak"],
            ),
            speed_range=tuple(sim_cfg.get("speed_range", [1, 5])),
            size_range=tuple(sim_cfg.get("size_range", [3, 8])),
        )


# Standard insect profiles
INSECT_CLASSES = ["mosquito", "gnat", "fly"]
