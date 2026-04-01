"""OSADS Main Pipeline - Detection → Tracking → Aiming → Validation.

Supports 3 target modes: mosquito, gnat, fly
Each mode uses a dedicated binary audio classifier.
Keyboard controls (GUI mode):
  q = quit, s = stats, 1 = mosquito, 2 = gnat, 3 = fly
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from osads.config import load_config
from osads.control.gimbal import SimulatedGimbal
from osads.detection.acoustic import AcousticDetection, FrequencyAnalyzer
from osads.detection.fusion import SensorFusion
from osads.detection.visual import VisualDetectionPipeline
from osads.simulation.fake_insects import InsectSwarm
from osads.tracking.tracker import MultiTracker
from osads.validation.metrics import PerformanceTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODES = {"mosquito": "MOSQUITO", "gnat": "GNAT/MUECKE", "fly": "FLY/FLIEGE"}


class OSADSPipeline:
    """Main pipeline: Detect → Track → Aim → Validate."""

    def __init__(self, config_path: str = "config/default.yaml") -> None:
        self.config = load_config(config_path)
        self.is_simulation = self.config["system"]["mode"] == "simulation"
        self.target_mode = self.config["system"].get("target_mode", "mosquito")

        # Detection
        self.visual = VisualDetectionPipeline(
            confidence_threshold=self.config["detection"]["visual"]["confidence_threshold"],
            min_area=2,
            max_area=500,
        )
        self.acoustic = FrequencyAnalyzer(
            sample_rate=self.config["detection"]["acoustic"]["sample_rate"],
            chunk_size=self.config["detection"]["acoustic"]["chunk_size"],
        )
        self.fusion = SensorFusion(
            visual_weight=self.config["detection"]["fusion"]["visual_weight"],
            acoustic_weight=self.config["detection"]["fusion"]["acoustic_weight"],
        )

        # Binary audio classifiers (one per mode)
        self.audio_classifiers: dict[str, Any] = {}
        self._load_audio_classifiers()

        # Tracking
        self.tracker = MultiTracker(
            max_lost_frames=self.config["tracking"]["max_lost_frames"],
            min_hits_to_confirm=self.config["tracking"]["min_hits"],
        )

        # Control
        sim_cfg = self.config.get("simulation", {})
        arena = sim_cfg.get("arena_size", [640, 480])
        self.gimbal = SimulatedGimbal(frame_width=arena[0], frame_height=arena[1])

        # Validation
        self.metrics = PerformanceTracker(hit_threshold_px=10.0)

        # Simulation
        if self.is_simulation:
            self.swarm = InsectSwarm(
                arena_width=arena[0],
                arena_height=arena[1],
                num_mosquitoes=2,
                num_gnats=1,
                num_flies=1,
            )
        else:
            self.swarm = None

        self.running = False
        self.frame_count = 0

    def _load_audio_classifiers(self) -> None:
        """Load binary audio classifiers for each mode if available."""
        try:
            from osads.training.binary_audio_model import BinaryAudioTrainer
            for mode in MODES:
                model_path = Path(f"models/{mode}_detector.pt")
                if model_path.exists():
                    trainer = BinaryAudioTrainer(target_insect=mode)
                    trainer.load(model_path)
                    self.audio_classifiers[mode] = trainer
                    logger.info(f"Loaded audio model: {mode}")
        except Exception as e:
            logger.warning(f"Could not load audio classifiers: {e}")

    def set_mode(self, mode: str) -> None:
        """Switch target mode."""
        if mode in MODES:
            self.target_mode = mode
            logger.info(f"Mode switched to: {MODES[mode]}")

    def run(self, max_frames: int = 0, show_gui: bool = True) -> dict[str, Any]:
        """Run the main pipeline loop."""
        self.running = True
        logger.info(
            f"OSADS starting | Mode: {MODES[self.target_mode]} | "
            f"{'SIMULATION' if self.is_simulation else 'HARDWARE'}"
        )

        try:
            while self.running:
                t_start = time.perf_counter()

                # 1. Get sensor data
                frame, audio = self._get_sensor_data()

                # 2. Visual detection
                visual_dets = self.visual.process_frame(frame)

                # 3. Acoustic detection (FFT)
                acoustic_det = self.acoustic.analyze(audio) if audio is not None else None

                # 4. Audio ML classification (binary, mode-specific)
                audio_confirmed = False
                audio_conf = 0.0
                if audio is not None and self.target_mode in self.audio_classifiers:
                    from osads.detection.acoustic import MelSpectrogramExtractor
                    if not hasattr(self, "_mel_ext"):
                        self._mel_ext = MelSpectrogramExtractor()
                    mel = self._mel_ext.extract(audio)
                    trainer = self.audio_classifiers[self.target_mode]
                    audio_confirmed, audio_conf = trainer.predict(mel)

                # 5. Sensor fusion
                fused = self.fusion.fuse(visual_dets, acoustic_det)

                # 6. Update tracker
                active_tracks = self.tracker.update(visual_dets)

                # 7. Aim gimbal at best target
                target = self.tracker.get_target()
                tracking_error = 999.0
                if target:
                    tx, ty = target.position
                    self.gimbal.aim_at_pixel(tx, ty)
                    self.gimbal.set_laser(True)
                    tracking_error = self.gimbal.tracking_error(tx, ty)
                else:
                    self.gimbal.set_laser(False)

                # 8. Measure performance
                latency = (time.perf_counter() - t_start) * 1000
                self.metrics.record_frame(
                    num_detections=len(visual_dets),
                    num_tracks=len(active_tracks),
                    laser_on=self.gimbal.laser_on,
                    tracking_error=tracking_error,
                    latency_ms=latency,
                )

                # 9. Visualize
                if show_gui:
                    display = self._render(
                        frame, visual_dets, active_tracks, target,
                        audio_confirmed, audio_conf,
                    )
                    cv2.imshow("OSADS - Air Defense System", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        self.running = False
                    elif key == ord("s"):
                        self._print_stats()
                    elif key == ord("1"):
                        self.set_mode("mosquito")
                    elif key == ord("2"):
                        self.set_mode("gnat")
                    elif key == ord("3"):
                        self.set_mode("fly")

                self.frame_count += 1
                if max_frames > 0 and self.frame_count >= max_frames:
                    self.running = False

                if self.frame_count % 100 == 0:
                    summary = self.metrics.summary()
                    logger.info(
                        f"Frame {self.frame_count} | "
                        f"Mode: {self.target_mode} | "
                        f"Dets: {len(visual_dets)} | "
                        f"Tracks: {len(active_tracks)} | "
                        f"Hit: {summary.get('hit_rate_pct', 0):.1f}% | "
                        f"Audio: {'YES' if audio_confirmed else 'no'} ({audio_conf:.2f}) | "
                        f"Lat: {latency:.1f}ms"
                    )

        except KeyboardInterrupt:
            logger.info("Stopped by user.")
        finally:
            if show_gui:
                cv2.destroyAllWindows()

        summary = self.metrics.summary()
        self._print_stats()
        return summary

    def _get_sensor_data(self) -> tuple[np.ndarray, np.ndarray | None]:
        if self.is_simulation and self.swarm:
            self.swarm.update()
            frame = self.swarm.render_frame()
            audio = self.swarm.generate_mixed_audio(
                sample_rate=self.config["detection"]["acoustic"]["sample_rate"],
                duration=self.config["detection"]["acoustic"]["chunk_size"]
                / self.config["detection"]["acoustic"]["sample_rate"],
            )
            return frame, audio
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        return frame, None

    def _render(
        self,
        frame: np.ndarray,
        detections: list,
        tracks: list,
        target: Any,
        audio_confirmed: bool = False,
        audio_conf: float = 0.0,
    ) -> np.ndarray:
        display = self.visual.draw_detections(frame, detections)

        # Draw tracks with trail
        for track in tracks:
            tx, ty = int(track.position[0]), int(track.position[1])
            if len(track.history) > 1:
                pts = [(int(x), int(y)) for x, y in track.history[-20:]]
                for i in range(1, len(pts)):
                    cv2.line(display, pts[i - 1], pts[i], (100, 100, 255), 1)
            cv2.putText(
                display, f"T{track.track_id}",
                (tx + 10, ty - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1,
            )

        # Draw laser
        if self.gimbal.laser_on and target:
            lx, ly = self.gimbal.get_laser_point()
            lx, ly = int(lx), int(ly)
            cv2.line(display, (display.shape[1] // 2, display.shape[0]), (lx, ly), (0, 0, 255), 1)
            cv2.circle(display, (lx, ly), 4, (0, 0, 255), -1)
            error = self.gimbal.tracking_error(target.position[0], target.position[1])
            color = (0, 255, 0) if error < 10 else (0, 165, 255)
            cv2.putText(
                display, f"ERR: {error:.1f}px",
                (10, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
            )

        # HUD - Top left
        summary = self.metrics.summary()
        h = display.shape[0]
        mode_color = {"mosquito": (0, 255, 0), "gnat": (0, 255, 255), "fly": (255, 0, 0)}
        mc = mode_color.get(self.target_mode, (255, 255, 255))

        hud = [
            (f"MODE: {MODES[self.target_mode]}", mc),
            (f"Frame: {self.frame_count}", (0, 255, 0)),
            (f"Detections: {len(detections)}", (0, 255, 0)),
            (f"Tracks: {len(tracks)}", (0, 255, 0)),
            (f"Hit Rate: {summary.get('hit_rate_pct', 0):.1f}%", (0, 255, 0)),
            (f"Latency: {summary.get('avg_latency_ms', 0):.1f}ms", (0, 255, 0)),
            (f"Audio: {'CONFIRMED' if audio_confirmed else 'none'} ({audio_conf:.2f})", mc),
        ]
        for i, (line, color) in enumerate(hud):
            cv2.putText(display, line, (10, 15 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

        # Mode switch hint - bottom right
        hint = "[1]Mosquito [2]Gnat [3]Fly [Q]Quit [S]Stats"
        cv2.putText(display, hint, (display.shape[1] - 380, h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128, 128, 128), 1)

        return display

    def _print_stats(self) -> None:
        summary = self.metrics.summary()
        logger.info("=" * 50)
        logger.info(f"OSADS Performance | Mode: {MODES[self.target_mode]}")
        logger.info("=" * 50)
        for key, val in summary.items():
            logger.info(f"  {key}: {val:.2f}")


def main() -> None:
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/default.yaml"
    pipeline = OSADSPipeline(config_path)
    pipeline.run()


if __name__ == "__main__":
    main()
