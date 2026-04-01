"""OSADS Stress Test — Dauertest für alle Modi, Komponenten und Edge Cases.

Testet:
  - Alle 3 Modi je 10.000 Frames (~5min bei 30fps)
  - Memory-Stabilität (Track-History, PerformanceTracker)
  - Kalman-Filter Numerik (Joseph-Form Stabilität)
  - FrequencyAnalyzer Robustheit bei extremen Eingaben
  - MultiTracker bei hoher Insektendichte
  - PID-Controller Konvergenz & Anti-Windup
  - ModeSwitch unter Last
  - Edge Cases: leerer Frame, Stille, NaN/Inf-Werte
"""

from __future__ import annotations

import gc
import logging
import math
import sys
import time
import traceback
import tracemalloc
from pathlib import Path

import numpy as np

sys.path.insert(0, "src")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("stress")
logging.getLogger("osads").setLevel(logging.WARNING)

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

results: list[tuple[str, bool, str]] = []


def record(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    mark = PASS if ok else FAIL
    print(f"  {mark} {name}" + (f" — {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────
# 1. KALMAN FILTER STRESS
# ─────────────────────────────────────────────────────────────
section("1. Kalman Filter — Numerische Stabilität (10.000 Updates)")

try:
    from osads.tracking.kalman import KalmanTracker

    tracker = KalmanTracker(initial_x=320, initial_y=240, dt=1.0 / 30)
    rng = np.random.RandomState(42)

    p_eigenvalues_min = []
    nan_detected = False
    inf_detected = False

    for i in range(10_000):
        tracker.predict()
        # Noisy measurement (realistic)
        nx = 320 + rng.normal(0, 5) + math.sin(i * 0.1) * 50
        ny = 240 + rng.normal(0, 5) + math.cos(i * 0.07) * 40
        state = tracker.update(nx, ny)

        if np.any(np.isnan(state)) or np.any(np.isnan(tracker.P)):
            nan_detected = True
            break
        if np.any(np.isinf(state)) or np.any(np.isinf(tracker.P)):
            inf_detected = True
            break

        # Check P stays positive semi-definite (all eigenvalues >= 0)
        eigs = np.linalg.eigvalsh(tracker.P)
        p_eigenvalues_min.append(float(eigs.min()))

    min_eig = min(p_eigenvalues_min) if p_eigenvalues_min else 0
    record("No NaN in state/P after 10k updates", not nan_detected)
    record("No Inf in state/P after 10k updates", not inf_detected)
    record(
        "P stays positive semi-definite (Joseph form)",
        min_eig >= -1e-9,
        f"min eigenvalue = {min_eig:.2e}",
    )
    # speed is in state units (px/sec with dt=1/30).
    # The sinusoidal target has max velocity 50*0.1 = 5 px/frame = 150 px/sec.
    # Allow up to 300 px/sec (= 10 px/frame) to account for noise.
    speed_px_per_frame = tracker.speed * tracker.dt
    record(
        "Tracking converges (speed < 10px/frame equiv.)",
        speed_px_per_frame < 10,
        f"speed={speed_px_per_frame:.2f}px/frame ({tracker.speed:.1f}px/sec)",
    )
except Exception as e:
    record("Kalman 10k stress", False, str(e))
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────
# 2. FREQUENCY ANALYZER STRESS
# ─────────────────────────────────────────────────────────────
section("2. FrequencyAnalyzer — Edge Cases & Robustheit")

try:
    from osads.detection.acoustic import FrequencyAnalyzer

    fft = FrequencyAnalyzer(sample_rate=44100, chunk_size=2048)
    rng = np.random.RandomState(0)
    SR = 44100

    def make_tone(freq, amp=0.3, sr=SR, n=2048):
        t = np.linspace(0, n / sr, n, endpoint=False)
        return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    # Silence
    r = fft.analyze(np.zeros(2048, dtype=np.float32))
    record("Silence → no detection", not r.detected, f"conf={r.confidence:.3f}")

    # Pure noise
    r = fft.analyze(rng.normal(0, 0.1, 2048).astype(np.float32))
    record("White noise → no detection", not r.detected, f"conf={r.confidence:.3f}")

    # Mosquito tone 511 Hz
    r = fft.analyze(make_tone(511))
    record("511Hz → mosquito detected", r.detected and r.insect_type == "mosquito",
           f"conf={r.confidence:.3f} freq={r.dominant_frequency:.0f}Hz")

    # Fly tone 190 Hz
    r = fft.analyze(make_tone(190))
    record("190Hz → fly detected", r.detected and r.insect_type == "fly",
           f"conf={r.confidence:.3f}")

    # 50Hz hum (loud!) — must NOT trigger
    r = fft.analyze(make_tone(50, amp=1.0))
    record("Loud 50Hz hum → no detection", not r.detected, f"conf={r.confidence:.3f}")

    # 2kHz (loud) — must NOT trigger
    r = fft.analyze(make_tone(2000, amp=0.8))
    record("2kHz tone → no detection", not r.detected, f"conf={r.confidence:.3f}")

    # Mosquito buried in loud 50Hz hum
    hum = make_tone(50, amp=0.5)
    mosq = make_tone(511, amp=0.15)
    r = fft.analyze(np.clip(hum + mosq, -1, 1))
    record(
        "Mosquito 511Hz + loud 50Hz hum — band analysis robust",
        r.detected and r.insect_type == "mosquito",
        f"conf={r.confidence:.3f} detected={r.detected} type={r.insect_type}",
    )

    # Clipped / saturated signal
    r = fft.analyze(np.ones(2048, dtype=np.float32))
    record("Saturated signal (all 1s) → no crash", True, f"detected={r.detected}")

    # Very short chunk
    r = fft.analyze(np.zeros(10, dtype=np.float32))
    record("Too-short chunk → graceful no-detection", not r.detected)

    # NaN input
    nan_input = np.full(2048, np.nan, dtype=np.float32)
    try:
        r = fft.analyze(nan_input)
        record("NaN input → no crash", True, f"detected={r.detected}")
    except Exception as e:
        record("NaN input → no crash", False, str(e))

    # 10.000 random chunks (speed + no-crash test)
    t0 = time.perf_counter()
    errors = 0
    for _ in range(10_000):
        chunk = rng.normal(0, 0.05, 2048).astype(np.float32)
        try:
            fft.analyze(chunk)
        except Exception:
            errors += 1
    elapsed = time.perf_counter() - t0
    record(
        "10k random chunks — no crashes",
        errors == 0,
        f"{elapsed*1000:.0f}ms total, {elapsed/10:.2f}ms/chunk avg",
    )

except Exception as e:
    record("FrequencyAnalyzer stress", False, str(e))
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────
# 3. MULTI-TRACKER STRESS
# ─────────────────────────────────────────────────────────────
section("3. MultiTracker — 10.000 Frames, viele Insekten")

try:
    from osads.detection.visual import Detection
    from osads.tracking.tracker import MultiTracker

    mt = MultiTracker(max_lost_frames=15, min_hits_to_confirm=3)
    rng = np.random.RandomState(7)
    arena_w, arena_h = 640, 480

    # Simulate 20 insects moving around
    positions = rng.uniform(50, [arena_w - 50, arena_h - 50], size=(20, 2))
    velocities = rng.uniform(-3, 3, size=(20, 2))

    max_history = 0
    max_tracks = 0
    nan_in_track = False

    for frame in range(10_000):
        # Move insects
        positions += velocities
        velocities += rng.normal(0, 0.2, velocities.shape)
        positions = np.clip(positions, 10, [arena_w - 10, arena_h - 10])

        # Randomly drop some detections (simulate missed frames)
        visible = rng.random(20) > 0.3
        dets = []
        for i in range(20):
            if visible[i]:
                dets.append(Detection(
                    x=int(positions[i, 0]),
                    y=int(positions[i, 1]),
                    w=6, h=4, confidence=0.7,
                    class_name="mosquito",
                ))

        tracks = mt.update(dets)

        # Check history cap
        for t in mt.tracks:
            if len(t.history) > max_history:
                max_history = len(t.history)
            pos = t.position
            if math.isnan(pos[0]) or math.isnan(pos[1]):
                nan_in_track = True

        max_tracks = max(max_tracks, len(mt.tracks))

    record("No NaN positions in tracks", not nan_in_track)
    record(
        "Track history capped at ≤100",
        max_history <= 100,
        f"max seen: {max_history}",
    )
    record(
        "Tracker handles 20 insects, 10k frames",
        True,
        f"max simultaneous tracks: {max_tracks}",
    )

except Exception as e:
    record("MultiTracker 10k stress", False, str(e))
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────
# 4. PID CONTROLLER STRESS
# ─────────────────────────────────────────────────────────────
section("4. PID Controller — Konvergenz & Anti-Windup (5.000 Frames)")

try:
    from osads.control.gimbal import PIDController, SimulatedGimbal

    gimbal = SimulatedGimbal(frame_width=640, frame_height=480)
    rng = np.random.RandomState(3)

    # Step input: target suddenly jumps to corner
    gimbal._laser_x = 320.0
    gimbal._laser_y = 240.0
    target_x, target_y = 600.0, 450.0

    errors_at = {}
    integral_overflow = False

    for frame in range(5_000):
        # Add slight target jitter (realistic)
        jitter_x = target_x + rng.normal(0, 2)
        jitter_y = target_y + rng.normal(0, 2)
        gimbal.aim_at_pixel(jitter_x, jitter_y)

        err = gimbal.tracking_error(target_x, target_y)
        if frame in (10, 50, 100, 500, 1000, 5000 - 1):
            errors_at[frame] = err

        # Check integral windup
        if abs(gimbal.pid_x._integral) > 500 or abs(gimbal.pid_y._integral) > 500:
            integral_overflow = True

    record(
        "PID converges within 50 frames (<10px)",
        errors_at.get(50, 999) < 10,
        f"err@50f={errors_at.get(50, '?'):.1f}px  err@5000f={errors_at.get(4999, '?'):.1f}px",
    )
    record("Anti-windup holds integral ≤500", not integral_overflow,
           f"ix={gimbal.pid_x._integral:.1f} iy={gimbal.pid_y._integral:.1f}")

    # Test reset
    gimbal.pid_x.reset()
    gimbal.pid_y.reset()
    record("PID reset clears state", gimbal.pid_x._integral == 0.0 and gimbal.pid_x._prev_error == 0.0)

    # Test with extreme error (full frame diagonal)
    gimbal2 = SimulatedGimbal(640, 480)
    gimbal2._laser_x = 0.0
    gimbal2._laser_y = 0.0
    for _ in range(200):
        gimbal2.aim_at_pixel(639, 479)
    err_diag = gimbal2.tracking_error(639, 479)
    record(
        "Extreme diagonal target converges (<5px in 200 frames)",
        err_diag < 5,
        f"err={err_diag:.1f}px",
    )

except Exception as e:
    record("PID stress", False, str(e))
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────
# 5. MEMORY STABILITY — FULL PIPELINE 10.000 FRAMES
# ─────────────────────────────────────────────────────────────
section("5. Memory-Stabilität — Full Pipeline 10.000 Frames")

try:
    from osads.main import OSADSPipeline

    tracemalloc.start()
    snap_start = tracemalloc.take_snapshot()

    pipeline = OSADSPipeline()
    pipeline.run(max_frames=10_000, show_gui=False)

    snap_end = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = snap_end.compare_to(snap_start, "lineno")
    # Sum of net memory growth
    total_growth_kb = sum(s.size_diff for s in stats) / 1024

    summary = pipeline.metrics.summary()

    record(
        "10k frames — no crash",
        True,
        f"hit={summary.get('hit_rate_pct', 0):.1f}%  err={summary.get('avg_tracking_error_px', 0):.1f}px",
    )
    # tracemalloc itself adds ~10-20 MB overhead; PyTorch lazily caches GPU buffers.
    # Real application memory growth is total_growth_kb minus that overhead.
    # Threshold set to 120 MB to account for both, while still catching true leaks.
    record(
        "Memory growth < 120 MB over 10k frames (incl. tracemalloc + PyTorch overhead)",
        total_growth_kb < 120_000,
        f"net growth: {total_growth_kb/1024:.1f} MB",
    )
    record(
        "PerformanceTracker buffer ≤ 9000 frames",
        len(pipeline.metrics.frames) <= 9000,
        f"buffer size: {len(pipeline.metrics.frames)}",
    )

    del pipeline
    gc.collect()

except Exception as e:
    record("Full pipeline 10k memory stress", False, str(e))
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────
# 6. MODE-SWITCH UNTER LAST
# ─────────────────────────────────────────────────────────────
section("6. Mode-Switch — Alle 3 Modi unter Dauerlast")

try:
    from osads.main import OSADSPipeline

    pipeline = OSADSPipeline()
    MODES = ["mosquito", "gnat", "fly"]
    switch_errors = 0
    results_per_mode: dict[str, dict] = {}

    for mode in MODES:
        pipeline.set_mode(mode)
        # Reset tracker and metrics for clean per-mode stats
        from osads.tracking.tracker import MultiTracker
        from osads.validation.metrics import PerformanceTracker
        pipeline.tracker = MultiTracker(max_lost_frames=15, min_hits_to_confirm=3)
        pipeline.metrics = PerformanceTracker(hit_threshold_px=10.0)
        pipeline.frame_count = 0
        pipeline.running = True

        try:
            r = pipeline.run(max_frames=3_000, show_gui=False)
            results_per_mode[mode] = r
        except Exception as e:
            switch_errors += 1
            results_per_mode[mode] = {}

    record("No mode-switch crashes", switch_errors == 0)
    for mode, r in results_per_mode.items():
        hr = r.get("hit_rate_pct", 0)
        err = r.get("avg_tracking_error_px", 999)
        record(
            f"Mode {mode}: hit>85%, err<15px",
            hr > 85 and err < 15,
            f"hit={hr:.1f}% err={err:.1f}px",
        )

except Exception as e:
    record("Mode-switch stress", False, str(e))
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────
# 7. SENSOR FUSION STRESS
# ─────────────────────────────────────────────────────────────
section("7. SensorFusion — 50.000 Fusionen")

try:
    from osads.detection.acoustic import AcousticDetection
    from osads.detection.fusion import SensorFusion
    from osads.detection.visual import Detection

    fusion = SensorFusion()
    rng = np.random.RandomState(99)
    errors = 0
    nan_conf = False

    for i in range(50_000):
        # Random visual detections
        n_dets = rng.randint(0, 5)
        vis = [
            Detection(
                x=int(rng.uniform(0, 640)),
                y=int(rng.uniform(0, 480)),
                w=int(rng.uniform(3, 20)),
                h=int(rng.uniform(3, 20)),
                confidence=float(rng.uniform(0.3, 1.0)),
                class_name=rng.choice(["mosquito", "gnat", "fly", "unknown"]),
            )
            for _ in range(n_dets)
        ]
        # Random acoustic
        acoustic = None
        if rng.random() > 0.5:
            acoustic = AcousticDetection(
                detected=bool(rng.random() > 0.3),
                insect_type=rng.choice(["mosquito", "gnat", "fly", None]),
                confidence=float(rng.uniform(0, 1)),
                dominant_frequency=float(rng.uniform(100, 1100)),
                spectrum_peak_db=float(rng.uniform(-60, 0)),
            )
        try:
            fused = fusion.fuse(vis, acoustic)
            for f in fused:
                if math.isnan(f.fused_confidence) or math.isinf(f.fused_confidence):
                    nan_conf = True
        except Exception:
            errors += 1

    record("50k fusions — no crashes", errors == 0, f"errors={errors}")
    record("No NaN/Inf in fused_confidence", not nan_conf)

except Exception as e:
    record("SensorFusion stress", False, str(e))
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────
# 8. AUDIO ML CLASSIFIERS STRESS
# ─────────────────────────────────────────────────────────────
section("8. Binary Audio Classifiers — 5.000 Inferenzen je Modus")

try:
    from osads.detection.acoustic import MelSpectrogramExtractor
    from osads.training.binary_audio_model import BinaryAudioTrainer

    mel_ext = MelSpectrogramExtractor()
    rng = np.random.RandomState(11)
    SR = 44100
    N = SR  # 1 second

    for mode in ["mosquito", "gnat", "fly"]:
        model_path = Path(f"models/{mode}_detector.pt")
        if not model_path.exists():
            record(f"{mode} model exists", False, "model file missing")
            continue

        trainer = BinaryAudioTrainer(target_insect=mode)
        trainer.load(model_path)

        errors = 0
        nan_conf = False
        t0 = time.perf_counter()

        for i in range(5_000):
            # Mix of silence, noise, tones, and insect-like signals
            choice = i % 5
            if choice == 0:
                audio = np.zeros(N, dtype=np.float32)
            elif choice == 1:
                audio = rng.normal(0, 0.05, N).astype(np.float32)
            elif choice == 2:
                t = np.linspace(0, 1, N, endpoint=False)
                f = rng.uniform(100, 2000)
                audio = (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)
            elif choice == 3:
                audio = np.ones(N, dtype=np.float32)  # saturated
            else:
                audio = rng.uniform(-1, 1, N).astype(np.float32)

            try:
                mel = mel_ext.extract(audio)
                detected, conf = trainer.predict(mel)
                if math.isnan(conf) or math.isinf(conf):
                    nan_conf = True
            except Exception:
                errors += 1

        elapsed = time.perf_counter() - t0
        record(
            f"{mode}: 5k inferences — no crash",
            errors == 0,
            f"{elapsed*1000:.0f}ms total, {elapsed/5:.2f}ms/inf avg",
        )
        record(f"{mode}: no NaN confidence", not nan_conf)

except Exception as e:
    record("Audio classifier stress", False, str(e))
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────
# 9. SIMULATION RENDER STRESS
# ─────────────────────────────────────────────────────────────
section("9. InsectSwarm Render — 10.000 Frames, 10 Insekten")

try:
    from osads.simulation.fake_insects import InsectSwarm

    swarm = InsectSwarm(640, 480, num_mosquitoes=4, num_gnats=3, num_flies=3)
    t0 = time.perf_counter()
    frame_times = []
    nan_pos = False

    for i in range(10_000):
        t_f = time.perf_counter()
        swarm.update()
        frame = swarm.render_frame()
        audio = swarm.generate_mixed_audio()
        frame_times.append(time.perf_counter() - t_f)

        for ins in swarm.insects:
            if math.isnan(ins.x) or math.isnan(ins.y):
                nan_pos = True
            if math.isnan(ins.wing_freq):
                nan_pos = True

        # Verify wall bounce
        for ins in swarm.insects:
            if not (0 <= ins.x < 640 and 0 <= ins.y < 480):
                record("Wall bounce keeps insects in bounds", False,
                       f"insect at ({ins.x:.0f},{ins.y:.0f})")
                break

        # Check audio clip
        if np.any(np.abs(audio) > 1.0):
            record("Audio clipped to [-1,1]", False, f"max={np.max(np.abs(audio)):.3f}")
            break

    elapsed = time.perf_counter() - t0
    avg_ms = np.mean(frame_times) * 1000
    p99_ms = np.percentile(frame_times, 99) * 1000

    record("10k render frames — no crash", True,
           f"avg={avg_ms:.2f}ms  p99={p99_ms:.2f}ms")
    record("No NaN insect positions/frequencies", not nan_pos)
    record(
        "Render fast enough for 30fps (avg <33ms)",
        avg_ms < 33,
        f"avg={avg_ms:.2f}ms",
    )

except Exception as e:
    record("SwarmRender 10k stress", False, str(e))
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("  STRESS TEST SUMMARY")
print(f"{'='*60}")
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
total = len(results)

for name, ok, detail in results:
    mark = PASS if ok else FAIL
    line = f"  {mark} {name}"
    if detail:
        line += f"\n       {detail}"
    print(line)

print(f"\n  Total: {passed}/{total} passed", end="")
if failed:
    print(f"  ({failed} FAILED ❌)")
else:
    print("  — alle Tests grün ✅")

sys.exit(0 if failed == 0 else 1)
