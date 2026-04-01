# OSADS - Open Source Air Defense System

Autonomous insect detection, tracking, and laser targeting system.
Designed as a mosquito-scale prototype, architectured to scale to drone defense.

## What it does

1. **Detects** insects via camera (motion detection) and microphone (wing-beat frequency analysis)
2. **Classifies** the insect type using a trained neural network (CNN on Mel spectrograms)
3. **Tracks** the target using a Kalman filter with trajectory prediction
4. **Aims** a laser at the target using a PID-controlled gimbal
5. **Validates** hits and logs performance metrics

## Modes

Three dedicated detection modes, each with its own binary classifier:

| Mode | Target | Frequency Range | Key |
|------|--------|----------------|-----|
| Mosquito | Aedes, Anopheles, Culex | 350-1000 Hz | `1` |
| Gnat/Muecke | Chironomidae midges | 500-1050 Hz | `2` |
| Fly/Fliege | Musca domestica | 100-300 Hz | `3` |

Frequency data verified from: Arthur et al. (PMC3985972), UF Insect Records, Rockstein & Bhatnagar 1966.

## Quick Start

```bash
# Install dependencies
pip install numpy opencv-python scipy torch torchaudio PyYAML scikit-learn

# Run tests
PYTHONPATH=src python -m pytest tests/ -v

# Train audio classifiers (all 3 modes)
PYTHONPATH=src python train_modes.py --test

# Run simulation (opens GUI window)
PYTHONPATH=src python -m osads.main

# Run headless (no GUI)
PYTHONPATH=src python -c "
from osads.main import OSADSPipeline
p = OSADSPipeline()
result = p.run(max_frames=1000, show_gui=False)
"
```

## GUI Controls

| Key | Action |
|-----|--------|
| `1` | Switch to Mosquito mode |
| `2` | Switch to Gnat mode |
| `3` | Switch to Fly mode |
| `s` | Print performance stats |
| `q` | Quit |

## Performance (Simulation)

| Metric | Value |
|--------|-------|
| Hit Rate | 90%+ |
| Tracking Error | <8 px |
| Latency | <1 ms |
| Laser Active | 99%+ |

## Architecture

```
Sensors → Detection → Tracking → Aiming → Validation
  |           |           |          |          |
Camera    Motion+ML    Kalman    PID+Gimbal   Metrics
Mic       FFT+CNN     Filter    Laser Ctrl   Hit Rate
```

The laser is a **modular, swappable effector**. The entire intelligence
(detection, tracking, targeting) is independent of laser power.
Replace the test laser with a high-power laser = ready for drone defense.

## Project Structure

```
osads/
├── src/osads/
│   ├── detection/     # Visual + acoustic + sensor fusion
│   ├── tracking/      # Kalman filter + multi-target tracker
│   ├── control/       # PID gimbal + laser control
│   ├── validation/    # Hit metrics + performance tracking
│   ├── simulation/    # Fake insects for laptop testing
│   ├── training/      # Audio CNN + binary classifier training
│   └── main.py        # Main pipeline
├── config/            # YAML configuration
├── models/            # Trained ML models
├── tests/             # pytest test suite
├── train_modes.py     # Per-mode training script
└── train_audio.py     # Multi-class training script
```

## License

MIT
