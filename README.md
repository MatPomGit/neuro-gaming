# NeuroGaming – Muse S Athena Controller

A Python / [Kivy](https://kivy.org) mobile application that lets you control
a game using EEG signals from the
[Muse S Athena](https://choosemuse.com/muse-s/) headband via Bluetooth Low
Energy (BLE).  The app also supports conventional keyboard/touch input
(arrow keys **or** WASD) as a fallback.

---

## Features

| Feature | Details |
|---|---|
| **EEG-driven control** | Alpha- and beta-band power → FORWARD / BACKWARD / LEFT / RIGHT |
| **Muse S Athena BLE** | Direct BLE connection via [`bleak`](https://bleak.readthedocs.io) |
| **Keyboard fallback** | Arrow keys **and** WASD supported simultaneously |
| **Live visualisation** | Real-time alpha / beta band-power bars per channel |
| **Calibration** | Per-session baseline recording for better classification |
| **Android APK** | Built with [Buildozer](https://buildozer.readthedocs.io); CI via GitHub Actions |

---

## Control Mapping

| EEG Condition | Command | Arrow | WASD |
|---|---|---|---|
| High **beta** power (AF7 + AF8) – concentration | FORWARD | ↑ | W |
| High **alpha** power (AF7 + AF8) – relaxation | BACKWARD | ↓ | S |
| AF7 alpha > AF8 alpha × 1.3 – left asymmetry | LEFT | ← | A |
| AF8 alpha > AF7 alpha × 1.3 – right asymmetry | RIGHT | → | D |

---

## Project Structure

```
neuro-gaming/
├── main.py                   # Kivy application entry point
├── neuro_gaming.kv           # KV layout (screens, widgets)
├── src/
│   ├── __init__.py
│   ├── muse_connector.py     # BLE scanner + EEG packet parser
│   ├── signal_processor.py   # FFT-based alpha/beta computation
│   └── game_controller.py    # Direction state machine + keyboard handler
├── tests/
│   ├── test_core.py          # Unit tests for processor and controller
│   └── test_muse_parser.py   # Unit tests for the BLE packet parser
├── buildozer.spec            # Android build configuration
├── requirements.txt          # Python dependencies
└── .github/
    └── workflows/
        └── build-apk.yml     # Manual APK build workflow
```

---

## Getting Started (Desktop / Development)

### Prerequisites

* Python 3.10+
* A Bluetooth adapter (for Muse connection)

### Installation

```bash
pip install -r requirements.txt
```

### Running

```bash
python main.py
```

On the **Scan** screen:

1. Press **Scan for Muse Devices** – the app searches for nearby Muse
   headbands over BLE.
2. Tap a device in the list to connect and start EEG streaming.
3. Alternatively, press **Continue in keyboard-only mode** to use arrow
   keys / WASD without a headband.

On the **Game** screen:

* The directional arrow that lights up blue shows the current command.
* Press **Calibrate** and sit still for 5–10 seconds so the app can learn
  your individual baseline.  Press **Stop Calibration** when done.
* Toggle between **arrow** and **wasd** key modes with the **Mode** button.

---

## Running Tests

```bash
pip install pytest numpy
pytest tests/ -v
```

---

## Building the Android APK

### Locally (requires Linux)

```bash
pip install buildozer cython
buildozer android debug
```

The APK is placed in the `bin/` directory.

### Via GitHub Actions (manual trigger)

1. Go to the **Actions** tab of this repository.
2. Select **Build Android APK**.
3. Click **Run workflow**, choose `debug` or `release`, and press
   **Run workflow**.
4. Once the workflow completes, download the APK from the **Artifacts**
   section of the run.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│              Kivy Main Thread               │
│  ScanScreen ──► GameScreen (100 ms clock)  │
└──────────────────┬──────────────────────────┘
                   │ add_samples() / get_direction()
┌──────────────────▼──────────────────────────┐
│           SignalProcessor                   │
│  rolling buffers │ FFT │ band power │ logic │
└──────────────────▲──────────────────────────┘
                   │ add_samples()
┌──────────────────┴──────────────────────────┐
│       MuseConnector (background thread)     │
│  asyncio event loop │ bleak BLE client      │
└─────────────────────────────────────────────┘
```

---

## License

MIT
