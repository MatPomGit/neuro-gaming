# NeuroGaming – Muse S Athena Controller

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://github.com/MatPomGit/neuro-gaming/actions/workflows/tests.yml/badge.svg)](https://github.com/MatPomGit/neuro-gaming/actions/workflows/tests.yml)
[![Build APK](https://github.com/MatPomGit/neuro-gaming/actions/workflows/build-apk.yml/badge.svg)](https://github.com/MatPomGit/neuro-gaming/actions/workflows/build-apk.yml)

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
| **Signal quality** | Per-channel electrode contact quality (TP9 / AF7 / AF8 / TP10) |
| **Calibration** | Per-session baseline recording for better signal classification |
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
        ├── tests.yml         # Automated test CI (runs on every push / PR)
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

Replay zapisanej sesji (zastępuje live BLE źródłem z pliku):

```bash
python main.py --replay sessions/example.session.jsonl
```

On Windows you can also use:

```bat
run_neuro_gaming.bat
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

## Session replay format (technical)

Sesje są zapisywane jako **JSON Lines** (`*.session.jsonl`):

1. Pierwsza linia to nagłówek:
   * `type = "session_header"`
   * `format_version` (aktualnie `1.0`)
   * metadane sesji (`device_model`, `active_channels`, `threshold_config`, `app_version`, itp.)
2. Kolejne linie to zdarzenia czasowe z polem `t` (sekundy od startu):
   * `type = "eeg"` – ramki EEG (`channel`, `samples`)
   * `type = "imu"` – ramki IMU (`sensor`, `samples`)
   * `type = "ppg"` – ramki PPG (`channel`, `samples`)
   * `type = "control_event"` – zdarzenia sterowania z UI/klawiatury/myszy
   * `type = "direction_decision"` – decyzja algorytmu z metrykami i confidence

### Version compatibility

* Wersja `1.x` gwarantuje kompatybilność wsteczną w obrębie pola `type` i znaczenia `t`.
* Nowe pola w zdarzeniach mogą być dodawane, ale istniejące pola nie powinny zmieniać semantyki.
* Loader replayu ignoruje nieznane pola, co pozwala na bezpieczne rozszerzanie formatu.

### Post-session report

Dla każdej sesji można wyeksportować raport `*.report.json`, który zawiera:

* `average_latency_ms` i `max_latency_ms`,
* `direction_stability` (0–1),
* `rejected_windows`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| No Muse devices found | BLE adapter off or no permission | Enable Bluetooth; grant Location permission on Android |
| Signal quality stays red | Electrode not touching skin | Adjust headband; ensure good scalp contact |
| Direction stays "Awaiting signal…" | Buffer not yet full (< 1 s) | Wait ~1 s after connecting before using the app |
| Erratic directions | No calibration | Press **Calibrate**, sit still for 5–10 s, then **Stop Calibration** |

---

## Muse S connection resilience

The connector includes a dedicated connection-state machine with explicit
states:

* `IDLE`
* `SCANNING`
* `CONNECTING`
* `STREAMING`
* `RECOVERING`
* `ERROR`

During normal operation, the EEG stream is supervised by a watchdog. If
the app does not receive EEG samples for a configured timeout window, it
automatically enters `RECOVERING` and starts a reconnect sequence with
progressive backoff (1s → 2s → 4s → 8s by default).

Reconnect progress is reported to the UI through status updates, including:

* current connection state,
* reconnect attempt number,
* success/failure information.

At disconnect, the app logs session-level transport metrics for diagnostics:

* total connection duration,
* reconnect count,
* average interval between incoming sample callbacks,
* estimated dropout percentage based on EEG packet sequence gaps.

This behavior is designed to make temporary BLE instability less disruptive
in real-world use (movement, RF interference, or transient signal loss).

---

## Contributing

1. Fork the repository and create a feature branch.
2. Install development dependencies: `pip install -r requirements.txt`
3. Run the test suite before and after your changes: `pytest tests/ -v`
4. Open a pull request with a clear description of the change.

---

## License

MIT
