# RadarX v2 — WiFi Human Detection Radar (No Router Required)

> Single ESP32 in **SoftAP + Promiscuous** mode captures CSI reflections
> of its own beacon frames to detect human presence. No router, no second
> board, no camera.

---

## Architecture

```
ESP32 (SoftAP)
  ├── Broadcasts beacon every 100ms on CH6 ("RadarX-Net")
  ├── Promiscuous mode captures all 802.11 frames
  └── CSI callback fires per frame → CSV to Serial @ 921600

PC (Kali Linux)
  ├── csi_reader.py  — serial parser, I/Q → amplitude
  ├── processor.py   — Hampel + Savgol, turbulence, FFT, adaptive thresholds
  ├── detector.py    — frame-confirmation FSM, LED write-back
  └── radarx_ui.py   — pyqtgraph: heatmap + score + radar sweep
```

---

## Quick Start

```bash
# 1. Create venv & install
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Flash firmware (Arduino IDE, board: ESP32 Dev Module, core 3.3.10)
#    Open firmware/radarx.ino → Upload

# 3. Grant serial access
sudo usermod -aG dialout $USER && newgrp dialout

# 4. Launch dashboard
python dashboard/radarx_ui.py
```

---

## Calibration (Important)

On first launch, the system **auto-calibrates** for 50 frames with the
room assumed empty. The adaptive thresholds are:

```
empty_threshold  = baseline_mean + 2.0 × baseline_std
moving_threshold = baseline_mean + 6.0 × baseline_std
```

To recalibrate at any time: **clear the room → click [CALIBRATE]**.

---

## Dashboard Panels

| Panel | Content |
|-------|---------|
| LEFT (40%) | CSI Amplitude Heatmap — thermal colormap, 100 frames × 52 subcarriers |
| CENTER (30%) | Activity Score graph + FFT Spectrum (green=breathing, red=motion) |
| RIGHT (30%) | Radar sweep animation + status metrics + buttons |

---

## Detection States

| State | Color | Condition |
|-------|-------|-----------|
| EMPTY | 🟢 Green | score < empty_threshold for 15 frames |
| HUMAN STILL | 🟡 Yellow | score > empty_threshold + breathing power detected |
| HUMAN MOVING | 🔴 Red | score > moving_threshold for 15 frames |

---

## Serial Protocol

```
CSI_DATA,<ms>,<rssi>,<noise>,<ch>,<n_bytes>,[I0,Q0,I1,Q1,...]
HEARTBEAT,<ms>,<free_heap>
```

PC → ESP32 commands:
```
LED:ON    — illuminate built-in LED (human detected)
LED:OFF   — blink LED (empty)
REBOOT    — restart ESP32
STATUS    — print heap/uptime
```

---

## Tuning (config.json)

| Key | Default | Effect |
|-----|---------|--------|
| `window_size` | 100 | Frames in sliding window |
| `baseline_frames` | 50 | Frames used for calibration |
| `confirm_frames` | 15 | Frames before state commits |
| `empty_multiplier` | 2.0 | Sensitivity of empty threshold |
| `moving_multiplier` | 6.0 | Sensitivity of moving threshold |

**Noisy environment**: raise both multipliers.
**Very still room**: lower `empty_multiplier` to 1.5.

---

## File Reference

| File | Role |
|------|------|
| `firmware/radarx.ino` | ESP32 Arduino sketch |
| `backend/csi_reader.py` | Serial thread, I/Q parser |
| `backend/processor.py` | Signal pipeline, adaptive thresholds |
| `backend/detector.py` | State machine, LED control |
| `dashboard/radarx_ui.py` | PyQt5 + pyqtgraph UI |
| `config.json` | All runtime parameters |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| No CSI frames | Check `dmesg` for CP2102, verify `/dev/ttyUSB0` |
| Always EMPTY | Room too large — place ESP32 centrally |
| Always MOVING | Raise `moving_multiplier` or recalibrate |
| Module not found | `source venv/bin/activate` first |
| Dashboard blank | Wait 5-10 s for calibration to complete |
# RadarX
