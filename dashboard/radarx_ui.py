"""
RadarX v2 — Real-Time Dashboard
=================================
PyQt5 + pyqtgraph, dark theme, 3 panels + top bar.

Panel 1 (LEFT  40%): CSI Amplitude Heatmap  — ImageItem, thermal colormap
Panel 2 (CENTER 30%): Activity Score + FFT Spectrum
Panel 3 (RIGHT  30%): Radar sweep animation + Status card + Buttons
"""

import sys, os, json, math, time, queue, threading, csv
from datetime import datetime
from typing import Optional

import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QLabel, QPushButton, QFrame, QSizePolicy, QGridLayout,
)
from PyQt5.QtCore  import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui   import QFont, QColor, QPainter, QPen, QBrush, QPolygonF
from PyQt5.QtCore  import QPointF
import pyqtgraph as pg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.csi_reader import CSIReader
from backend.processor  import CSIProcessor
from backend.detector   import HumanDetector, PresenceState

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#0a0a0f"
PANEL   = "#0f0f1a"
CARD    = "#13131f"
BORDER  = "#1e1e3a"
GREEN   = "#00ff41"
CYAN    = "#00ffff"
YELLOW  = "#ffdd00"
RED     = "#ff2244"
AMBER   = "#ffaa00"
MUTED   = "#445566"
WHITE   = "#e0e8f0"

STATE_COLOR = {
    PresenceState.EMPTY:        GREEN,
    PresenceState.HUMAN_STILL:  YELLOW,
    PresenceState.HUMAN_MOVING: RED,
}

NUM_SUBCARRIERS  = 52
HEATMAP_ROWS     = 100
SCORE_HISTORY    = 100   # data points kept
REFRESH_MS       = 100   # 10 Hz

pg.setConfigOption("background", BG)
pg.setConfigOption("foreground", GREEN)


# ─────────────────────────────────────────────────────────────────────────────
# Radar sweep widget (custom QPainter)
# ─────────────────────────────────────────────────────────────────────────────

class RadarSweepWidget(QWidget):
    """Animated green-on-black radar sweep with human-presence dots."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self._angle      = 0.0          # current sweep angle (degrees)
        self._human      = False
        self._dots: list = []           # list of (angle_deg, radius_frac, age)
        self._timer      = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(33)           # ~30 fps for sweep

    def set_human(self, human: bool):
        self._human = human
        if human:
            # Place a dot near the sweep position at random radius
            r = 0.35 + np.random.random() * 0.45
            self._dots.append([self._angle, r, 1.0])

    def _tick(self):
        self._angle = (self._angle + 3.0) % 360.0  # 3°/frame = 1 rev / ~4 s
        # Decay dots
        self._dots = [[a, r, age - 0.02] for a, r, age in self._dots if age > 0]
        self.update()

    def paintEvent(self, event):
        w, h   = self.width(), self.height()
        cx, cy = w / 2, h / 2
        R      = min(cx, cy) - 8

        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Background
        p.fillRect(0, 0, w, h, QColor(BG))

        # Grid rings
        grid_pen = QPen(QColor(GREEN))
        grid_pen.setWidth(1)
        grid_pen.setStyle(Qt.DotLine)
        p.setPen(grid_pen)
        for frac in [0.33, 0.66, 1.0]:
            r = int(R * frac)
            p.drawEllipse(int(cx - r), int(cy - r), r * 2, r * 2)

        # Cross-hairs
        p.drawLine(int(cx - R), int(cy), int(cx + R), int(cy))
        p.drawLine(int(cx), int(cy - R), int(cx), int(cy + R))

        # Sweep gradient (fading arc)
        sweep_pen = QPen(QColor(GREEN))
        sweep_pen.setWidth(2)
        p.setPen(sweep_pen)

        for fade_steps in range(60):
            fade_angle = (self._angle - fade_steps * 1.0) % 360
            alpha      = int(255 * (1.0 - fade_steps / 60.0) * 0.7)
            color      = QColor(0, 255, 65, alpha)
            p.setPen(QPen(color, 1))
            rad = math.radians(fade_angle)
            p.drawLine(
                int(cx), int(cy),
                int(cx + R * math.cos(rad)),
                int(cy - R * math.sin(rad)),
            )

        # Main sweep line
        p.setPen(QPen(QColor(GREEN), 2))
        rad = math.radians(self._angle)
        p.drawLine(int(cx), int(cy),
                   int(cx + R * math.cos(rad)),
                   int(cy - R * math.sin(rad)))

        # Blip dots
        for a, r_frac, age in self._dots:
            dot_r  = R * r_frac
            dot_x  = cx + dot_r * math.cos(math.radians(a))
            dot_y  = cy - dot_r * math.sin(math.radians(a))
            alpha  = int(255 * min(age, 1.0))
            color  = QColor(255, 50, 50, alpha) if self._human else QColor(0, 255, 65, alpha)
            size   = 6
            p.setBrush(QBrush(color))
            p.setPen(Qt.NoPen)
            p.drawEllipse(int(dot_x - size/2), int(dot_y - size/2), size, size)

        p.end()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline worker
# ─────────────────────────────────────────────────────────────────────────────

class PipelineWorker(QObject):
    """Runs CSIReader → Processor → Detector in a daemon thread; emits results."""
    result_ready = pyqtSignal(object, object, object)  # det, proc, stats

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg     = cfg
        th           = cfg["thresholds"]
        self._q      = queue.Queue(maxsize=500)
        self._reader = CSIReader(port=cfg["serial_port"],
                                 baud_rate=cfg["baud_rate"],
                                 frame_queue=self._q)
        self._proc   = CSIProcessor(
            window_size       = cfg["window_size"],
            baseline_frames   = cfg["baseline_frames"],
            empty_multiplier  = th["empty_multiplier"],
            moving_multiplier = th["moving_multiplier"],
            breathing_hz      = (th["breathing_min_hz"], th["breathing_max_hz"]),
            motion_hz         = (th["motion_min_hz"],    th["motion_max_hz"]),
        )
        self._det    = HumanDetector(
            confirm_frames=cfg["confirm_frames"],
            serial_reader=self._reader,
        )
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._record_file: Optional[csv.writer] = None
        self._record_fh   = None

    def start(self):
        self._running = True
        self._reader.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._reader.stop()
        self.stop_recording()

    def recalibrate(self):
        self._proc.recalibrate()

    def start_recording(self, path: str):
        self._record_fh   = open(path, "w", newline="")
        self._record_file = csv.writer(self._record_fh)
        self._record_file.writerow(
            ["timestamp", "rssi", "noise", "activity_score",
             "spatial_turbulence", "dominant_freq", "state"]
        )

    def stop_recording(self):
        if self._record_fh:
            self._record_fh.close()
            self._record_fh   = None
            self._record_file = None

    @property
    def is_recording(self):
        return self._record_file is not None

    @property
    def reader_connected(self):
        return self._reader.is_connected

    def _loop(self):
        while self._running:
            frame = self._reader.get_frame(timeout=0.5)
            if not frame:
                continue
            proc = self._proc.push_frame(frame)
            if proc is None:
                continue
            det = self._det.update(proc)
            if self._record_file:
                self._record_file.writerow([
                    proc.timestamp, proc.rssi, proc.noise_floor,
                    round(proc.activity_score, 6),
                    round(proc.spatial_turbulence, 4),
                    round(proc.dominant_freq, 3),
                    det.state.value,
                ])
            try:
                self.result_ready.emit(det, proc, self._reader.get_stats())
            except RuntimeError:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Main window
# ─────────────────────────────────────────────────────────────────────────────

_MONO = QFont("Monospace", 9)
_MONO.setStyleHint(QFont.Monospace)


def _label(text: str, size: int = 9, bold: bool = False,
           color: str = WHITE) -> QLabel:
    lbl = QLabel(text)
    f   = QFont("Monospace", size)
    f.setStyleHint(QFont.Monospace)
    if bold:
        f.setBold(True)
    lbl.setFont(f)
    lbl.setStyleSheet(f"color:{color}; background:transparent;")
    return lbl


def _card(layout_cls=QVBoxLayout) -> tuple:
    """Return (QFrame, inner_layout) with dark card styling."""
    frame = QFrame()
    frame.setStyleSheet(f"QFrame{{background:{CARD};border:1px solid {BORDER};"
                        f"border-radius:6px;}}")
    lay = layout_cls(frame)
    lay.setContentsMargins(8, 6, 8, 6)
    lay.setSpacing(4)
    return frame, lay


class RadarXWindow(QMainWindow):

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("RadarX v2.0 — WiFi Human Detection Radar")
        self.setMinimumSize(1400, 820)
        self.setStyleSheet(f"QMainWindow,QWidget{{background:{BG};color:{WHITE};}}"
                           f"QPushButton{{background:{CARD};color:{GREEN};"
                           f"border:1px solid {GREEN};border-radius:4px;"
                           f"padding:4px 12px;font-family:Monospace;font-size:9pt;}}"
                           f"QPushButton:hover{{background:{BORDER};}}")

        # ── State ──────────────────────────────────────────────────────────────
        self._heatmap_data  = np.zeros((HEATMAP_ROWS, NUM_SUBCARRIERS), np.float32)
        self._score_history = np.zeros(SCORE_HISTORY, np.float32)
        self._latest_det    = None
        self._latest_proc   = None
        self._latest_stats  = {}
        self._lock          = threading.Lock()

        # ── Build UI ───────────────────────────────────────────────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(8)

        root.addWidget(self._build_topbar())

        body = QHBoxLayout()
        body.setSpacing(8)
        body.addWidget(self._build_heatmap_panel(),  stretch=40)
        body.addWidget(self._build_activity_panel(), stretch=30)
        body.addWidget(self._build_status_panel(),   stretch=30)
        root.addLayout(body)

        # ── Pipeline ───────────────────────────────────────────────────────────
        self._worker = PipelineWorker(cfg)
        self._worker.result_ready.connect(self._on_result)
        self._worker.start()

        # ── Refresh timer ──────────────────────────────────────────────────────
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(REFRESH_MS)

    # ── Top bar ───────────────────────────────────────────────────────────────

    def _build_topbar(self) -> QWidget:
        bar = QFrame()
        bar.setStyleSheet(f"QFrame{{background:{PANEL};border:1px solid {BORDER};"
                          f"border-radius:4px;}}")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 6, 12, 6)

        lay.addWidget(_label("RADARX V2.0", 11, bold=True, color=GREEN))
        lay.addWidget(_label("  |  ", color=MUTED))
        self.lbl_conn = _label("● CONNECTING…", color=AMBER)
        lay.addWidget(self.lbl_conn)
        lay.addStretch()
        self.lbl_clock = _label(datetime.now().strftime("%b %d %Y  %H:%M:%S"), color=MUTED)
        lay.addWidget(self.lbl_clock)

        # Clock update
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(
            lambda: self.lbl_clock.setText(datetime.now().strftime("%b %d %Y  %H:%M:%S"))
        )
        self._clock_timer.start(1000)
        return bar

    # ── Panel 1: Heatmap ──────────────────────────────────────────────────────

    def _build_heatmap_panel(self) -> QWidget:
        frame, lay = _card()
        lay.addWidget(_label("CSI AMPLITUDE MATRIX", bold=True, color=GREEN))

        pw = pg.PlotWidget()
        pw.setLabel("bottom", "SUBCARRIER INDEX", **{"color": GREEN, "font-size": "8pt"})
        pw.setLabel("left",   "TIME (FRAMES)",    **{"color": GREEN, "font-size": "8pt"})
        pw.getAxis("bottom").setTextPen(pg.mkPen(GREEN))
        pw.getAxis("left").setTextPen(pg.mkPen(GREEN))

        # Thermal colormap: black→blue→red→white
        colors = [
            (0,   0,   0,   255),
            (0,   0,   180, 255),
            (200, 0,   0,   255),
            (255, 200, 0,   255),
            (255, 255, 255, 255),
        ]
        cmap = pg.ColorMap(
            pos=np.linspace(0, 1, len(colors)),
            color=np.array(colors, dtype=np.uint8),
        )
        self._img_item = pg.ImageItem(self._heatmap_data.T)
        self._img_item.setColorMap(cmap)
        self._img_item.setLevels([0, 80])
        pw.addItem(self._img_item)
        pw.setAspectLocked(False)
        lay.addWidget(pw)
        return frame

    # ── Panel 2: Activity + Spectrum ──────────────────────────────────────────

    def _build_activity_panel(self) -> QWidget:
        frame, lay = _card()
        lay.addWidget(_label("ACTIVITY MONITOR", bold=True, color=GREEN))

        # Score graph
        score_pw = pg.PlotWidget()
        score_pw.setLabel("bottom", "TIME (FRAMES)", **{"color": GREEN, "font-size": "8pt"})
        score_pw.setLabel("left",   "ACTIVITY SCORE", **{"color": GREEN, "font-size": "8pt"})
        score_pw.setBackground(BG)
        for ax in ("bottom", "left"):
            score_pw.getAxis(ax).setTextPen(pg.mkPen(GREEN))

        self._score_curve = score_pw.plot(
            pen=pg.mkPen(CYAN, width=2), fillLevel=0,
            brush=pg.mkBrush(0, 255, 255, 30),
        )
        self._line_empty  = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen(GREEN, width=1, style=Qt.DashLine),
            label="EMPTY", labelOpts={"color": GREEN, "fill": BG},
        )
        self._line_moving = pg.InfiniteLine(
            angle=0, movable=False,
            pen=pg.mkPen(YELLOW, width=1, style=Qt.DashLine),
            label="MOVING", labelOpts={"color": YELLOW, "fill": BG},
        )
        score_pw.addItem(self._line_empty)
        score_pw.addItem(self._line_moving)
        score_pw.setMaximumHeight(220)
        lay.addWidget(score_pw)

        # Spectrum bar chart
        lay.addWidget(_label("FFT SPECTRUM  (0–3 Hz)", bold=True, color=GREEN))
        spec_pw = pg.PlotWidget()
        spec_pw.setBackground(BG)
        spec_pw.setLabel("bottom", "FREQUENCY (Hz)", **{"color": GREEN, "font-size": "8pt"})
        spec_pw.setLabel("left",   "POWER",          **{"color": GREEN, "font-size": "8pt"})
        for ax in ("bottom", "left"):
            spec_pw.getAxis(ax).setTextPen(pg.mkPen(GREEN))

        self._spec_bars = pg.BarGraphItem(x=[], height=[], width=0.05, brush=GREEN)
        spec_pw.addItem(self._spec_bars)
        spec_pw.setXRange(0, 3.0)
        lay.addWidget(spec_pw)

        return frame

    # ── Panel 3: Radar + Status + Buttons ─────────────────────────────────────

    def _build_status_panel(self) -> QWidget:
        frame, lay = _card()

        # Radar sweep
        lay.addWidget(_label("RADAR SWEEP", bold=True, color=GREEN))
        self._radar = RadarSweepWidget()
        self._radar.setMinimumHeight(200)
        lay.addWidget(self._radar)

        # Status metrics
        lay.addWidget(_label("DETECTION STATUS", bold=True, color=GREEN))
        status_frame, sgrid = _card(QGridLayout)
        status_frame.setStyleSheet(
            f"QFrame{{background:{PANEL};border:1px solid {BORDER};border-radius:4px;}}"
        )

        self.lbl_state = _label("EMPTY", 16, bold=True, color=GREEN)
        self.lbl_state.setAlignment(Qt.AlignCenter)
        sgrid.addWidget(self.lbl_state, 0, 0, 1, 2, Qt.AlignCenter)

        def row(grid, r, key, color=WHITE):
            k = _label(key + ":", color=MUTED)
            v = _label("—", color=color)
            v.setAlignment(Qt.AlignRight)
            grid.addWidget(k, r, 0)
            grid.addWidget(v, r, 1)
            return v

        self.lbl_conf  = row(sgrid, 1, "CONFIDENCE",  CYAN)
        self.lbl_score = row(sgrid, 2, "ACTIVITY",    CYAN)
        self.lbl_freq  = row(sgrid, 3, "FREQ",        CYAN)
        self.lbl_rssi  = row(sgrid, 4, "RSSI",        WHITE)
        self.lbl_snr   = row(sgrid, 5, "SNR",         WHITE)
        self.lbl_fps   = row(sgrid, 6, "FRAME RATE",  MUTED)
        self.lbl_cal   = row(sgrid, 7, "STATUS",      AMBER)

        lay.addWidget(status_frame)
        lay.addStretch()

        # Buttons
        btn_row = QHBoxLayout()
        self.btn_cal  = QPushButton("[ CALIBRATE ]")
        self.btn_rec  = QPushButton("[  RECORD   ]")
        self.btn_quit = QPushButton("[   QUIT    ]")
        self.btn_cal.clicked.connect(self._on_calibrate)
        self.btn_rec.clicked.connect(self._on_record)
        self.btn_quit.clicked.connect(self.close)
        for btn in (self.btn_cal, self.btn_rec, self.btn_quit):
            btn.setFont(_MONO)
            btn_row.addWidget(btn)
        lay.addLayout(btn_row)

        return frame

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_result(self, det, proc, stats):
        with self._lock:
            self._latest_det   = det
            self._latest_proc  = proc
            self._latest_stats = stats

    def _refresh(self):
        with self._lock:
            det   = self._latest_det
            proc  = self._latest_proc
            stats = self._latest_stats

        if det is None or proc is None:
            return

        # ── Heatmap ──────────────────────────────────────────────────────────
        self._heatmap_data[:-1] = self._heatmap_data[1:]
        self._heatmap_data[-1]  = proc.amplitude_vector
        self._img_item.setImage(self._heatmap_data.T, autoLevels=False)

        # ── Score graph ───────────────────────────────────────────────────────
        self._score_history[:-1] = self._score_history[1:]
        self._score_history[-1]  = det.activity_score
        self._score_curve.setData(self._score_history)

        if proc.calibrated:
            self._line_empty.setValue(proc.empty_threshold)
            self._line_moving.setValue(proc.moving_threshold)

        # ── Spectrum ──────────────────────────────────────────────────────────
        turb = np.array(proc.turbulence_series, dtype=np.float64)
        fps  = max(proc.frame_rate, 1.0)
        n    = len(turb)
        if n > 4:
            fft_v = np.abs(np.fft.rfft(turb))
            freqs = np.fft.rfftfreq(n, d=1.0 / fps)
            mask  = (freqs >= 0.05) & (freqs <= 3.0)
            f_roi = freqs[mask]
            p_roi = fft_v[mask]

            brushes = []
            for f in f_roi:
                if 0.1 <= f <= 0.5:
                    brushes.append(pg.mkBrush(GREEN))
                elif 0.5 < f <= 3.0:
                    brushes.append(pg.mkBrush(RED))
                else:
                    brushes.append(pg.mkBrush(MUTED))

            self._spec_bars.setOpts(x=f_roi, height=p_roi, width=0.04,
                                    brushes=brushes)

        # ── Radar ─────────────────────────────────────────────────────────────
        self._radar.set_human(det.state != PresenceState.EMPTY)

        # ── Status labels ─────────────────────────────────────────────────────
        color = STATE_COLOR.get(det.state, WHITE)
        self.lbl_state.setText(det.label)
        self.lbl_state.setStyleSheet(f"color:{color};background:transparent;")

        self.lbl_conf.setText(f"{det.confidence}%")
        self.lbl_score.setText(f"{det.activity_score:.6f}")
        self.lbl_freq.setText(f"{det.dominant_freq:.3f} HZ")
        self.lbl_rssi.setText(f"{proc.rssi} DBM")
        self.lbl_snr.setText(f"{proc.snr:.1f} DB")
        self.lbl_fps.setText(f"{proc.frame_rate:.1f} HZ")

        if proc.calibrated:
            self.lbl_cal.setText("CALIBRATED")
            self.lbl_cal.setStyleSheet(f"color:{GREEN};background:transparent;")
        else:
            n_cal = len(self._worker._proc._baseline_scores)
            n_tot = self._worker._proc.baseline_frames
            self.lbl_cal.setText(f"CALIBRATING {n_cal}/{n_tot}")

        # ── Top bar connection indicator ───────────────────────────────────────
        if stats.get("connected", False) and stats.get("fps", 0) > 0.5:
            self.lbl_conn.setText(
                f"● CONNECTED  {self.cfg['serial_port']}  "
                f"{stats.get('fps', 0):.1f} FPS"
            )
            self.lbl_conn.setStyleSheet(f"color:{GREEN};background:transparent;")
        else:
            self.lbl_conn.setText("● DISCONNECTED")
            self.lbl_conn.setStyleSheet(f"color:{RED};background:transparent;")

    # ── Buttons ───────────────────────────────────────────────────────────────

    def _on_calibrate(self):
        self._worker.recalibrate()
        self.lbl_cal.setText("RECALIBRATING…")
        self.lbl_cal.setStyleSheet(f"color:{AMBER};background:transparent;")

    def _on_record(self):
        if self._worker.is_recording:
            self._worker.stop_recording()
            self.btn_rec.setText("[  RECORD   ]")
            self.btn_rec.setStyleSheet("")
        else:
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(os.path.dirname(__file__),
                                "..", f"radarx_recording_{ts}.csv")
            self._worker.start_recording(path)
            self.btn_rec.setText("[ STOP REC  ]")
            self.btn_rec.setStyleSheet(f"color:{RED};border-color:{RED};")

    def closeEvent(self, event):
        self._timer.stop()
        self._clock_timer.stop()
        self._worker.stop()
        event.accept()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    if not os.path.exists(cfg_path):
        print(f"ERROR: config.json not found at {cfg_path}")
        sys.exit(1)
    with open(cfg_path) as f:
        cfg = json.load(f)

    app = QApplication(sys.argv)
    app.setApplicationName("RadarX")
    win = RadarXWindow(cfg)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
