"""
RadarX v2 — CSI Signal Processor
====================================
Implements the full signal processing pipeline described in the RadarX spec:

Per-frame:
  1. Hampel filter  — spike removal per subcarrier (median ± 3·MAD)
  2. Savitzky-Golay — smooth across subcarriers (window=11, poly=3)
  3. Spatial turbulence = std(amp_smooth)  [scalar per frame]

Sliding window [N=100 frames]:
  4. activity_score = var(turbulence_series)
  5. FFT on turbulence series → dominant_freq, breathing_power, motion_power

Adaptive thresholds:
  6. Calibrate on first 50 frames (empty room) → compute dynamic thresholds
  7. Expose recalibrate() for UI button

Output dict per frame (see ProcessorOutput).
"""

import time
import logging
from collections import deque
from typing import Optional

import numpy as np
from scipy.signal import savgol_filter

logger = logging.getLogger("RadarX.Processor")

NUM_SUBCARRIERS = 52


class ProcessorOutput:
    """Container for one processed frame's metrics."""

    __slots__ = (
        "timestamp", "activity_score", "spatial_turbulence",
        "dominant_freq", "breathing_power", "motion_power",
        "amplitude_vector", "turbulence_series",
        "rssi", "noise_floor", "snr", "frame_rate",
        "calibrated", "empty_threshold", "moving_threshold",
    )

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def to_dict(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__
                if not isinstance(getattr(self, s), np.ndarray)}


class CSIProcessor:
    """
    Stateful signal processor with adaptive thresholds.

    Parameters
    ----------
    window_size      : frames in sliding window (default 100)
    baseline_frames  : frames used for calibration (default 50)
    sample_rate      : nominal fps from ESP32 (default 10)
    empty_multiplier : baseline_mean + M * baseline_std = empty_threshold
    moving_multiplier: baseline_mean + M * baseline_std = moving_threshold
    breathing_hz     : (low, high) Hz tuple
    motion_hz        : (low, high) Hz tuple
    """

    def __init__(
        self,
        window_size:       int   = 100,
        baseline_frames:   int   = 50,
        sample_rate:       float = 10.0,
        empty_multiplier:  float = 2.0,
        moving_multiplier: float = 6.0,
        breathing_hz:      tuple = (0.1, 0.5),
        motion_hz:         tuple = (0.5, 3.0),
    ):
        self.window_size       = window_size
        self.baseline_frames   = baseline_frames
        self.sample_rate       = sample_rate
        self.empty_mult        = empty_multiplier
        self.moving_mult       = moving_multiplier
        self.breathing_hz      = breathing_hz
        self.motion_hz         = motion_hz

        # Circular buffer: amplitude rows
        self._amp_buffer: deque = deque(maxlen=window_size)
        # Scalar turbulence per frame
        self._turb_buffer: deque = deque(maxlen=window_size)
        # Timestamp ring for fps estimation
        self._ts_ring: deque = deque(maxlen=30)

        # Calibration state
        self._baseline_scores: list = []
        self._calibrated:      bool = False
        self.empty_threshold:  float = 0.0
        self.moving_threshold: float = 0.0

        self._last_output: Optional[ProcessorOutput] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def push_frame(self, frame: dict) -> Optional[ProcessorOutput]:
        """
        Accept one CSIReader frame dict; return ProcessorOutput or None
        until the buffer is at least half full.
        """
        amp        = frame["amplitudes"]   # float32[52]
        rssi       = frame["rssi"]
        noise      = frame["noise_floor"]
        timestamp  = frame["timestamp"]

        self._ts_ring.append(time.monotonic())

        # ── Step 1 & 2: filter + smooth amplitude ────────────────────────────
        amp_filtered = self._hampel_filter(amp)
        amp_smooth   = self._savgol_smooth(amp_filtered)

        # ── Step 3: spatial turbulence ───────────────────────────────────────
        spatial_turb = float(np.std(amp_smooth))

        self._amp_buffer.append(amp_smooth)
        self._turb_buffer.append(spatial_turb)

        # Need at least half the window
        if len(self._turb_buffer) < max(10, self.window_size // 2):
            return None

        # ── Steps 4–5: window statistics ─────────────────────────────────────
        turb_arr = np.array(self._turb_buffer, dtype=np.float64)
        activity_score = float(np.var(turb_arr))

        fps = self._estimate_fps()
        dominant_freq, breathing_power, motion_power = self._spectral_analysis(
            turb_arr, fps
        )

        # ── Step 6: adaptive threshold calibration ───────────────────────────
        if not self._calibrated:
            self._baseline_scores.append(activity_score)
            if len(self._baseline_scores) >= self.baseline_frames:
                self._compute_thresholds()

        snr = float(rssi - noise)

        output = ProcessorOutput(
            timestamp          = timestamp,
            activity_score     = activity_score,
            spatial_turbulence = spatial_turb,
            dominant_freq      = dominant_freq,
            breathing_power    = breathing_power,
            motion_power       = motion_power,
            amplitude_vector   = amp_smooth.astype(np.float32),
            turbulence_series  = list(turb_arr),
            rssi               = rssi,
            noise_floor        = noise,
            snr                = snr,
            frame_rate         = fps,
            calibrated         = self._calibrated,
            empty_threshold    = self.empty_threshold,
            moving_threshold   = self.moving_threshold,
        )
        self._last_output = output
        return output

    def recalibrate(self) -> None:
        """Reset adaptive thresholds — call when room is empty."""
        self._baseline_scores.clear()
        self._calibrated      = False
        self.empty_threshold  = 0.0
        self.moving_threshold = 0.0
        logger.info("Recalibration started — keep room empty for %d frames",
                    self.baseline_frames)

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def last_output(self) -> Optional[ProcessorOutput]:
        return self._last_output

    @property
    def buffer_fill(self) -> float:
        return len(self._turb_buffer) / self.window_size

    # ── Signal processing helpers ─────────────────────────────────────────────

    @staticmethod
    def _hampel_filter(amp: np.ndarray, k: int = 3, n_sigmas: float = 3.0) -> np.ndarray:
        """
        Hampel identifier: for each sample, if |x - median(neighbors)| > n_sigmas * MAD,
        replace with the local median.  Removes impulsive spike outliers.

        Parameters
        ----------
        amp      : input amplitude array [52]
        k        : half-window size (total window = 2k+1)
        n_sigmas : detection threshold in MAD units
        """
        result = amp.copy()
        n = len(amp)
        for i in range(n):
            lo = max(0, i - k)
            hi = min(n, i + k + 1)
            window = amp[lo:hi]
            med = np.median(window)
            mad = np.median(np.abs(window - med))
            if mad > 0 and abs(amp[i] - med) > n_sigmas * 1.4826 * mad:
                result[i] = med
        return result

    @staticmethod
    def _savgol_smooth(amp: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
        """
        Savitzky-Golay smoothing across the subcarrier axis.
        Preserves spectral shape while suppressing high-frequency noise.
        Falls back gracefully if array is too short.
        """
        if len(amp) < window:
            return amp.copy()
        # window must be odd
        w = window if window % 2 == 1 else window + 1
        return savgol_filter(amp.astype(np.float64), w, poly).astype(np.float32)

    def _spectral_analysis(
        self, turb_series: np.ndarray, fps: float
    ) -> tuple:
        """
        Run rfft on turbulence time series.
        Returns (dominant_freq_hz, breathing_power, motion_power).
        """
        n = len(turb_series)
        fft_vals = np.abs(np.fft.rfft(turb_series))
        freqs    = np.fft.rfftfreq(n, d=1.0 / max(fps, 1.0))

        # Skip DC bin (index 0)
        if len(fft_vals) <= 1:
            return 0.0, 0.0, 0.0

        fft_no_dc = fft_vals[1:]
        freqs_no_dc = freqs[1:]

        dominant_freq = float(freqs_no_dc[np.argmax(fft_no_dc)]) if len(fft_no_dc) else 0.0

        b_lo, b_hi = self.breathing_hz
        m_lo, m_hi = self.motion_hz

        b_mask = (freqs >= b_lo) & (freqs <= b_hi)
        m_mask = (freqs >= m_lo) & (freqs <= m_hi)

        breathing_power = float(np.sum(fft_vals[b_mask]))
        motion_power    = float(np.sum(fft_vals[m_mask]))

        return dominant_freq, breathing_power, motion_power

    def _estimate_fps(self) -> float:
        ring = list(self._ts_ring)
        if len(ring) < 2:
            return self.sample_rate
        elapsed = ring[-1] - ring[0]
        if elapsed <= 0:
            return self.sample_rate
        return float(np.clip((len(ring) - 1) / elapsed, 1.0, 100.0))

    def _compute_thresholds(self) -> None:
        arr  = np.array(self._baseline_scores)
        mean = float(np.mean(arr))
        std  = float(np.std(arr))
        self.empty_threshold  = mean + self.empty_mult  * std
        self.moving_threshold = mean + self.moving_mult * std
        self._calibrated = True
        logger.info(
            "Calibrated — baseline_mean=%.6f std=%.6f "
            "empty_thresh=%.6f moving_thresh=%.6f",
            mean, std, self.empty_threshold, self.moving_threshold,
        )


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from csi_reader import CSIReader

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    th = cfg["thresholds"]
    reader = CSIReader(port=cfg["serial_port"], baud_rate=cfg["baud_rate"])
    proc   = CSIProcessor(
        window_size=cfg["window_size"],
        baseline_frames=cfg["baseline_frames"],
        empty_multiplier=th["empty_multiplier"],
        moving_multiplier=th["moving_multiplier"],
    )

    reader.start()
    print("Processing CSI — Ctrl-C to stop\n")
    try:
        while True:
            frame = reader.get_frame(timeout=1.0)
            if frame is None:
                continue
            out = proc.push_frame(frame)
            if out:
                cal = "CAL" if out.calibrated else f"CAL({len(proc._baseline_scores)}/{proc.baseline_frames})"
                print(f"[{cal}] score={out.activity_score:.6f}  "
                      f"turb={out.spatial_turbulence:.3f}  "
                      f"freq={out.dominant_freq:.3f}Hz  "
                      f"fps={out.frame_rate:.1f}")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        reader.stop()
