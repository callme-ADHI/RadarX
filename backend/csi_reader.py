"""
RadarX v2 — CSI Serial Reader
================================
Reads raw CSI frames from ESP32 (SoftAP + Promiscuous mode) over
/dev/ttyUSB0 at 921600 baud.

Serial line format from firmware:
    CSI_DATA,<ms>,<rssi>,<noise>,<ch>,<n_bytes>,[I0,Q0,I1,Q1,...]

Parsing pipeline:
  1. Split CSV → fixed header fields + raw I/Q list
  2. Convert I/Q pairs → amplitude: amp[i] = sqrt(I[i]² + Q[i]²)
  3. Skip null subcarriers (indices 0,1,2,63 are invalid in HT20)
  4. Output numpy float32 array of shape [52]
  5. Push to thread-safe queue (drop oldest when full)
  6. Log frame rate every 5 seconds

Thread model: single daemon thread reads serial; main thread consumes queue.
"""

import threading
import queue
import time
import re
import logging
from typing import Optional

import numpy as np
import serial

logger = logging.getLogger("RadarX.CSIReader")

NUM_SUBCARRIERS = 52

# Null subcarrier indices to skip (HT20, 64-point FFT)
# Pilot subcarriers: ±7, ±21; guard/DC: 0, ±27..±32
# In the raw ESP32 output the invalid slots are typically the first 2 bytes
# and the last 2 bytes of the 64-pair buffer.  We zero-out and skip them.
NULL_INDICES = {0, 1, 63, 64}  # conservative — applied before slicing to 52


class CSIReader:
    """
    Threaded serial reader for ESP32 CSI data.

    Parameters
    ----------
    port        : serial device, default "/dev/ttyUSB0"
    baud_rate   : must match firmware (921600)
    frame_queue : shared Queue; created internally if None
    max_queue   : max frames buffered before dropping
    serial_ref  : optional reference slot; if provided, stores the
                  open Serial object so detector can write LED commands
    """

    def __init__(
        self,
        port: str = "/dev/ttyUSB0",
        baud_rate: int = 921600,
        frame_queue: Optional[queue.Queue] = None,
        max_queue: int = 500,
    ):
        self.port        = port
        self.baud_rate   = baud_rate
        self.frame_queue = frame_queue or queue.Queue(maxsize=max_queue)
        self.max_queue   = max_queue

        self._ser: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

        # Expose serial reference for LED command write-back
        self.serial_ref: Optional[serial.Serial] = None

        # Stats
        self.frames_received = 0
        self.frames_dropped  = 0
        self.parse_errors    = 0
        self._start_time     = None
        self._last_stat_log  = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._stop_evt.clear()
        self._open_serial()
        self._start_time = time.time()
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="CSIReaderThread",
            daemon=True,
        )
        self._thread.start()
        logger.info("CSIReader started on %s @ %d baud", self.port, self.baud_rate)

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        self._close_serial()
        logger.info("CSIReader stopped — rx=%d drop=%d err=%d",
                    self.frames_received, self.frames_dropped, self.parse_errors)

    def get_frame(self, timeout: float = 0.5) -> Optional[dict]:
        """Blocking read from queue. Returns None on timeout."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def send_command(self, cmd: str) -> None:
        """Write a newline-terminated command to the ESP32 (e.g. "LED:ON")."""
        if self._ser and self._ser.is_open:
            try:
                self._ser.write((cmd.strip() + "\n").encode("ascii"))
            except serial.SerialException as exc:
                logger.warning("Serial write error: %s", exc)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_connected(self) -> bool:
        return self._ser is not None and self._ser.is_open

    def get_stats(self) -> dict:
        elapsed = time.time() - self._start_time if self._start_time else 1e-9
        return {
            "frames_received": self.frames_received,
            "frames_dropped":  self.frames_dropped,
            "parse_errors":    self.parse_errors,
            "fps":             round(self.frames_received / elapsed, 2),
            "elapsed_s":       round(elapsed, 1),
            "connected":       self.is_connected,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _open_serial(self, retries: int = 10) -> None:
        for attempt in range(retries):
            try:
                self._ser = serial.Serial(
                    port=self.port,
                    baudrate=self.baud_rate,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=1.0,
                )
                self._ser.reset_input_buffer()
                self.serial_ref = self._ser
                logger.info("Serial opened: %s (attempt %d)", self.port, attempt + 1)
                return
            except serial.SerialException as exc:
                logger.warning("Serial open failed [%d/%d]: %s", attempt + 1, retries, exc)
                time.sleep(2.0)
        raise serial.SerialException(f"Cannot open {self.port} after {retries} attempts")

    def _close_serial(self) -> None:
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None
        self.serial_ref = None

    def _reader_loop(self) -> None:
        while not self._stop_evt.is_set():
            # --- Read one line ---
            try:
                raw = self._ser.readline()
            except serial.SerialException as exc:
                if not self._stop_evt.is_set():
                    logger.error("Serial read error: %s — reconnecting in 3 s", exc)
                    self._close_serial()
                    time.sleep(3.0)
                    try:
                        self._open_serial()
                    except serial.SerialException:
                        pass
                continue

            if not raw:
                continue

            try:
                line = raw.decode("ascii", errors="replace").strip()
            except Exception:
                self.parse_errors += 1
                continue

            if not line.startswith("CSI_DATA"):
                logger.debug("ESP32 >> %s", line)
                continue

            frame = self._parse_line(line)
            if frame is None:
                self.parse_errors += 1
                continue

            self.frames_received += 1
            self._enqueue(frame)

            # Log stats every 5 seconds
            now = time.time()
            if now - self._last_stat_log >= 5.0:
                self._last_stat_log = now
                stats = self.get_stats()
                logger.info("CSI stats: fps=%.1f rx=%d drop=%d err=%d",
                            stats["fps"], stats["frames_received"],
                            stats["frames_dropped"], stats["parse_errors"])

    # ── Parsing ───────────────────────────────────────────────────────────────

    # Pre-compiled regex for the bracket-delimited I/Q list
    _IQ_RE = re.compile(r"\[([^\]]*)\]")

    def _parse_line(self, line: str) -> Optional[dict]:
        """
        Parse one CSI_DATA CSV line.

        Format:
          CSI_DATA,<ms>,<rssi>,<noise>,<ch>,<n>,[I0,Q0,...,In,Qn]

        Returns a frame dict or None on parse failure.
        """
        try:
            # Split on first '[' to isolate the header and the I/Q block
            bracket_match = self._IQ_RE.search(line)
            if not bracket_match:
                return None

            header_part = line[:bracket_match.start()]
            iq_part     = bracket_match.group(1)

            # Parse header CSV fields
            fields = header_part.rstrip(",").split(",")
            if len(fields) < 6:
                return None

            timestamp   = int(fields[1])
            rssi        = int(fields[2])
            noise_floor = int(fields[3])
            channel     = int(fields[4])
            n_declared  = int(fields[5])

            # Parse I/Q integers
            if not iq_part.strip():
                return None
            raw_iq = np.array([int(x) for x in iq_part.split(",")], dtype=np.int8)

        except (ValueError, IndexError) as exc:
            logger.debug("Parse error: %s — %s", exc, line[:80])
            return None

        amplitudes = self._iq_to_amplitude(raw_iq)

        return {
            "timestamp":   timestamp,
            "rssi":        rssi,
            "noise_floor": noise_floor,
            "channel":     channel,
            "n_declared":  n_declared,
            "amplitudes":  amplitudes,
        }

    @staticmethod
    def _iq_to_amplitude(raw_iq: np.ndarray) -> np.ndarray:
        """
        Convert interleaved signed I/Q bytes to amplitude vector.

        Layout: [I0, Q0, I1, Q1, ..., In, Qn]
        Amplitude_k = sqrt(I_k² + Q_k²)

        Null subcarriers (first 2 and last 2 pairs) are zeroed.
        Returns float32 array of shape (NUM_SUBCARRIERS,).
        """
        n_pairs = len(raw_iq) // 2
        I = raw_iq[0::2].astype(np.float32)
        Q = raw_iq[1::2].astype(np.float32)
        amp = np.sqrt(I**2 + Q**2)

        # Skip known-null slots at head and tail
        if len(amp) > 4:
            amp[0] = 0.0
            amp[1] = 0.0
            amp[-1] = 0.0
            amp[-2] = 0.0

        # Pad or trim to NUM_SUBCARRIERS
        result = np.zeros(NUM_SUBCARRIERS, dtype=np.float32)
        count  = min(n_pairs, NUM_SUBCARRIERS)
        result[:count] = amp[:count]
        return result

    def _enqueue(self, frame: dict) -> None:
        """Push frame; drop oldest if full (real-time priority)."""
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
                self.frames_dropped += 1
            except queue.Empty:
                pass
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            self.frames_dropped += 1


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, os, sys

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    reader = CSIReader(port=cfg["serial_port"], baud_rate=cfg["baud_rate"])
    try:
        reader.start()
        print("Reading CSI — Ctrl-C to stop\n")
        while True:
            frame = reader.get_frame(timeout=1.0)
            if frame:
                a = frame["amplitudes"]
                print(f"ts={frame['timestamp']:10d}  rssi={frame['rssi']:4d}  "
                      f"amp_mean={a.mean():.2f}  amp_max={a.max():.2f}")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        reader.stop()
