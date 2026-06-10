"""
RadarX v2 — Human Presence Detector
=======================================
Frame-confirmation state machine that consumes ProcessorOutput objects
and outputs a stable presence classification.

States:   EMPTY | HUMAN_STILL | HUMAN_MOVING
Hysteresis: require CONFIRM_FRAMES consecutive agreeing frames before
            committing to a new state (prevents flicker).

Also writes LED:ON / LED:OFF commands to the ESP32 over serial.
"""

import time
import logging
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger("RadarX.Detector")


class PresenceState(Enum):
    EMPTY        = "EMPTY"
    HUMAN_STILL  = "HUMAN_STILL"
    HUMAN_MOVING = "HUMAN_MOVING"


class DetectorOutput:
    """Snapshot of one detector tick."""
    __slots__ = ("state", "confidence", "activity_score",
                 "dominant_freq", "timestamp", "state_age_s")

    def __init__(self, state, confidence, activity_score,
                 dominant_freq, timestamp, state_age_s):
        self.state          = state
        self.confidence     = confidence      # 0–100 int
        self.activity_score = activity_score
        self.dominant_freq  = dominant_freq
        self.timestamp      = timestamp
        self.state_age_s    = state_age_s

    @property
    def label(self) -> str:
        return {
            PresenceState.EMPTY:        "EMPTY",
            PresenceState.HUMAN_STILL:  "HUMAN STILL",
            PresenceState.HUMAN_MOVING: "HUMAN MOVING",
        }.get(self.state, self.state.value)

    def to_dict(self) -> dict:
        return {
            "state":          self.state.value,
            "label":          self.label,
            "confidence":     self.confidence,
            "activity_score": round(self.activity_score, 6),
            "dominant_freq":  round(self.dominant_freq, 3),
            "timestamp":      self.timestamp,
            "state_age_s":    round(self.state_age_s, 1),
        }


class HumanDetector:
    """
    Frame-confirmation state machine.

    Parameters
    ----------
    confirm_frames      : consecutive frames needed to confirm a new state
    breathing_threshold : breathing_power threshold for HUMAN_STILL
    serial_reader       : CSIReader instance (for LED write-back), optional
    """

    def __init__(
        self,
        confirm_frames:      int   = 15,
        breathing_threshold: float = 5.0,
        serial_reader=None,
    ):
        self.confirm_frames      = confirm_frames
        self.breathing_threshold = breathing_threshold
        self._reader             = serial_reader

        self._state              = PresenceState.EMPTY
        self._state_entry_time   = time.monotonic()
        self._candidate          = PresenceState.EMPTY
        self._confirm_counter    = 0
        self._last_led_state     = None

        self._last_output: Optional[DetectorOutput] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, proc: "ProcessorOutput") -> DetectorOutput:
        """
        Consume one ProcessorOutput; return DetectorOutput.
        Requires proc.calibrated == True for non-EMPTY states.
        """
        score   = proc.activity_score
        b_power = proc.breathing_power
        now     = time.monotonic()

        # ── Determine raw candidate ───────────────────────────────────────────
        if proc.calibrated:
            if score > proc.moving_threshold:
                raw_candidate = PresenceState.HUMAN_MOVING
            elif score > proc.empty_threshold and b_power > self.breathing_threshold:
                raw_candidate = PresenceState.HUMAN_STILL
            else:
                raw_candidate = PresenceState.EMPTY
        else:
            # During calibration, stay EMPTY
            raw_candidate = PresenceState.EMPTY

        # ── Confirmation counter ──────────────────────────────────────────────
        if raw_candidate == self._candidate:
            self._confirm_counter += 1
        else:
            self._candidate       = raw_candidate
            self._confirm_counter = 1

        # ── Commit transition ─────────────────────────────────────────────────
        if self._confirm_counter >= self.confirm_frames:
            if self._candidate != self._state:
                logger.info("State: %s → %s (score=%.6f)",
                            self._state.value, self._candidate.value, score)
                self._state          = self._candidate
                self._state_entry_time = now
            self._confirm_counter = 0

        # ── Confidence ────────────────────────────────────────────────────────
        confidence = min(100, int((self._confirm_counter / self.confirm_frames) * 100))

        # ── LED write-back ────────────────────────────────────────────────────
        self._update_led()

        state_age = now - self._state_entry_time

        output = DetectorOutput(
            state          = self._state,
            confidence     = confidence,
            activity_score = score,
            dominant_freq  = proc.dominant_freq,
            timestamp      = proc.timestamp,
            state_age_s    = state_age,
        )
        self._last_output = output
        return output

    def set_reader(self, reader) -> None:
        """Attach a CSIReader for LED write-back."""
        self._reader = reader

    def set_breathing_threshold(self, val: float) -> None:
        self.breathing_threshold = val

    @property
    def current_state(self) -> PresenceState:
        return self._state

    @property
    def last_output(self) -> Optional[DetectorOutput]:
        return self._last_output

    # ── LED ───────────────────────────────────────────────────────────────────

    def _update_led(self) -> None:
        """Send LED command only when state changes to avoid serial spam."""
        human_present = (self._state != PresenceState.EMPTY)
        if human_present == self._last_led_state:
            return
        self._last_led_state = human_present
        cmd = "LED:ON" if human_present else "LED:OFF"
        if self._reader is not None:
            try:
                self._reader.send_command(cmd)
            except Exception as exc:
                logger.debug("LED write error: %s", exc)
        logger.info("LED: %s", cmd)


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from csi_reader import CSIReader
    from processor  import CSIProcessor

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    th = cfg["thresholds"]
    reader   = CSIReader(port=cfg["serial_port"], baud_rate=cfg["baud_rate"])
    proc     = CSIProcessor(
        window_size=cfg["window_size"],
        baseline_frames=cfg["baseline_frames"],
        empty_multiplier=th["empty_multiplier"],
        moving_multiplier=th["moving_multiplier"],
    )
    detector = HumanDetector(
        confirm_frames=cfg["confirm_frames"],
        serial_reader=reader,
    )

    reader.start()
    print("Detecting presence — Ctrl-C to stop\n")
    try:
        while True:
            frame = reader.get_frame(timeout=1.0)
            if frame is None:
                continue
            out = proc.push_frame(frame)
            if out is None:
                continue
            det = detector.update(out)
            bar = "█" * (det.confidence // 5)
            print(f"[{det.label:<14}] {det.confidence:3d}% |{bar:<20}| "
                  f"score={det.activity_score:.6f}  "
                  f"freq={det.dominant_freq:.3f}Hz")
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        reader.stop()
