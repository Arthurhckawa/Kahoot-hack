"""Fast screen capture using mss."""
import mss
import numpy as np
import cv2


class ScreenGrabber:
    """Grabs the primary monitor as a BGR numpy array. Designed to be cheap and reusable."""

    def __init__(self, monitor_index: int = 1):
        # mss is not thread-safe -> create one instance per thread that uses it
        self._sct = mss.mss()
        # monitor 0 = all monitors, 1 = primary
        self.monitor = self._sct.monitors[monitor_index]

    def grab(self) -> np.ndarray:
        shot = self._sct.grab(self.monitor)
        # BGRA -> BGR
        frame = np.array(shot)
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def close(self):
        try:
            self._sct.close()
        except Exception:
            pass
