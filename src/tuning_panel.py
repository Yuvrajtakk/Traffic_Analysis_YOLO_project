"""
src/tuning_panel.py

Live trackbar control panel for interactively tuning every numeric
threshold in config/live_config.py while watching the RTSP feed —
built for exactly the "adjust a threshold and see the effect on the
live feed immediately" workflow requested for this phase.

cv2 trackbars only support NON-NEGATIVE INTEGERS. Every float or
negative-range value below is stored as a scaled integer on the
trackbar itself, and converted back to its real value inside the
callback before being written into the shared LiveConfig object. The
mapping for each knob is documented inline at its creation.

Because cv2 trackbar callbacks fire synchronously (during the same
cv2.waitKey()/cv2.imshow() processing main.py already calls every
frame), there's no threading or locking concern here — the callback
runs on the same thread as the main loop, same as any other cv2 GUI
event.

Usage (from main.py):
    from src.tuning_panel import TuningPanel

    config = LiveConfig()
    panel = TuningPanel(config)     # config is the SAME LiveConfig
                                     # instance passed to YOLOTracker
                                     # and every analytics detector
    ...
    # in the main loop, alongside the existing key-toggle handling:
    if key == ord("p"):
        panel.print_snapshot()
"""

import cv2


class TuningPanel:
    WINDOW_NAME = "Tuning Panel"

    def __init__(self, config):
        self.config = config

        # WINDOW_NORMAL (not the default WINDOW_AUTOSIZE) so the
        # window can actually be resized/dragged to fit alongside the
        # main "Traffic Dashboard" window on screen.
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 420, 400)

        # ── YOLO confidence: trackbar 0-100 -> real 0.00-1.00 ──────
        cv2.createTrackbar(
            "YOLO Conf x100", self.WINDOW_NAME,
            int(config.YOLO_CONFIDENCE_THRESHOLD * 100), 100,
            self._make_callback("YOLO_CONFIDENCE_THRESHOLD", lambda v: v / 100.0),
        )

        # ── Stationary duration: trackbar 1-30 seconds, 1:1 ────────
        cv2.createTrackbar(
            "Stationary Sec", self.WINDOW_NAME,
            int(config.STATIONARY_DURATION_SEC), 30,
            self._make_callback("STATIONARY_DURATION_SEC", lambda v: max(1, v)),
        )

        # ── Stationary pixel threshold: trackbar 1-50px, 1:1 ───────
        cv2.createTrackbar(
            "Stationary Px", self.WINDOW_NAME,
            int(config.STATIONARY_PIXEL_THRESHOLD), 50,
            self._make_callback("STATIONARY_PIXEL_THRESHOLD", lambda v: max(1, v)),
        )

        # ── Stationary area-change: trackbar 0-100 -> 0.00-1.00 ────
        cv2.createTrackbar(
            "Stationary Area x100", self.WINDOW_NAME,
            int(config.STATIONARY_AREA_CHANGE_THRESHOLD * 100), 100,
            self._make_callback("STATIONARY_AREA_CHANGE_THRESHOLD", lambda v: v / 100.0),
        )

        # ── Wrong-way duration: trackbar 1-30 seconds, 1:1 ─────────
        cv2.createTrackbar(
            "WrongWay Sec", self.WINDOW_NAME,
            int(config.WRONG_WAY_DURATION_SEC), 30,
            self._make_callback("WRONG_WAY_DURATION_SEC", lambda v: max(1, v)),
        )

        # ── Wrong-way cosine: trackbar 0-200 -> real -1.00..+1.00 ──
        # (trackbars can't start below 0, so we center trackbar=100
        # on real value 0.0; trackbar=0 -> -1.0, trackbar=200 -> +1.0)
        cosine_trackbar_start = int((config.WRONG_WAY_COSINE_THRESHOLD + 1.0) * 100)
        cv2.createTrackbar(
            "WrongWay Cosine (0=-1,200=+1)", self.WINDOW_NAME,
            cosine_trackbar_start, 200,
            self._make_callback("WRONG_WAY_COSINE_THRESHOLD", lambda v: (v / 100.0) - 1.0),
        )

        # ── Hazard confidence: trackbar 0-100 -> real 0.00-1.00 ────
        cv2.createTrackbar(
            "Hazard Conf x100", self.WINDOW_NAME,
            int(config.HAZARD_CONFIDENCE_THRESHOLD * 100), 100,
            self._make_callback("HAZARD_CONFIDENCE_THRESHOLD", lambda v: v / 100.0),
        )

        # ── Hazard persistence: trackbar 1-30 seconds, 1:1 ─────────
        cv2.createTrackbar(
            "Hazard Sec", self.WINDOW_NAME,
            int(config.HAZARD_PERSISTENCE_SEC), 30,
            self._make_callback("HAZARD_PERSISTENCE_SEC", lambda v: max(1, v)),
        )

        # ── Congestion capacity: trackbar 1-50 vehicles, 1:1 ───────
        # NOTE: with the stopped-vehicle congestion logic, this is the
        # number of SIMULTANEOUSLY STOPPED vehicles the ROI tolerates
        # before congestion fires — not a raw visible-vehicle count.
        cv2.createTrackbar(
            "Congestion Cap", self.WINDOW_NAME,
            int(config.CONGESTION_CAPACITY), 50,
            self._make_callback("CONGESTION_CAPACITY", lambda v: max(1, v)),
        )

        # ── Congestion stopped-duration: trackbar 1-30 seconds, 1:1 ─
        # How long a single vehicle must stay put before it counts as
        # "stopped" toward the congestion tally.
        cv2.createTrackbar(
            "Congestion Stop Sec", self.WINDOW_NAME,
            int(config.CONGESTION_STOPPED_DURATION_SEC), 30,
            self._make_callback("CONGESTION_STOPPED_DURATION_SEC", lambda v: max(1, v)),
        )

        # ── Congestion stopped-pixel tolerance: trackbar 1-50px, 1:1 ─
        cv2.createTrackbar(
            "Congestion Stop Px", self.WINDOW_NAME,
            int(config.CONGESTION_STOPPED_PIXEL_THRESHOLD), 50,
            self._make_callback("CONGESTION_STOPPED_PIXEL_THRESHOLD", lambda v: max(1, v)),
        )

    def _make_callback(self, attr_name, transform):
        """
        Returns a closure cv2 can call with the raw trackbar int.
        transform() converts that raw int into the real value and
        writes it straight onto the shared LiveConfig instance — every
        analytics module sees the new value on its very next check()
        call, no restart needed.
        """
        def _callback(raw_value):
            setattr(self.config, attr_name, transform(raw_value))
        return _callback

    def print_snapshot(self):
        """
        Call this (main.py wires it to the 'p' key) to dump the
        CURRENT live tuning to the console in a paste-ready format —
        once you've found values worth keeping permanently, copy them
        straight into thresholds.py's defaults.
        """
        print("\n--- Current live tuning (copy into thresholds.py if you like it) ---")
        for key, value in self.config.snapshot().items():
            print(f"{key} = {value}")
        print("---------------------------------------------------------------\n")
