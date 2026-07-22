"""
src/analytics/congestion.py

Module IV — Congestion Analysis (STOPPED-vehicle based).

LOGIC CHANGE (v2): congestion is no longer "how many vehicles are
visible inside the ROI right now" — a busy but free-flowing road is
NOT congestion. Congestion now means TRAFFIC HAS STOPPED MOVING:

    A vehicle counts toward congestion only if it has been STOPPED
    (centroid barely moved) for at least
    CONGESTION_STOPPED_DURATION_SEC continuous seconds while inside
    the ROI. When the number of such stopped vehicles exceeds
    CONGESTION_CAPACITY, a congestion event fires (rising edge only,
    with the usual cooldown).

This reuses the exact same sliding-window movement test proven in
stationary.py (max centroid displacement over a time window), but with
its OWN duration/pixel thresholds — congestion should react FASTER
than the stationary module (a queue forming for 3 seconds is
congestion; a single car parked for 5+ seconds is a stationary
incident). Both sets of thresholds are live-tunable via LiveConfig.

DESIGN DECISION: loops over tracks.keys(), NOT get_active_ids() —
the OPPOSITE of the old snapshot logic, and the SAME choice as
stationary.py, for the same reason: a stopped car briefly occluded by
a passing truck is still stopped. Its history is untouched either way;
we must not let one missed detection frame reset a queue count.

LIVE TUNING: CONGESTION_CAPACITY, CONGESTION_STOPPED_DURATION_SEC and
CONGESTION_STOPPED_PIXEL_THRESHOLD are read from the shared LiveConfig
object (self.config) fresh every check() call — see
config/live_config.py.

KNOWN LIMITATION: CONGESTION_ROI_POLYGON_NORM in thresholds.py is
traced against the current test footage — it still needs per-camera
calibration in a real deployment (see tools/calibrate_zones.py).
Only VEHICLE_CLASSES (Car/Bike/Bus/Truck) are counted — Person and
Animal detections inside the ROI are deliberately ignored.
"""

import time

from config.thresholds import (
    CONGESTION_ROI_POLYGON_NORM,
    VEHICLE_CLASSES,
    EVENT_COOLDOWN_SEC,
)
from src.geometry import denormalize_polygon, point_in_polygon


class CongestionDetector:
    def __init__(self, track_manager, frame_width, frame_height, config):
        self.track_manager = track_manager

        # The SAME LiveConfig instance every other tunable module reads
        # from. All three congestion thresholds are read fresh inside
        # check() below, never cached here, so the tuning panel's
        # sliders take effect immediately.
        self.config = config

        # ROI polygon is stored NORMALIZED (0-1) in thresholds.py so it
        # works at ANY resolution. Converted to real pixel coordinates
        # ONCE here, via the shared helper in src/geometry.py, since
        # frame_width/frame_height never change for a given video — no
        # reason to redo this math every frame.
        self.roi_polygon = denormalize_polygon(
            CONGESTION_ROI_POLYGON_NORM, frame_width, frame_height
        )

        # ONE shared cooldown value, not a per-ID dictionary — congestion
        # describes the state of the whole ROI, not any single vehicle.
        # Start at -inf so the very first qualifying frame is allowed.
        self.last_triggered = float("-inf")

        # Tracks whether we're currently inside a congestion episode.
        # Starts False because no over-threshold frame has been seen yet.
        self.is_congested = False

        # Most recent stopped-vehicle count — read by main.py's HUD so
        # the dashboard can display "Stopped: N / cap" every frame
        # without re-running the whole check.
        self.last_stopped_count = 0

    def _get_centroid(self, bbox):
        """bbox is (x1, y1, x2, y2). Return (cx, cy) — the box's center."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _distance(self, p1, p2):
        """Straight-line distance between two (x, y) points."""
        x1, y1 = p1
        x2, y2 = p2
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def _is_stopped(self, history, timestamp, stopped_duration_sec, pixel_threshold):
        """
        Same sliding-window movement test as stationary.py, with
        congestion's own (shorter) duration: keep only history entries
        from the last stopped_duration_sec seconds, require the window
        to actually span (most of) that duration, and require the
        centroid never strayed more than pixel_threshold from where the
        window started.
        """
        window = [
            entry for entry in history
            if timestamp - entry[0] <= stopped_duration_sec
        ]

        # Need at least 2 points to measure any movement at all.
        if len(window) < 2:
            return False

        # The OLDEST point must be close to the full duration old — a
        # car that entered frame one second ago can't yet prove it's
        # been stopped for three. The *0.9 grace matches stationary.py.
        if (timestamp - window[0][0]) < stopped_duration_sec * 0.9:
            return False

        centroids = [self._get_centroid(entry[1]) for entry in window]
        max_distance = max(
            self._distance(centroids[0], c) for c in centroids
        )
        return max_distance <= pixel_threshold

    def check(self, timestamp=None):
        """
        Call once per frame. Returns a list with ZERO or ONE event dict:
            [{"stopped_count": int, "capacity": int,
              "stopped_duration_sec": float, "timestamp": float}]
        """
        if timestamp is None:
            timestamp = time.time()

        # Read every live-tunable value once per call, consistent with
        # the pattern used in every other module.
        congestion_capacity = self.config.CONGESTION_CAPACITY
        stopped_duration_sec = self.config.CONGESTION_STOPPED_DURATION_SEC
        pixel_threshold = self.config.CONGESTION_STOPPED_PIXEL_THRESHOLD

        stopped_count = 0

        # tracks.keys(), not get_active_ids() — a stopped car briefly
        # occluded by a passing truck is still stopped; don't let one
        # missed detection frame reset the queue count.
        for track_id in list(self.track_manager.tracks.keys()):
            info = self.track_manager.tracks[track_id]

            # Only vehicles count toward congestion — not people,
            # animals, or anything else that might be on/near the road.
            if info["class_name"] not in VEHICLE_CLASSES:
                continue

            history = info["history"]
            if not history:
                continue

            # Most recent known position must be inside the ROI —
            # a car stopped in a driveway outside the road area is
            # not road congestion.
            centroid = self._get_centroid(history[-1][1])
            if not point_in_polygon(centroid, self.roi_polygon):
                continue

            # THE new core test: has this vehicle been genuinely
            # stopped for the whole congestion window?
            if self._is_stopped(history, timestamp, stopped_duration_sec, pixel_threshold):
                stopped_count += 1

        self.last_stopped_count = stopped_count

        events = []
        currently_over = stopped_count > congestion_capacity

        # Rising edge only: fire when the ROI transitions from flowing
        # to congested. While the episode remains active, suppress repeats.
        if currently_over and not self.is_congested:
            if timestamp - self.last_triggered >= EVENT_COOLDOWN_SEC:
                event = {
                    "stopped_count": stopped_count,
                    "capacity": congestion_capacity,
                    "stopped_duration_sec": stopped_duration_sec,
                    "timestamp": timestamp,
                }
                events.append(event)
                self.last_triggered = timestamp
            self.is_congested = True

        # Falling edge: once enough stopped vehicles start moving again
        # (count back at/below threshold), reset so the next rise can
        # trigger a fresh event.
        elif not currently_over and self.is_congested:
            self.is_congested = False

        return events
