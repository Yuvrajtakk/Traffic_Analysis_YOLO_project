"""
src/analytics/congestion.py

Module IV — Spatial Congestion / Density Analysis.

Unlike stationary.py and wrong_way.py, this module needs NO time
persistence at all — it's a per-frame SNAPSHOT question: "how many
vehicles are currently, visibly, inside the ROI right now?" Not "has
this been true for N seconds."

DESIGN DECISION: loops over get_active_ids(), NOT tracks.keys() (the
opposite choice from stationary/wrong_way). Counting an occluded
vehicle risks double-counting or counting a car that's already left
frame — for a snapshot count, we only trust what's currently visible.

LIVE TUNING: CONGESTION_CAPACITY is read from the shared LiveConfig
object (self.config) fresh every check() call — see
config/live_config.py.

REFACTOR: point-in-polygon logic now lives in src/geometry.py, shared
with wrong_way.py's per-zone checks, instead of being duplicated here.

KNOWN LIMITATION: CONGESTION_ROI_POLYGON_NORM in thresholds.py is a
PLACEHOLDER — a rough rectangle covering most of the frame, not an
actual traced road boundary. It does not know where the real drivable
surface is for any specific camera. Same category of limitation as
wrong_way.py's zone polygons: this needs manual per-camera calibration
in a real deployment (a human looking at a sample frame from the
actual guest-house camera and tracing the real road edges as polygon
points) before this module's counts are meaningfully accurate.
Only VEHICLE_CLASSES (Car/Bike/Bus/Truck) are counted toward
congestion — Person and Animal detections inside the ROI are
deliberately ignored, since "congestion" here specifically means
vehicle traffic density, not general crowd density (a different
concept, not built here).
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
        # from. CONGESTION_CAPACITY is read fresh inside check() below,
        # never cached here, so the tuning panel's slider takes effect
        # immediately.
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

    def _get_centroid(self, bbox):
        """bbox is (x1, y1, x2, y2). Return (cx, cy) — the box's center."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def check(self, timestamp=None):
        """
        Call once per frame. Returns a list with ZERO or ONE event dict:
            [{"count": int, "capacity": int, "timestamp": float}]
        """
        if timestamp is None:
            timestamp = time.time()

        # Read the live-tunable capacity once per call, consistent with
        # the pattern used in every other module.
        congestion_capacity = self.config.CONGESTION_CAPACITY

        # get_active_ids(), not tracks.keys() — deliberately the
        # OPPOSITE choice from stationary/wrong_way. We only trust
        # vehicles currently, visibly detected — not ones sitting in
        # the occlusion buffer that might have already left frame.
        #
        # Must pass the SAME timestamp through, so it's compared
        # consistently against last_seen values, not against real
        # wall-clock time.
        active_ids = self.track_manager.get_active_ids(timestamp)

        count = 0

        for track_id in active_ids:
            info = self.track_manager.tracks[track_id]

            # Only vehicles count toward congestion — not people,
            # animals, or anything else that might be on/near the road.
            if info["class_name"] not in VEHICLE_CLASSES:
                continue

            # Most recent known position for this vehicle.
            centroid = self._get_centroid(info["history"][-1][1])

            if point_in_polygon(centroid, self.roi_polygon):
                count += 1

        events = []

        if count > congestion_capacity and (timestamp - self.last_triggered >= EVENT_COOLDOWN_SEC):
            event = {
                "count": count,
                "capacity": congestion_capacity,
                "timestamp": timestamp,
            }
            events.append(event)
            self.last_triggered = timestamp

        return events
