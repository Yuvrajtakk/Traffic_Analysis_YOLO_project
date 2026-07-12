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

KNOWN LIMITATION: CONGESTION_ROI_POLYGON_NORM in thresholds.py is a
PLACEHOLDER — a rough rectangle covering most of the frame, not an
actual traced road boundary. It does not know where the real drivable
surface is for any specific camera. Same category of limitation as
wrong_way.py's single global flow vector: this needs manual per-camera
calibration in a real deployment (a human looking at a sample frame
from the actual guest-house camera and tracing the real road edges as
polygon points) before this module's counts are meaningfully accurate.
Only VEHICLE_CLASSES (Bus/Car/Bike) are counted toward congestion —
Person and Animal detections inside the ROI are deliberately ignored,
since "congestion" here specifically means vehicle traffic density, not
general crowd density (a different concept, not built here).
"""

import time

import cv2
import numpy as np

from config.thresholds import (
    CONGESTION_ROI_POLYGON_NORM,
    CONGESTION_CAPACITY,
    VEHICLE_CLASSES,
    EVENT_COOLDOWN_SEC,
)


class CongestionDetector:
    def __init__(self, track_manager, frame_width, frame_height):
        self.track_manager = track_manager

        # ROI polygon is stored NORMALIZED (0-1) in thresholds.py so it
        # works at ANY resolution. Converted to real pixel coordinates
        # ONCE here, since frame_width/frame_height never change for a
        # given video — no reason to redo this math every frame.
        self.roi_polygon = np.array(
            [(nx * frame_width, ny * frame_height)
            for (nx, ny) in CONGESTION_ROI_POLYGON_NORM],
            dtype=np.int32,
        )

        # ONE shared cooldown value, not a per-ID dictionary — congestion
        # describes the state of the whole ROI, not any single vehicle.
        # Start at -inf so the very first qualifying frame is allowed.
        self.last_triggered = float("-inf")

    def _get_centroid(self, bbox):
        """bbox is (x1, y1, x2, y2). Return (cx, cy) — the box's center."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _is_inside_roi(self, point):
        """
        point is (x, y) in pixel coordinates. Returns True if the point
        falls inside self.roi_polygon, False otherwise.
        cv2.pointPolygonTest returns: positive = inside, negative =
        outside, zero = exactly on the edge.
        """
        return cv2.pointPolygonTest(
            self.roi_polygon, (float(point[0]), float(point[1])), False
        ) > 0

    def check(self, timestamp=None):
        """
        Call once per frame. Returns a list with ZERO or ONE event dict:
            [{"count": int, "capacity": int, "timestamp": float}]
        """
        if timestamp is None:
            timestamp = time.time()

        # get_active_ids(), not tracks.keys() — deliberately the
        # OPPOSITE choice from stationary/wrong_way. We only trust
        # vehicles currently, visibly detected — not ones sitting in
        # the occlusion buffer that might have already left frame.
        
        # active_ids = self.track_manager.get_active_ids()
        
        # FIX: must pass the SAME timestamp through, so it's compared
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

            if self._is_inside_roi(centroid):
                count += 1

        events = []

        if count > CONGESTION_CAPACITY and (timestamp - self.last_triggered >= EVENT_COOLDOWN_SEC):
            event = {
                "count": count,
                "capacity": CONGESTION_CAPACITY,
                "timestamp": timestamp,
            }
            events.append(event)
            self.last_triggered = timestamp

        return events