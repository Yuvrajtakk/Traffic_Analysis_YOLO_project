"""
src/analytics/stationary.py

Module I — Stationary Vehicle Detection.

A vehicle is flagged if its centroid position stays within a small
pixel tolerance for a continuous window of STATIONARY_DURATION_SEC,
using the shared TrackManager's per-ID history — NOT a private copy of
position data, so this module can never disagree with wrong_way.py or
congestion.py about whether a given ID is still "the same object."
"""

import time

from config.thresholds import (
    STATIONARY_DURATION_SEC,
    STATIONARY_PIXEL_THRESHOLD,
    STATIONARY_AREA_CHANGE_THRESHOLD,
    DEBUG,
    VEHICLE_CLASSES,
)


class StationaryDetector:
    def __init__(self, track_manager):
        # The SAME TrackManager instance every other module reads from.
        # We never keep our own private copy of position history — this
        # guarantees stationary.py, wrong_way.py, and congestion.py can
        # never disagree about whether a given ID is still "alive."
        self.track_manager = track_manager

        # Keyed by track_id -> the centroid position and area at the moment
        # this incident fired. Using an anchor point (not a set) lets us
        # check "has it actually moved away from where it stopped" going
        # forward, instead of re-deriving that from the same noisy sliding
        # window that caused the false re-fires.
        self.currently_stationary = {}

    def _get_centroid(self, bbox):
        """bbox is (x1, y1, x2, y2). Return (cx, cy) — the box's center."""
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        return (cx, cy)

    def _distance(self, p1, p2):
        """
        p1, p2 are (x, y) points. Return the straight-line distance
        between them — plain Pythagoras: sqrt(dx^2 + dy^2).
        This is our "how much did it actually wobble" measurement.
        """
        x1, y1 = p1
        x2, y2 = p2
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def check(self, timestamp=None):
        """
        Call once per frame. Returns a list of event dicts for every
        vehicle newly confirmed stationary this frame:
            {"id": int, "class_name": str, "bbox": tuple, "timestamp": float}
        """
        if timestamp is None:
            timestamp = time.time()

        events = []

        # Drop any track_ids that no longer exist in TrackManager at all.
        live_ids = set(self.track_manager.tracks.keys())
        for stale_id in list(self.currently_stationary.keys()):
            if stale_id not in live_ids:
                del self.currently_stationary[stale_id]

        # DESIGN DECISION: we loop over tracks.keys(), NOT
        # get_active_ids(). get_active_ids() only returns IDs seen in
        # roughly the last 1.5 frames (~0.06s) — far too narrow a
        # window. tracks.keys() returns every ID still within
        # ByteTrack's ~30-frame occlusion buffer, active OR briefly
        # occluded. This matters because a car blocked by a passing
        # truck for a frame or two should NOT have its stationary
        # streak silently skipped just because it wasn't detected in
        # that exact frame — its history is untouched either way, we
        # just want to make sure we actually LOOK at it every frame.
        for track_id in self.track_manager.tracks.keys():

            info = self.track_manager.tracks[track_id]

            # Not this module's job — stationary PEOPLE aren't tracked
            # here, only vehicles (Bus/Car/Bike, per VEHICLE_CLASSES).
            if info["class_name"] not in VEHICLE_CLASSES:
                continue

            history = info["history"]  # list of (timestamp, bbox) tuples

            # Keep only the entries from the last STATIONARY_DURATION_SEC
            # seconds. Same filtering pattern TrackManager itself uses
            # to decide what's still "recent enough" to matter.
            window = [
                entry for entry in history
                if timestamp - entry[0] <= STATIONARY_DURATION_SEC
            ]

            # Quick, gated diagnostic for each track's sliding window.
            # Runs only when DEBUG is True so normal operation is
            # unaffected. Computes a temporary max-distance in the
            # same way the main logic would (only for reporting).
            if DEBUG:
                oldest_age = (timestamp - window[0][0]) if window else None
                if len(window) >= 2:
                    centroids_tmp = [self._get_centroid(entry[1]) for entry in window]
                    max_dist_tmp = max(self._distance(centroids_tmp[0], c) for c in centroids_tmp)
                else:
                    max_dist_tmp = "N/A"
                print(f"[stationary-debug] id={track_id} class={info['class_name']} window_len={len(window)} oldest_age={oldest_age} max_distance={max_dist_tmp}")

            # Guard 1: need at least 2 points to measure any movement
            # at all — a single point can't prove "stayed still."
            if len(window) < 2:
                continue

            # Guard 2: the OLDEST point in window must be close to the
            # full duration old. A car that entered frame half a second
            # ago has only half a second of proof — not enough to claim
            # "stationary for 5 seconds." The *0.9 gives a little grace
            # instead of demanding an exact 5.000-second match.
            oldest_timestamp = window[0][0]
            if (timestamp - oldest_timestamp) < STATIONARY_DURATION_SEC * 0.9:
                continue

            # Convert every (timestamp, bbox) in the window into just
            # its centroid point, then find the BIGGEST distance
            # between the first centroid and any other one in the
            # window. Small max distance = barely moved = candidate
            # for "stationary." Large max distance = actually moving.
            centroids = [self._get_centroid(entry[1]) for entry in window]
            max_distance = max(
                self._distance(centroids[0], c) for c in centroids
            )
            areas = [
                (entry[1][2] - entry[1][0]) * (entry[1][3] - entry[1][1])
                for entry in window
            ]
            base_area = areas[0]
            if base_area > 0:
                max_area_ratio_change = max(
                    abs(a - base_area) / base_area for a in areas
                )
            else:
                max_area_ratio_change = 0.0

            # Only vehicles that barely moved get considered further.
            # Everything below this line is INSIDE this if-block on
            # purpose — a moving vehicle must never reach the event-
            # firing code at all.
            current_centroid = centroids[-1]

            if track_id in self.currently_stationary:
                # Already fired for this incident. Only clear it if the
                # vehicle has genuinely moved away from the ANCHOR point
                # recorded at fire-time — not based on the window's own
                # max_distance, which can be corrupted by a single stale
                # jitter frame still sitting in the 5-second lookback.
                anchor_centroid, anchor_area = self.currently_stationary[track_id]
                current_area = areas[-1]
                area_ratio_change = (
                    abs(current_area - anchor_area) / anchor_area
                    if anchor_area > 0 else 0.0
                )
                if (
                    self._distance(current_centroid, anchor_centroid) > STATIONARY_PIXEL_THRESHOLD
                    or area_ratio_change > STATIONARY_AREA_CHANGE_THRESHOLD
                ):
                    del self.currently_stationary[track_id]
                continue

            if max_distance <= STATIONARY_PIXEL_THRESHOLD and max_area_ratio_change <= STATIONARY_AREA_CHANGE_THRESHOLD:
                # Not yet fired for this incident, and the window
                # confirms it's genuinely been still — fire once, and
                # anchor future re-checks to THIS position.
                event = {
                    "id": track_id,
                    "class_name": info["class_name"],
                    "bbox": info["history"][-1][1],
                    "timestamp": timestamp,
                }
                events.append(event)
                self.currently_stationary[track_id] = (current_centroid, areas[-1])

        return events