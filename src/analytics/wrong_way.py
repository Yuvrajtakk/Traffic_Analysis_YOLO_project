"""
src/analytics/wrong_way.py

Module II — Wrong-Way Trajectory Detection (multi-lane / multi-zone).

A vehicle is flagged if its trajectory vector points against the
EXPECTED flow vector *for whichever zone its centroid currently sits
in* (cosine < threshold), for a continuous window of
WRONG_WAY_DURATION_SEC. Uses the SAME shared TrackManager history that
stationary.py reads from.

UPGRADE FROM v1: the previous version used ONE global
AUTHORIZED_FLOW_VECTOR for the entire frame, which would incorrectly
flag the opposite lane as wrong-way on any two-lane road with
opposite-direction traffic, or any 4-way intersection — that limitation
was explicitly flagged in v1's own docstring as a deferred Phase 2
item. This version replaces the single vector with WRONG_WAY_ZONES
(config/thresholds.py): a list of ROI polygons, each carrying its own
expected flow direction, checked the same way congestion.py already
checks its ROI — a vehicle is judged against the flow vector of
whichever zone it's currently inside, not a single frame-wide rule.

A vehicle whose centroid isn't inside ANY defined zone falls back to
WRONG_WAY_DEFAULT_FLOW_VECTOR rather than being silently skipped —
keeps behavior conservative outside calibrated zones instead of going
blind there entirely.

LIVE TUNING: WRONG_WAY_DURATION_SEC and WRONG_WAY_COSINE_THRESHOLD are
read from the shared LiveConfig object (self.config) fresh every
check() call — see config/live_config.py. Zone geometry itself
(WRONG_WAY_ZONES) is NOT live-tunable — it's structural per-camera
calibration, not a threshold you'd nudge while watching results.

REMAINING KNOWN LIMITATION: WRONG_WAY_ZONES is still placeholder
geometry (see thresholds.py) — a rough half-frame split, not a
manually traced lane boundary. Real deployment needs a human tracing
the actual lane edges for the specific camera before zone assignment
is meaningfully accurate.
"""

import time

from config.thresholds import (
    WRONG_WAY_ZONES,
    WRONG_WAY_DEFAULT_FLOW_VECTOR,
    WRONG_WAY_SMOOTHING_WINDOW,
    VEHICLE_CLASSES,
    EVENT_COOLDOWN_SEC,
)
from src.geometry import denormalize_polygon, point_in_polygon


class WrongWayDetector:
    def __init__(self, track_manager, config, frame_width, frame_height):
        # The SAME shared TrackManager every other module reads from —
        # never our own private copy of position data.
        self.track_manager = track_manager

        # The SAME LiveConfig instance every other tunable module reads
        # from. WRONG_WAY_DURATION_SEC and WRONG_WAY_COSINE_THRESHOLD
        # are read fresh inside check() below, never cached here.
        self.config = config

        # Convert every zone's normalized polygon to real pixel
        # coordinates ONCE here — frame_width/frame_height never
        # change for a given video, so there's no reason to redo this
        # conversion every frame. Each zone keeps its polygon (now in
        # pixels) paired with its own expected flow_vector.
        self.zones = [
            {
                "polygon_px": denormalize_polygon(zone["polygon"], frame_width, frame_height),
                "flow_vector": zone["flow_vector"],
            }
            for zone in WRONG_WAY_ZONES
        ]

        # Remembers "the last time we fired a wrong-way event for this
        # ID" — prevents spamming a new event every frame while the
        # violation continues. Same purpose as stationary.py's version.
        self.last_triggered = {}

        # Remembers "the timestamp this ID FIRST started going wrong-way,
        # continuously, with no break." Starts completely empty — no
        # pre-population needed. Entries get added the moment a car
        # first goes wrong-way, and get REMOVED the instant it stops
        # (see the else branch in check()) — this is what makes the
        # duration measurement mean "continuous," not "total time ever
        # spent wrong-way across separate incidents."
        self.wrong_way_since = {}

    def _get_centroid(self, bbox):
        """bbox is (x1, y1, x2, y2). Return (cx, cy) — the box's center."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _dot_product(self, v1, v2):
        """
        v1, v2 are (x, y) vectors. Returns a single number:
        positive = pointing the same general way,
        negative = pointing the opposite general way,
        near zero = roughly perpendicular.
        """
        return v1[0] * v2[0] + v1[1] * v2[1]

    def _vector_length(self, v):
        """
        v is (x, y). Returns its length (magnitude) — same Pythagoras
        formula as stationary.py's _distance(), just measuring the
        distance from (0,0) to this one point instead of between two
        separate points.
        """
        return (v[0] ** 2 + v[1] ** 2) ** 0.5

    def _cosine_similarity(self, v1, v2):
        """
        Returns cos(angle) between v1 and v2 — always between -1 and +1.
        +1 = exact same direction, 0 = perpendicular, -1 = exact
        opposite direction. Dividing by vector lengths is what makes
        this independent of how FAST either vector is moving — only
        direction matters, not speed.
        """
        dot = self._dot_product(v1, v2)
        len1 = self._vector_length(v1)
        len2 = self._vector_length(v2)

        # GUARD: a vehicle that hasn't moved at all this window has a
        # trajectory vector of (0,0) — length 0. Dividing by zero would
        # crash Python. If either vector has zero length, direction is
        # genuinely undefined, so we return 0.0 (treated as neutral —
        # neither correct-way nor wrong-way) instead of crashing.
        if len1 == 0 or len2 == 0:
            return 0.0

        return dot / (len1 * len2)

    def _expected_flow_for(self, centroid):
        """
        Returns the flow_vector for whichever zone `centroid` falls
        inside — the core of the multi-lane upgrade. First matching
        zone wins if zones are ever accidentally overlapping (they
        shouldn't be, by design — zones should partition the road into
        distinct lanes/directions — but first-match keeps this
        predictable rather than raising an error on overlap).

        Falls back to WRONG_WAY_DEFAULT_FLOW_VECTOR if centroid is
        outside every defined zone (e.g. zones don't cover the full
        frame, or the vehicle briefly strayed off-road).
        """
        for zone in self.zones:
            if point_in_polygon(centroid, zone["polygon_px"]):
                return zone["flow_vector"]
        return WRONG_WAY_DEFAULT_FLOW_VECTOR

    def check(self, timestamp=None):
        """
        Call once per frame. Returns a list of event dicts for every
        vehicle newly confirmed wrong-way (continuously, for at least
        WRONG_WAY_DURATION_SEC) this frame.
        """
        if timestamp is None:
            timestamp = time.time()

        # Read both tunables once per call, consistent with the
        # pattern used in every other module.
        wrong_way_duration_sec = self.config.WRONG_WAY_DURATION_SEC
        wrong_way_cosine_threshold = self.config.WRONG_WAY_COSINE_THRESHOLD

        events = []

        # Same design decision as stationary.py, same reasoning:
        # tracks.keys() sees everything still within the occlusion
        # buffer, not just what was detected this exact frame.
        for track_id in self.track_manager.tracks.keys():
            info = self.track_manager.tracks[track_id]

            if info["class_name"] not in VEHICLE_CLASSES:
                continue

            history = info["history"]

            # Take the last WRONG_WAY_SMOOTHING_WINDOW entries — a
            # fixed COUNT of recent points, not a time duration (unlike
            # stationary.py's window). This smooths out tiny bounding-
            # box jitter that could otherwise fake a brief zigzag.
            # Safe even if history has fewer than that many entries —
            # Python's slicing just returns whatever exists, no crash,
            # no wraparound.
            window = history[-WRONG_WAY_SMOOTHING_WINDOW:]

            # Need at least 2 points to compute ANY direction at all.
            if len(window) < 2:
                continue

            # Trajectory vector T: where it ended up, minus where it
            # started, across this smoothing window. This IS the
            # "arrow" representing this vehicle's recent movement.
            window_first_centroid = self._get_centroid(window[0][1])
            window_last_centroid = self._get_centroid(window[-1][1])
            T = (
                window_last_centroid[0] - window_first_centroid[0],
                window_last_centroid[1] - window_first_centroid[1],
            )

            # NEW: look up the expected flow for THIS vehicle's current
            # zone (based on its latest centroid), instead of comparing
            # against one global vector for the whole frame — this is
            # the actual multi-lane fix.
            expected_flow = self._expected_flow_for(window_last_centroid)

            # Compare T against THIS zone's expected direction.
            cosine = self._cosine_similarity(T, expected_flow)

            if cosine < wrong_way_cosine_threshold:
                # Wrong-way THIS frame. If this is the first frame we've
                # seen it go wrong-way, start the clock right now.
                if track_id not in self.wrong_way_since:
                    self.wrong_way_since[track_id] = timestamp

                # How long has it been continuously wrong-way, counting
                # from the moment the streak started?
                duration = timestamp - self.wrong_way_since[track_id]

                if duration >= wrong_way_duration_sec:
                    # Been wrong-way long enough — but still respect
                    # cooldown, same pattern as stationary.py, so we
                    # don't fire a new event every single frame while
                    # the violation continues.
                    last_time = self.last_triggered.get(track_id, float("-inf"))
                    if timestamp - last_time >= EVENT_COOLDOWN_SEC:
                        event = {
                            "id": track_id,
                            "class_name": info["class_name"],
                            "bbox": info["history"][-1][1],
                            "timestamp": timestamp,
                            "cosine": cosine,
                        }
                        events.append(event)
                        self.last_triggered[track_id] = timestamp

            else:
                # NOT wrong-way this frame — the streak is broken.
                # Remove track_id from wrong_way_since entirely (if
                # present) so a FUTURE wrong-way moment starts counting
                # from zero, not from a stale old timestamp. This is
                # what makes "continuous" actually mean continuous.
                self.wrong_way_since.pop(track_id, None)

        return events
