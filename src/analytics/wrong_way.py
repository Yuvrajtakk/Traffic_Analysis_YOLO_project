"""
src/analytics/wrong_way.py

Module II — Wrong-Way Trajectory Detection.

A vehicle is flagged if its trajectory vector points against the
authorized flow vector (cosine < threshold) for a continuous window of
WRONG_WAY_DURATION_SEC, using the SAME shared TrackManager history that
stationary.py reads from.

KNOWN LIMITATION: this version uses ONE global AUTHORIZED_FLOW_VECTOR
for the entire frame. On a two-lane road with opposite-direction traffic,
or a 4-way intersection, this would incorrectly flag the "opposite lane"
as wrong-way, since it has no concept of per-lane/per-zone expected
direction. The correct fix is per-zone flow vectors (one vector per
ROI polygon, matching congestion.py's existing ROI-polygon pattern) —
deliberately deferred as a Phase 2 upgrade, not built here, to match
the original single-vector design in the project plan first.
"""

import time

from config.thresholds import (
    AUTHORIZED_FLOW_VECTOR,
    WRONG_WAY_SMOOTHING_WINDOW,
    WRONG_WAY_COSINE_THRESHOLD,
    WRONG_WAY_DURATION_SEC,
    VEHICLE_CLASSES,
    EVENT_COOLDOWN_SEC,
)


class WrongWayDetector:
    def __init__(self, track_manager):
        # The SAME shared TrackManager every other module reads from —
        # never our own private copy of position data.
        self.track_manager = track_manager

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

    def check(self, timestamp=None):
        """
        Call once per frame. Returns a list of event dicts for every
        vehicle newly confirmed wrong-way (continuously, for at least
        WRONG_WAY_DURATION_SEC) this frame.
        """
        if timestamp is None:
            timestamp = time.time()

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

            # Compare T against the road's expected direction.
            cosine = self._cosine_similarity(T, AUTHORIZED_FLOW_VECTOR)

            if cosine < WRONG_WAY_COSINE_THRESHOLD:
                # Wrong-way THIS frame. If this is the first frame we've
                # seen it go wrong-way, start the clock right now.
                if track_id not in self.wrong_way_since:
                    self.wrong_way_since[track_id] = timestamp

                # How long has it been continuously wrong-way, counting
                # from the moment the streak started?
                duration = timestamp - self.wrong_way_since[track_id]

                if duration >= WRONG_WAY_DURATION_SEC:
                    # Been wrong-way long enough — but still respect
                    # cooldown, same pattern as stationary.py, so we
                    # don't fire a new event every single frame while
                    # the violation continues.
                    last_time = self.last_triggered.get(track_id, 0)
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