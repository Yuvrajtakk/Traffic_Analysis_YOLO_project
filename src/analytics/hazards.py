"""
src/analytics/hazards.py

Module III — Environmental Hazard Recognition (Fire, Smoke, Accident).

Unlike stationary.py and wrong_way.py, this module does NOT use
TrackManager or per-object position history — hazards don't have a
meaningful stable "identity" to track across frames the way a vehicle
does. Instead, this module works directly off each frame's RAW
detections (straight from YOLOTracker.track()), checking: has a given
hazard CLASS been seen, confidently, in EVERY frame continuously for at
least HAZARD_PERSISTENCE_SEC?

KNOWN LIMITATION: persistence requires a detection in literally every
processed frame with no gaps — a single missed/low-confidence frame due
to detector flicker resets the streak entirely, even if the underlying
real-world hazard never actually stopped. A more forgiving version
(e.g. "confident in 8 of the last 10 frames") would tolerate flicker
better but is deliberately deferred, matching this project's pattern of
building the simple strict version first.
"""

import time

from config.thresholds import (
    HAZARD_CLASSES,
    HAZARD_CONFIDENCE_THRESHOLD,
    HAZARD_PERSISTENCE_SEC,
    EVENT_COOLDOWN_SEC,
)


class HazardDetector:
    def __init__(self):
        # NOTE: no track_manager parameter here at all — this module
        # doesn't need shared position history, just raw per-frame
        # detections handed to check() directly.

        # Keyed by CLASS NAME (e.g. "Fire"), not by track_id — the key
        # difference from stationary/wrong_way's per-ID dicts. Value =
        # the timestamp this class's streak first started.
        self.hazard_since = {}

        # Same cooldown pattern as before, still keyed by class name.
        self.last_triggered = {}

    def check(self, detections, timestamp=None):
        """
        Call once per frame, passing the RAW detections list straight
        from YOLOTracker.track() (NOT TrackManager — this module reads
        current-frame detections directly, no history needed).

        Returns a list of event dicts for every hazard class newly
        confirmed this frame:
            {"class_name": str, "confidence": float, "bbox": tuple, "timestamp": float}
        """
        if timestamp is None:
            timestamp = time.time()

        events = []

        # STEP A — figure out which hazard classes were genuinely,
        # confidently seen THIS frame. Starts truly empty: a class only
        # becomes a key here if it actually passed both checks below.
        seen_this_frame = {}

        for det in detections:
            # Not Fire/Smoke/Accident at all (e.g. "Car", "Person") —
            # not this module's concern, skip immediately.
            if det["class_name"] not in HAZARD_CLASSES:
                continue

            # Is a hazard class, but YOLO isn't confident enough about
            # it — could easily be a false positive (red truck, sunset
            # glare mistaken for fire). Not trustworthy enough, skip.
            if det["confidence"] < HAZARD_CONFIDENCE_THRESHOLD:
                continue

            # Passed both checks — record it. If the SAME hazard class
            # appears twice this frame (two separate fire detections),
            # this simply overwrites with the later one — we only care
            # about class-level presence, not counting instances.
            seen_this_frame[det["class_name"]] = det

        # STEP B — for every possible hazard class, decide: seen this
        # frame or not, and update the persistence streak accordingly.
        # Structurally identical to wrong_way.py's if/else block, just
        # keyed by class name instead of track_id.
        for cls in HAZARD_CLASSES:

            if cls in seen_this_frame:
                # Seen, confidently, this frame.

                # First frame we've seen this class? Start its clock.
                if cls not in self.hazard_since:
                    self.hazard_since[cls] = timestamp

                # How long has it been continuously present, counting
                # from the moment the streak started?
                duration = timestamp - self.hazard_since[cls]

                if duration >= HAZARD_PERSISTENCE_SEC:
                    # Persisted long enough — but still respect
                    # cooldown, same pattern as the other two modules,
                    # so we don't fire a new event every single frame
                    # while the hazard remains visible.
                    last_time = self.last_triggered.get(cls, 0)
                    if timestamp - last_time >= EVENT_COOLDOWN_SEC:
                        det = seen_this_frame[cls]
                        event = {
                            "class_name": cls,
                            "confidence": det["confidence"],
                            "bbox": det["bbox"],
                            "timestamp": timestamp,
                        }
                        events.append(event)
                        self.last_triggered[cls] = timestamp

            else:
                # NOT seen this frame — streak broken. Remove this
                # class's entry entirely so a FUTURE sighting starts
                # counting from zero, not from a stale old timestamp.
                # This is what makes "continuous" actually mean
                # continuous, with zero tolerance for gaps.
                self.hazard_since.pop(cls, None)

        return events