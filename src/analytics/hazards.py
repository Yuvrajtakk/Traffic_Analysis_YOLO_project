"""
src/analytics/hazards.py

Module III — Environmental Hazard Recognition (Fire, Smoke, Accident).

Unlike stationary.py and wrong_way.py, this module does NOT use
TrackManager or per-object position history — hazards don't have a
meaningful stable "identity" to track across frames the way a vehicle
does. Instead, this module works directly off each frame's RAW
detections (straight from YOLOTracker.track()), checking: has a given
hazard CLASS been seen confidently in enough of the last few frames,
for at least HAZARD_PERSISTENCE_SEC?

LIVE TUNING: HAZARD_CONFIDENCE_THRESHOLD and HAZARD_PERSISTENCE_SEC are
read from the shared LiveConfig object (self.config) fresh every
check() call — see config/live_config.py.

The flicker rule is intentionally simple and explainable: for each
hazard class, remember only the last M frames as True/False values.
If at least N of those M frames were confident detections, count the
hazard as present. This prevents smoke/fire/smoke flicker from causing
repeated start/stop behavior.
"""

import time

from config.thresholds import (
    HAZARD_CLASSES,
    HAZARD_EVENT_COOLDOWN_SEC,
    HAZARD_FLICKER_MIN_CONFIDENT_FRAMES,
    HAZARD_FLICKER_WINDOW_FRAMES,
)


class HazardDetector:
    def __init__(self, config):
        # NOTE: no track_manager parameter here at all — this module
        # doesn't need shared position history, just raw per-frame
        # detections handed to check() directly.

        # The SAME LiveConfig instance every other tunable module reads
        # from. Never cache HAZARD_CONFIDENCE_THRESHOLD or
        # HAZARD_PERSISTENCE_SEC out of this in __init__ — always read
        # fresh inside check() so the tuning panel takes effect live.
        self.config = config

        # Keyed by CLASS NAME (e.g. "Fire"), not by track_id — the key
        # difference from stationary/wrong_way's per-ID dicts. Value =
        # the timestamp this class's streak first started.
        self.hazard_since = {}

        # Same cooldown pattern as before, still keyed by class name.
        self.last_triggered = {}

        # Keyed by class name. Each value is a short list of booleans:
        # True means this class was confidently seen in that frame,
        # False means it was missed or below confidence.
        self.recent_presence = {cls: [] for cls in HAZARD_CLASSES}

    def _remember_presence(self, cls, was_seen):
        """
        Store this frame's True/False result for one hazard class,
        keeping only the most recent HAZARD_FLICKER_WINDOW_FRAMES.
        """
        recent = self.recent_presence.setdefault(cls, [])
        recent.append(was_seen)

        if len(recent) > HAZARD_FLICKER_WINDOW_FRAMES:
            recent.pop(0)

    def _is_present_despite_flicker(self, cls):
        """
        Return True when this class was confidently detected in enough
        of the recent frames. Example with defaults: 3 of last 5.
        """
        recent = self.recent_presence.get(cls, [])
        return recent.count(True) >= HAZARD_FLICKER_MIN_CONFIDENT_FRAMES

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

        # Read both tunables once per call — see stationary.py's same
        # comment for why (consistency within this one pass).
        hazard_confidence_threshold = self.config.HAZARD_CONFIDENCE_THRESHOLD
        hazard_persistence_sec = self.config.HAZARD_PERSISTENCE_SEC

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
            if det["confidence"] < hazard_confidence_threshold:
                continue

            # Passed both checks — record it. If the SAME hazard class
            # appears twice this frame (two separate fire detections),
            # this simply overwrites with the later one — we only care
            # about class-level presence, not counting instances.
            seen_this_frame[det["class_name"]] = det

        # STEP B — for every possible hazard class, remember whether it
        # was seen this frame, then decide whether recent history is
        # strong enough to count it as currently present.
        # Structurally identical to wrong_way.py's if/else block, just
        # keyed by class name instead of track_id.
        for cls in HAZARD_CLASSES:
            self._remember_presence(cls, cls in seen_this_frame)
            present_now = self._is_present_despite_flicker(cls)

            if present_now:
                # Seen confidently in enough recent frames.

                # First frame we've seen this class? Start its clock.
                if cls not in self.hazard_since:
                    self.hazard_since[cls] = timestamp

                # How long has it been continuously present, counting
                # from the moment the streak started?
                duration = timestamp - self.hazard_since[cls]

                if duration >= hazard_persistence_sec:
                    # Persisted long enough — but still respect
                    # cooldown, same pattern as the other two modules,
                    # so we don't fire a new event every single frame
                    # while the hazard remains visible.
                    last_time = self.last_triggered.get(cls, float("-inf"))
                    if timestamp - last_time >= HAZARD_EVENT_COOLDOWN_SEC:
                        # If this frame is one of the tolerated missed
                        # frames, wait for a fresh bbox before recording.
                        if cls not in seen_this_frame:
                            continue

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
                # Not present in enough of the recent frames — streak
                # broken. One missed frame no longer breaks it; several
                # misses in the rolling window do.
                self.hazard_since.pop(cls, None)

        return events
