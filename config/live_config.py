"""
config/live_config.py

Mutable, live-tunable configuration.

Unlike thresholds.py (plain constants, read once at import time and
frozen for the life of the program), every value here can be changed
WHILE the program is running — via the trackbar panel in
src/tuning_panel.py — and every analytics module reads these values
fresh, every single frame, through this ONE shared object. This is the
exact same "one shared object, everyone reads the same truth" pattern
TrackManager already uses for tracking state — LiveConfig is that same
idea, applied to thresholds instead of positions.

Only NUMERIC THRESHOLDS you'd actually want to nudge while watching
the live feed and seeing the effect immediately live here. Taxonomy
(VEHICLE_CLASSES), zone/ROI geometry, and dev flags (DEBUG) stay in
thresholds.py — those aren't "tune while watching" values, they're
structural decisions.

IMPORTANT for anyone adding a new analytics module: always read
self.config.SOMETHING fresh inside check(), never cache a threshold
value in __init__ — caching would silently freeze that one value at
whatever it was the moment the object was built, defeating the entire
point of live tuning.
"""

from config.thresholds import (
    YOLO_CONFIDENCE_THRESHOLD,
    STATIONARY_DURATION_SEC,
    STATIONARY_PIXEL_THRESHOLD,
    STATIONARY_AREA_CHANGE_THRESHOLD,
    WRONG_WAY_DURATION_SEC,
    WRONG_WAY_COSINE_THRESHOLD,
    HAZARD_CONFIDENCE_THRESHOLD,
    HAZARD_PERSISTENCE_SEC,
    CONGESTION_CAPACITY,
    CONGESTION_STOPPED_DURATION_SEC,
    CONGESTION_STOPPED_PIXEL_THRESHOLD,
)


class LiveConfig:
    """
    Create ONE instance of this in main.py and pass the SAME instance
    into every module that needs a tunable value (YOLOTracker, all
    four analytics detectors) and into TuningPanel. Every attribute
    can be overwritten at any time by a trackbar callback.
    """

    def __init__(self):
        # Seed every value from thresholds.py's defaults, so the panel
        # starts wherever the last-known-good tuning left off, not
        # from some arbitrary hardcoded midpoint.
        self.YOLO_CONFIDENCE_THRESHOLD = YOLO_CONFIDENCE_THRESHOLD
        self.STATIONARY_DURATION_SEC = STATIONARY_DURATION_SEC
        self.STATIONARY_PIXEL_THRESHOLD = STATIONARY_PIXEL_THRESHOLD
        self.STATIONARY_AREA_CHANGE_THRESHOLD = STATIONARY_AREA_CHANGE_THRESHOLD
        self.WRONG_WAY_DURATION_SEC = WRONG_WAY_DURATION_SEC
        self.WRONG_WAY_COSINE_THRESHOLD = WRONG_WAY_COSINE_THRESHOLD
        self.HAZARD_CONFIDENCE_THRESHOLD = HAZARD_CONFIDENCE_THRESHOLD
        self.HAZARD_PERSISTENCE_SEC = HAZARD_PERSISTENCE_SEC
        self.CONGESTION_CAPACITY = CONGESTION_CAPACITY
        self.CONGESTION_STOPPED_DURATION_SEC = CONGESTION_STOPPED_DURATION_SEC
        self.CONGESTION_STOPPED_PIXEL_THRESHOLD = CONGESTION_STOPPED_PIXEL_THRESHOLD

    def snapshot(self):
        """
        Returns a plain dict of every current value. Used by
        TuningPanel.print_snapshot() when you've found a tuning worth
        keeping and want a paste-ready copy for thresholds.py's
        permanent defaults.
        """
        return dict(self.__dict__)
