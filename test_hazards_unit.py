from src.analytics.hazards import HazardDetector


class DummyConfig:
    HAZARD_CONFIDENCE_THRESHOLD = 0.25
    HAZARD_PERSISTENCE_SEC = 1


SMOKE_DETECTION = {
    "class_name": "Smoke",
    "confidence": 0.9,
    "bbox": (10, 10, 20, 20),
}


def test_smoke_one_frame_flicker_does_not_reset_persistence():
    detector = HazardDetector(DummyConfig())

    assert detector.check([SMOKE_DETECTION], timestamp=0.0) == []
    assert detector.check([SMOKE_DETECTION], timestamp=0.5) == []
    assert detector.check([SMOKE_DETECTION], timestamp=1.0) == []

    # One missed frame is tolerated because Smoke is still present in
    # 3 of the last 4 frames.
    assert detector.check([], timestamp=1.5) == []
    assert detector.hazard_since["Smoke"] == 1.0

    events = detector.check([SMOKE_DETECTION], timestamp=2.1)

    assert len(events) == 1
    assert events[0]["class_name"] == "Smoke"
