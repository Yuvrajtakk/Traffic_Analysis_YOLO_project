from src.analytics.wrong_way import WrongWayDetector


class DummyConfig:
    WRONG_WAY_DURATION_SEC = 0
    WRONG_WAY_COSINE_THRESHOLD = -0.3


class DummyTrackManager:
    def __init__(self):
        self.tracks = {}


def _bbox_at(cx, cy):
    return (cx - 5, cy - 5, cx + 5, cy + 5)


def test_positive_cosine_does_not_trigger_wrong_way():
    track_manager = DummyTrackManager()
    track_manager.tracks[1009] = {
        "class_name": "Car",
        "history": [
            (0.0, _bbox_at(10, 20)),
            (1.0, _bbox_at(48.7, 20)),
        ],
    }

    detector = WrongWayDetector(
        track_manager=track_manager,
        config=DummyConfig(),
        frame_width=100,
        frame_height=100,
    )

    events = detector.check(timestamp=1.0)

    assert events == []


def test_negative_cosine_can_trigger_wrong_way():
    track_manager = DummyTrackManager()
    track_manager.tracks[42] = {
        "class_name": "Car",
        "history": [
            (0.0, _bbox_at(80, 20)),
            (1.0, _bbox_at(20, 20)),
        ],
    }

    detector = WrongWayDetector(
        track_manager=track_manager,
        config=DummyConfig(),
        frame_width=100,
        frame_height=100,
    )

    events = detector.check(timestamp=1.0)

    assert len(events) == 1
    assert events[0]["id"] == 42
    assert events[0]["cosine"] < 0
