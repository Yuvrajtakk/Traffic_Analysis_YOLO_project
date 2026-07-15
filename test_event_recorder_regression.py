import os

import numpy as np

from src.event_recorder import EventRecorder


def test_same_second_events_create_distinct_jpg_files(tmp_path):
    recorder = EventRecorder(output_dir=str(tmp_path))
    frame = np.zeros((16, 16, 3), dtype=np.uint8)

    first_ts = 1700000000.123456
    second_ts = first_ts + 0.000001

    recorder.trigger_event(frame, first_ts, "stationary")
    recorder.trigger_event(frame, second_ts, "stationary")

    jpg_files = sorted(p.name for p in tmp_path.glob("*.jpg"))

    assert len(jpg_files) == 2
    assert jpg_files[0] != jpg_files[1]
    assert all(name.startswith("stationary_") for name in jpg_files)
