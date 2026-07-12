"""
test_stationary_unit.py — tests StationaryDetector's LOGIC directly,
by manually building fake history data, bypassing camera/YOLO entirely.
Proves the math (window filtering, wobble measurement, cooldown) is
correct in isolation.
"""

from src.tracker import TrackManager
from src.analytics.stationary import StationaryDetector

# Build a TrackManager and manually stuff in fake data for one
# "car" (ID 1) that barely moves across 6 seconds.
track_manager = TrackManager()

# Simulate: car ID 1, "car" class, sitting almost still from t=0 to t=6
# (tiny 2px jitter each step, well under the 5px STATIONARY_PIXEL_THRESHOLD)
fake_detections_over_time = [
    (0.0, {"id": 1, "class_name": "Car", "confidence": 0.9, "bbox": (100, 200, 150, 250)}),
    (1.0, {"id": 1, "class_name": "Car", "confidence": 0.9, "bbox": (101, 200, 151, 250)}),
    (2.0, {"id": 1, "class_name": "Car", "confidence": 0.9, "bbox": (100, 201, 150, 251)}),
    (3.0, {"id": 1, "class_name": "Car", "confidence": 0.9, "bbox": (102, 200, 152, 250)}),
    (4.0, {"id": 1, "class_name": "Car", "confidence": 0.9, "bbox": (100, 200, 150, 250)}),
    (5.0, {"id": 1, "class_name": "Car", "confidence": 0.9, "bbox": (101, 201, 151, 251)}),
    (6.0, {"id": 1, "class_name": "Car", "confidence": 0.9, "bbox": (100, 200, 150, 250)}),
]

detector = StationaryDetector(track_manager)

fired_timestamps = []
for ts, det in fake_detections_over_time:
    track_manager.update([det], timestamp=ts)
    events = detector.check(timestamp=ts)
    if events:
        fired_timestamps.append(ts)
        print(f"t={ts}: FIRED -> {events}")
    else:
        print(f"t={ts}: no event")

assert fired_timestamps == [5.0], f"Expected one stationary event at t=5.0, got {fired_timestamps}"
print("\nDone. Expect: silence until ~t=4.5-6.0, then exactly one FIRED line.")