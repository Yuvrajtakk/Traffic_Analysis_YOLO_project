"""
test_wrong_way_unit.py — tests WrongWayDetector's LOGIC directly, using
hand-built fake history data. No camera, no YOLO — pure logic check.
"""

from src.tracker import TrackManager
from src.analytics.wrong_way import WrongWayDetector

track_manager = TrackManager()
detector = WrongWayDetector(track_manager)

# Simulate: car ID 1, moving LEFTWARD (x decreasing) steadily —
# AUTHORIZED_FLOW_VECTOR = (1.0, 0.0) means "correct" is RIGHTWARD,
# so this car should eventually get flagged wrong-way.
fake_detections_over_time = [
    (0.0, {"id": 1, "class_name": "Car", "confidence": 0.9, "bbox": (200, 200, 250, 250)}),
    (1.0, {"id": 1, "class_name": "Car", "confidence": 0.9, "bbox": (250, 200, 300, 250)}),
    (2.0, {"id": 1, "class_name": "Car", "confidence": 0.9, "bbox": (300, 200, 350, 250)}),
    (3.0, {"id": 1, "class_name": "Car", "confidence": 0.9, "bbox": (350, 200, 400, 250)}),
    (4.0, {"id": 1, "class_name": "Car", "confidence": 0.9, "bbox": (400, 200, 450, 250)}),
    (5.0, {"id": 1, "class_name": "Car", "confidence": 0.9, "bbox": (450, 200, 500, 250)}),
    (6.0, {"id": 1, "class_name": "Car", "confidence": 0.9, "bbox": (500, 200, 550, 250)}),
]

for ts, det in fake_detections_over_time:
    track_manager.update([det], timestamp=ts)
    events = detector.check(timestamp=ts)
    if events:
        print(f"t={ts}: FIRED -> id={events[0]['id']}, cosine={events[0]['cosine']:.2f}")
    else:
        print(f"t={ts}: no event")

print("\nDone. Expect: silence for a while, then exactly one FIRED line"
      " (cosine should be close to -1.0, since car moves straight left"
      " against a straight-right authorized flow).")