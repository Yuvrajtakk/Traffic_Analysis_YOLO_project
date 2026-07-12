"""
test_congestion_unit.py — tests CongestionDetector's logic directly,
using hand-built fake detections fed through the real TrackManager.
No camera, no YOLO needed.
"""

from src.tracker import TrackManager
from src.analytics.congestion import CongestionDetector

track_manager = TrackManager()

# Assume a 1000x1000 pixel frame for simplicity.
FRAME_W, FRAME_H = 1000, 1000
detector = CongestionDetector(track_manager, FRAME_W, FRAME_H)

# ROI (from thresholds.py defaults) roughly covers x:50-950, y:350-950
# in pixel terms on a 1000x1000 frame. We'll place fake cars WELL
# inside that region.

print("Test 1: 5 vehicles inside ROI, capacity=5 (from thresholds.py)"
    " -> 5 is NOT > 5, should NOT fire")
fake_dets_5 = [
    {"id": i, "class_name": "Car", "confidence": 0.9,
     "bbox": (100 + i * 100, 500, 150 + i * 100, 550)}
    for i in range(5)
]
track_manager.update(fake_dets_5, timestamp=1.0)
events = detector.check(timestamp=1.0)
print(f"  count check -> {'FIRED: ' + str(events) if events else 'no event'}")

print("\nTest 2: 7 vehicles inside ROI -> 7 > 5, SHOULD fire")
track_manager2 = TrackManager()
detector2 = CongestionDetector(track_manager2, FRAME_W, FRAME_H)
fake_dets_7 = [
    {"id": i, "class_name": "Car", "confidence": 0.9,
     "bbox": (100 + i * 100, 500, 150 + i * 100, 550)}
    for i in range(7)
]
track_manager2.update(fake_dets_7, timestamp=1.0)
events2 = detector2.check(timestamp=1.0)
print(f"  count check -> {'FIRED: ' + str(events2) if events2 else 'no event'}")

print("\nTest 3: 7 vehicles, but OUTSIDE the ROI (e.g. y=50, above the"
    " road region) -> should NOT fire, none actually in ROI")
track_manager3 = TrackManager()
detector3 = CongestionDetector(track_manager3, FRAME_W, FRAME_H)
fake_dets_outside = [
    {"id": i, "class_name": "Car", "confidence": 0.9,
     "bbox": (100 + i * 100, 50, 150 + i * 100, 100)}  # y=50, above ROI
    for i in range(7)
]
track_manager3.update(fake_dets_outside, timestamp=1.0)
events3 = detector3.check(timestamp=1.0)
print(f"  count check -> {'FIRED: ' + str(events3) if events3 else 'no event'}")

print("\nTest 4: 7 PEOPLE inside ROI (not vehicles) -> should NOT fire,"
    " Person isn't in VEHICLE_CLASSES")
track_manager4 = TrackManager()
detector4 = CongestionDetector(track_manager4, FRAME_W, FRAME_H)
fake_dets_people = [
    {"id": i, "class_name": "Person", "confidence": 0.9,
     "bbox": (100 + i * 100, 500, 150 + i * 100, 550)}
    for i in range(7)
]
track_manager4.update(fake_dets_people, timestamp=1.0)
events4 = detector4.check(timestamp=1.0)
print(f"  count check -> {'FIRED: ' + str(events4) if events4 else 'no event'}")