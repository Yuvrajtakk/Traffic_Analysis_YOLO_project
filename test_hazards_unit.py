"""
test_hazards_unit.py — tests HazardDetector's logic directly, using
hand-built fake per-frame detections. No camera, no YOLO needed —
COCO weights don't even have Fire/Smoke/Accident classes to test with.
"""

from src.analytics.hazards import HazardDetector

detector = HazardDetector()

# Simulate: "Fire" confidently detected every frame from t=0.0 to t=1.5,
# with NO gaps — should fire once, right around t=1.0
# (HAZARD_PERSISTENCE_SEC = 1.0 per your thresholds.py)
frames = [
    (0.0, [{"class_name": "Fire", "confidence": 0.90, "bbox": (100, 100, 200, 200)}]),
    (0.5, [{"class_name": "Fire", "confidence": 0.88, "bbox": (101, 100, 200, 201)}]),
    (1.0, [{"class_name": "Fire", "confidence": 0.92, "bbox": (100, 101, 201, 200)}]),
    (1.5, [{"class_name": "Fire", "confidence": 0.85, "bbox": (99, 100, 200, 200)}]),
    (2.0, [{"class_name": "Fire", "confidence": 0.91, "bbox": (100, 100, 200, 200)}]),
    (2.5, [{"class_name": "Fire", "confidence": 0.89, "bbox": (100, 100, 200, 200)}]),
    (3.0, [{"class_name": "Fire", "confidence": 0.90, "bbox": (100, 100, 200, 200)}]),
]

for ts, dets in frames:
    events = detector.check(dets, timestamp=ts)
    if events:
        print(f"t={ts}: FIRED -> {events}")
    else:
        print(f"t={ts}: no event")

print("\nTest 2: same fire, but LOW confidence throughout — should NEVER fire")
detector2 = HazardDetector()
frames_low_conf = [
    (0.0, [{"class_name": "Fire", "confidence": 0.10, "bbox": (100, 100, 200, 200)}]),
    (0.5, [{"class_name": "Fire", "confidence": 0.15, "bbox": (100, 100, 200, 200)}]),
    (1.0, [{"class_name": "Fire", "confidence": 0.20, "bbox": (100, 100, 200, 200)}]),
    (1.5, [{"class_name": "Fire", "confidence": 0.05, "bbox": (100, 100, 200, 200)}]),
]
for ts, dets in frames_low_conf:
    events = detector2.check(dets, timestamp=ts)
    print(f"t={ts}: {'FIRED -> ' + str(events) if events else 'no event'}")

print("\nTest 3: fire with a GAP at t=0.5 (missing that frame) — streak should reset")
detector3 = HazardDetector()
frames_gap = [
    (0.0, [{"class_name": "Fire", "confidence": 0.90, "bbox": (100, 100, 200, 200)}]),
    (0.4, [{"class_name": "Fire", "confidence": 0.90, "bbox": (100, 100, 200, 200)}]),
    (0.8, []),  # gap!
    (1.2, [{"class_name": "Fire", "confidence": 0.90, "bbox": (100, 100, 200, 200)}]),
    (1.6, [{"class_name": "Fire", "confidence": 0.90, "bbox": (100, 100, 200, 200)}]),
    (2.0, [{"class_name": "Fire", "confidence": 0.90, "bbox": (100, 100, 200, 200)}]),
    (2.4, [{"class_name": "Fire", "confidence": 0.90, "bbox": (100, 100, 200, 200)}]),
    (2.8, [{"class_name": "Fire", "confidence": 0.90, "bbox": (100, 100, 200, 200)}]),
]
for ts, dets in frames_gap:
    events = detector3.check(dets, timestamp=ts)
    print(f"t={ts}: {'FIRED -> ' + str(events) if events else 'no event'}")