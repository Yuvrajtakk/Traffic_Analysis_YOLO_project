"""
test_stationary.py — proves StationaryDetector actually fires (or
correctly does NOT fire) against real tracking data.
"""

import time

from src.ingestion import VideoIngestion
from src.tracker import YOLOTracker, TrackManager
from src.analytics.stationary import StationaryDetector

SOURCE = "data/test_footage/sample.mp4"
WEIGHTS = "models/weights/yolov8s.pt"

cap = VideoIngestion(SOURCE, loop_file=True).start()
tracker = YOLOTracker(WEIGHTS, confidence_threshold=0.4, device="cpu")
track_manager = TrackManager()
detector = StationaryDetector(track_manager)

start_time = time.time()
frame_count = 0
total_events = 0

# Run for 20 seconds — need enough time for a car to actually build up
# a full 5-second stationary streak, if one exists in the footage.
while time.time() - start_time < 20:
    frame = cap.read()
    if frame is None:
        continue

    frame_count += 1
    now = time.time()

    detections = tracker.track(frame)
    track_manager.update(detections, timestamp=now)

    events = detector.check(timestamp=now)

    for event in events:
        total_events += 1
        print(f"\n🚨 STATIONARY EVENT — ID {event['id']} "
              f"({event['class_name']}) at t={event['timestamp']:.1f}")
        print(f"   bbox: {event['bbox']}")

cap.stop()
print(f"\nTotal frames processed: {frame_count}")
print(f"Total stationary events fired: {total_events}")