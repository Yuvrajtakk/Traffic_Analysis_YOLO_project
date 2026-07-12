"""
Quick manual test for YOLOTracker + TrackManager — proves detection,
tracking, and history-building actually work against real video before
we build analytics modules on top of them.
"""

import time

from src.ingestion import VideoIngestion
from src.tracker import YOLOTracker, TrackManager

SOURCE = "data/test_footage/sample.mp4"        # your existing test video
WEIGHTS = "models/weights/yolov8s.pt"           # stock pretrained weights

# Start video ingestion (same as before)
cap = VideoIngestion(SOURCE, loop_file=True).start()

# Load YOLO + create the shared TrackManager
tracker = YOLOTracker(WEIGHTS, confidence_threshold=0.4, device="cpu")
track_manager = TrackManager()

frame_count = 0
start_time = time.time()

# Run for 15 real seconds — enough frames to actually see history build up
while time.time() - start_time < 15:
    frame = cap.read()

    if frame is None:
        continue  # guard clause, same reasoning as before

    frame_count += 1

    # Run detection + tracking on this single frame
    detections = tracker.track(frame)

    # Feed those detections into the shared TrackManager, using the
    # actual current time as the timestamp
    now = time.time()
    track_manager.update(detections, timestamp=now)

    # Print something readable every 20 frames, not every single one
    if frame_count % 20 == 0:
        active_ids = track_manager.get_active_ids(timestamp=now)
        print(f"\n--- Frame {frame_count} ---")
        print(f"Detections this frame: {len(detections)}")
        print(f"Currently active IDs: {active_ids}")

        # Pick the first active ID (if any) and show its history length
        if active_ids:
            sample_id = active_ids[0]
            history = track_manager.get_history(sample_id)
            print(f"ID {sample_id} history length: {len(history)} entries")
            print(f"ID {sample_id} most recent bbox: {history[-1][1]}")

cap.stop()
print(f"\nTotal frames processed: {frame_count}")
print(f"Total unique IDs ever tracked: {len(track_manager.tracks)}")