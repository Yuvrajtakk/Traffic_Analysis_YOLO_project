"""
test_event_recorder.py — proves EventRecorder actually writes real
.jpg + .mp4 files, including handling TWO overlapping events at once.
No camera, no YOLO — synthetic frames built with numpy.
"""

import os
import time
import numpy as np

from src.event_recorder import EventRecorder

# Clean output dir before testing, so we can clearly see what THIS run produced
OUTPUT_DIR = "outputs/events"
recorder = EventRecorder(output_dir=OUTPUT_DIR)

# A simple fake "frame" — just a solid gray image, correct shape/type
# for OpenCV to accept (480 height, 640 width, 3 color channels).
def make_fake_frame(brightness=100):
    return np.full((480, 640, 3), brightness, dtype=np.uint8)

print("Simulating a video feed: 6 seconds of frames at ~20fps...")

start_time = time.time()
frame_count = 0
triggered_A = False
triggered_B = False

# Run for 6 real seconds (needs to comfortably exceed PRE_EVENT_SEC +
# POST_EVENT_SEC = 2 + 2 = 4, so both before/after windows fully close)
while time.time() - start_time < 6:
    now = time.time()
    frame = make_fake_frame()

    # Always feed the rolling buffer, every frame, unconditionally
    recorder.add_frame(frame, timestamp=now)

    # Fire a fake "stationary" event once, partway through
    if not triggered_A and (now - start_time) > 1.5:
        print(f"  -> triggering EVENT A at t={now - start_time:.2f}s")
        recorder.trigger_event(frame, now, "stationary", metadata={"id": 1})
        triggered_A = True

    # Fire a SECOND, overlapping fake "wrong_way" event shortly after —
    # this tests that two pending collections can run at once
    if not triggered_B and (now - start_time) > 2.0:
        print(f"  -> triggering EVENT B at t={now - start_time:.2f}s (overlaps with A)")
        recorder.trigger_event(frame, now, "wrong_way", metadata={"id": 2})
        triggered_B = True

    frame_count += 1
    time.sleep(0.05)  # ~20fps pacing

print(f"\nProcessed {frame_count} frames over 6 seconds.")

# Give background writer threads a moment to actually finish writing
print("Waiting 2s for background writer threads to finish...")
time.sleep(2)

# Check what actually landed on disk
print(f"\nFiles in {OUTPUT_DIR}:")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    if fname == ".gitkeep":
        continue
    full_path = os.path.join(OUTPUT_DIR, fname)
    size = os.path.getsize(full_path)
    print(f"  {fname}  ({size} bytes)")