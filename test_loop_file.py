"""
test_loop_file.py — confirms loop_file=False actually stops at EOF
instead of looping, now that it's wired up.
"""

import time
from src.ingestion import VideoIngestion

SOURCE = "data/test_footage/sample.mp4"

cap = VideoIngestion(SOURCE, loop_file=False).start()

frame_count = 0
last_frame_count = -1
start_time = time.time()

# Run for up to 10 seconds, but we EXPECT it to plateau well before that
while time.time() - start_time < 10:
    frame = cap.read()
    if frame is not None:
        frame_count += 1
    time.sleep(0.05)  # slow down on purpose so we can watch it plateau

cap.stop()
print(f"Final frame count: {frame_count}")
print(f"running flag at end: {cap.running}")  # should be False