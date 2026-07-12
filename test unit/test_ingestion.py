"""
Quick manual test for VideoIngestion — not part of the final system,
just here to prove the threaded read/reconnect logic actually works
before we build anything on top of it.
"""

import time
from src.ingestion import VideoIngestion

# CHANGE this path to wherever you put your test video
SOURCE = "data/test_footage/sample.mp4"

# start() opens the file and kicks off the background thread
cap = VideoIngestion(SOURCE, loop_file=True).start()

frame_count = 0
start_time = time.time()

# run for 5 real seconds, just calling read() over and over,
# same way your main loop eventually will
while time.time() - start_time < 5:
    frame = cap.read()

    if frame is not None:
        frame_count += 1
        # print occasionally, not every frame, so the terminal isn't spammed
        if frame_count % 20 == 0:
            print(f"Got frame #{frame_count}, shape: {frame.shape}")

cap.stop()
print(f"\nTotal frames read in 5 seconds: {frame_count}")