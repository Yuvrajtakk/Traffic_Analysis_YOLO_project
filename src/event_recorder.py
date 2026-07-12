"""
src/event_recorder.py

Timestamp-based ring buffer + threaded MP4/JPG event archiving.

Per decision log 3.4: buffer is filtered by TIME (seconds), not a fixed
frame count — real FPS drifts (RTSP instability, slow CPU), so a
frame-count buffer would silently become the wrong duration. Disk
writes happen in a background thread so the main loop never blocks.

DESIGN: supports MULTIPLE simultaneous in-progress event collections
(e.g. two different vehicles triggering different events within a few
frames of each other) — each pending collection tracks its own state
independently, rather than one shared global "currently collecting"
flag that could only handle one event at a time.
"""

import os
import time
import threading
from collections import deque

import cv2

from config.thresholds import (
    PRE_EVENT_SEC,
    POST_EVENT_SEC,
)


class EventRecorder:
    def __init__(self, output_dir="outputs/events"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # The ALWAYS-RUNNING ring buffer — the "rolling security tape."
        # Every frame gets added here regardless of whether any event
        # fired. No maxlen here — size is managed by TIME (see
        # add_frame's trim step below), not a fixed frame count, since
        # a frame-count cap is exactly the FPS-drift trap this project
        # deliberately avoids everywhere else.
        self.buffer = deque()

        # List of currently in-progress "collect the after-frames" jobs.
        # Each entry is an independent dict — this is what lets MULTIPLE
        # overlapping events (two different vehicles triggering events
        # a moment apart) each accumulate their own post-event frames
        # without interfering with each other. Starts empty.
        self.pending_collections = []

    def add_frame(self, frame, timestamp=None):
        """
        Call ONCE PER FRAME, unconditionally — the "always recording"
        ring buffer step, independent of whether any event fired.
        """
        if timestamp is None:
            timestamp = time.time()

        # Add this frame to the rolling tape. .copy() so this stored
        # snapshot can never be silently mutated by anything else in
        # main.py that still holds a reference to the original frame.
        self.buffer.append((timestamp, frame.copy()))

        # Trim anything older than PRE_EVENT_SEC off the FRONT. `while`,
        # not `if` — if the main loop briefly lagged, MULTIPLE stale
        # entries could need removing at once, not just one.
        while self.buffer and (timestamp - self.buffer[0][0]) > PRE_EVENT_SEC:
            self.buffer.popleft()

        # Every currently in-progress "after" collection independently
        # receives this same new frame — this is what allows several
        # events to be mid-collection at once without conflicting.
        for entry in self.pending_collections:
            entry["after_frames"].append((timestamp, frame.copy()))

        # Check which pending collections have now gathered a full
        # POST_EVENT_SEC of after-frames and are ready to be finalized.
        # Collect the ready ones into a separate list first — same
        # "collect, then remove" pattern as TrackManager's purge —
        # since we can't safely remove list items while iterating the
        # same list.
        ready_entries = []
        for entry in self.pending_collections:
            if (timestamp - entry["trigger_timestamp"]) >= POST_EVENT_SEC:
                ready_entries.append(entry)

        for entry in ready_entries:
            self._dispatch_to_background_writer(entry)
            self.pending_collections.remove(entry)

    def trigger_event(self, frame, timestamp, event_type, metadata=None):
        """
        Call this the moment any analytics module fires an event.
        Immediately saves an annotated .jpg, and starts a NEW pending
        collection to gather the next POST_EVENT_SEC of frames.
        """
        if metadata is None:
            metadata = {}

        # Step A: save the annotated frame as a still image right now.
        # Fast (milliseconds) — no threading needed for a single image.
        filename = f"{event_type}_{int(timestamp)}.jpg"
        cv2.imwrite(os.path.join(self.output_dir, filename), frame)

        # Step B: snapshot the CURRENT rolling buffer as this event's
        # "before" half. list(self.buffer) creates a genuinely SEPARATE
        # list with its own copies — critical, because self.buffer
        # keeps changing (growing, trimming) every subsequent frame.
        # Without list(), before_frames would just be another name for
        # the SAME live deque, and would silently shrink/change later
        # as add_frame() keeps trimming it — corrupting this snapshot.
        before_frames = list(self.buffer)

        # Register this as a new in-progress collection. after_frames
        # starts empty — add_frame()'s loop above will fill it in over
        # the next POST_EVENT_SEC, frame by frame, automatically.
        new_entry = {
            "event_id": f"{event_type}_{int(timestamp * 1000)}",
            "event_type": event_type,
            "trigger_timestamp": timestamp,
            "before_frames": before_frames,
            "after_frames": [],
        }
        self.pending_collections.append(new_entry)

    def _dispatch_to_background_writer(self, entry):
        """
        Hands ONE finished (before+after complete) collection off to a
        background thread to actually write to disk. Main loop never
        waits — no .join() here, deliberately.
        """
        # Each call creates its OWN independent thread — if several
        # events finish around the same time, each gets its own
        # helper, writing its own clip, none blocking on each other.
        thread = threading.Thread(
            target=self._write_clip, args=(entry,), daemon=True
        )
        # daemon=True: this thread dies automatically if the main
        # program exits, same reasoning as VideoIngestion's thread —
        # no lingering zombie background work after shutdown.
        thread.start()

    def _write_clip(self, entry):
        """
        Runs in a BACKGROUND THREAD. Combines before+after frames and
        writes them to an .mp4 file. Slow — this is exactly why it's
        threaded rather than running on the main loop.
        """
        # Glue the two piles together into one continuous sequence.
        all_frames = entry["before_frames"] + entry["after_frames"]

        if len(all_frames) < 2:
            # Not enough real data to make a meaningful clip (e.g. an
            # event fired right at startup, before the buffer had
            # anything in it yet) — skip rather than write junk.
            return

        # Real playback speed, derived from ACTUAL elapsed time between
        # the first and last collected frame — not an assumed constant.
        # This is the same timestamp-based philosophy used everywhere
        # else in this project (TrackManager, all four analytics
        # modules) — real FPS can drift, so we measure it directly.
        duration = all_frames[-1][0] - all_frames[0][0]
        fps = len(all_frames) / duration if duration > 0 else 20

        # Ask the very first frame how big it is — cv2's VideoWriter
        # needs to know the exact frame size upfront.
        height, width = all_frames[0][1].shape[:2]

        path = os.path.join(self.output_dir, entry["event_id"] + ".mp4")

        # mp4v is a widely-compatible codec choice — a safe default for
        # .mp4 output without requiring extra codec installs.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

        # Write every collected frame, in order, into the video file.
        for ts, frame in all_frames:
            writer.write(frame)

        # Finalizes the file so it's a valid, playable video — without
        # this, the file could be left in a broken/incomplete state.
        writer.release()