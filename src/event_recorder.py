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
from datetime import datetime

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
        if metadata is None:
            metadata = {}

        module = event_type  # now the module name: stationary/wrong_way/hazard/congestion
        class_name = metadata.get("class_name", "unknown")
        track_id_str = metadata.get("id", "NA")
        # Event math uses the caller's timestamp, which is video PTS for
        # files. Filenames should stay human-readable wall-clock time.
        wall_time = datetime.now()
        readable_time = wall_time.strftime("%H%M%S_%f")[:-3]

        event_id = f"{module}_{class_name}_id{track_id_str}_{readable_time}"
        filename = f"{event_id}.jpg"
        cv2.imwrite(os.path.join(self.output_dir, filename), frame)

        extra = {k: v for k, v in metadata.items() if k not in ("class_name", "id")}
        print(
            f"[EVENT TRIGGERED] id={event_id} | module={module} | class={class_name} "
            f"| track_id={track_id_str} | time={wall_time.strftime('%H:%M:%S')} "
            f"| video_ts={timestamp:.3f}s "
            f"| meta={extra}"
        )

        before_frames = list(self.buffer)

        new_entry = {
            "event_id": event_id,
            "event_type": module,
            "trigger_timestamp": timestamp,
            "before_frames": before_frames,
            "after_frames": [],
        }
        self.pending_collections.append(new_entry)

    def _dispatch_to_background_writer(self, entry):
        thread = threading.Thread(
            target=self._write_clip, args=(entry,), daemon=True
        )
        thread.start()

    def _write_clip(self, entry):
        all_frames = entry["before_frames"] + entry["after_frames"]

        if len(all_frames) < 2:
            print(f"[EVENT SKIPPED] id={entry['event_id']} — not enough frames to write a clip")
            return

        duration = all_frames[-1][0] - all_frames[0][0]
        # (N frames span N-1 intervals, so fps = (N-1)/duration, NOT
        # N/duration — the off-by-one would make every saved clip play
        # slightly fast. With the ingestion layer's PTS timestamps the
        # frames are exactly 1/source_fps apart, so this now recovers
        # the source's EXACT fps.)
        fps = (len(all_frames) - 1) / duration if duration > 0 else 20

        height, width = all_frames[0][1].shape[:2]
        path = os.path.join(self.output_dir, entry["event_id"] + ".mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        for ts, frame in all_frames:
            writer.write(frame)
        writer.release()

        print(
            f"[EVENT SAVED] id={entry['event_id']} | frames={len(all_frames)} "
            f"| duration={duration:.2f}s | fps={fps:.1f} | file={path}"
        )
