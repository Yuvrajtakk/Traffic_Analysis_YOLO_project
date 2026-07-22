"""
src/ingestion.py

Threaded video ingestion with TWO deliberately different delivery modes:

1. LOCAL FILE  -> LOSSLESS QUEUE. Every single frame of the file is
   delivered to the main loop exactly once, in order — NOTHING is ever
   dropped. If YOLO inference runs slower than the file's FPS, the
   reader thread simply blocks and waits (a file has no "real time" to
   fall behind — it's just bytes on disk). Each frame is delivered
   together with its PTS (presentation timestamp = frame_index / fps,
   i.e. VIDEO time, not wall-clock time), so every downstream module —
   analytics windows, the event recorder's pre/post buffers, and the
   saved MP4 clips — measures time in the video's own clock. This is
   what makes recorded event clips come out at the source's EXACT fps
   with zero skipped frames, no matter how slow the processing machine is.

2. WEBCAM / RTSP -> LATEST-FRAME MODE BY DEFAULT. A live camera cannot
   be paused, so if YOLO processing falls behind, the dashboard uses the
   newest frame instead of waiting through old frames. This keeps the
   dashboard playing at normal real-time speed. Timestamps here are real
   wall-clock time.

read() therefore now returns a (frame, timestamp) PAIR in both modes
(or (None, None) when nothing is available), so main.py can use ONE
consistent clock for everything downstream.
"""

import os
import queue
import sys
import threading
import time

import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.thresholds import RECONNECT_MAX_RETRIES, RECONNECT_DELAY_SEC, DEFAULT_FPS

# How many decoded-but-not-yet-consumed frames the file queue may hold.
# Big enough to smooth out momentary slowdowns (a single slow YOLO
# frame), small enough that memory stays bounded (~64 frames of 720p
# BGR is ~170 MB worst case at 1080p — acceptable, and usually far less).
FILE_QUEUE_MAX_FRAMES = 64


class VideoIngestion:
    def __init__(self, source, loop_file=True, drop_live_frames=True):
        """
        source: int (webcam index), or str (file path OR rtsp:// / http:// URL)
        loop_file: if True, local video files restart from frame 0 on EOF
        instead of stopping (keeps threaded logic path consistent
        for both files and live streams)
        """
        self.raw_source = source

        # if True, when a local video file ends, we restart it from the
        # beginning instead of stopping. Just a setting we remember.
        self.loop_file = loop_file
        self.drop_live_frames = drop_live_frames

        # Normalize common protocol typos so malformed RTSP URLs still open.
        self.source = self._normalize_source(source)

        # is this an RTSP camera link or not?
        self.is_rtsp = isinstance(self.source, str) and self.source.startswith("rtsp://")

        # True only for local video files — NOT webcam (int) and NOT rtsp.
        self.is_file = isinstance(self.source, str) and not self.is_rtsp

        # self.cap will hold the actual OpenCV camera/video object once opened.
        self.cap = None

        # ── LIVE latest-frame mode state: the "latest frame" whiteboard ──
        self.frame = None
        self.lock = threading.Lock()

        # ── Ordered delivery queue ───────────────────────────────────────
        # Each queue item is a (frame, pts_seconds) tuple. maxsize makes
        # put() BLOCK when the consumer falls behind — that blocking IS
        # the no-drop guarantee for files and smooth live mode: the reader
        # waits instead of overwriting.
        self.frame_queue = queue.Queue(maxsize=FILE_QUEUE_MAX_FRAMES)

        # Monotonic count of frames successfully read from the file since
        # start(). PTS = frames_read * frame_interval. NOT reset when the
        # file loops — timestamps must keep increasing forever, or every
        # time-window comparison downstream would see time run backwards.
        self._frames_read = 0

        # a flag: is the background worker allowed to keep running?
        self.running = False

        # will hold the background reader Thread once started
        self.thread = None

        # The source's own frame rate. For files this is read from the
        # container in _open_capture(); for live sources it stays None
        # (many RTSP cameras lie about FPS anyway). Exposed publicly so
        # main.py can display it and compare against measured FPS.
        self.source_fps = None
        self.source_frame_interval = None

        # Wall-clock time of the last successful file read — used only to
        # pace file playback down to real time when processing is FASTER
        # than the source FPS (so the dashboard doesn't fast-forward).
        self._last_read_time = None

    @staticmethod
    def _normalize_source(source):
        if isinstance(source, str):
            normalized = source.strip()
            if normalized.startswith("rstp://"):
                normalized = "rtsp://" + normalized[len("rstp://"):]
            return normalized
        return source

    def _open_capture(self):
        """Opens (or re-opens) the underlying cv2.VideoCapture."""

        if self.is_rtsp:
            # Force TCP transport instead of default UDP — UDP drops
            # packets, which causes gray smudged artifacts in the video.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        else:
            # local webcams (0) or .mp4 files open normally
            cap = cv2.VideoCapture(self.source)

        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {self.source}")

        # For local files, read the file's FPS once and derive the
        # per-frame interval — this drives BOTH the PTS timestamps and
        # the real-time pacing below.
        if self.is_file:
            fps = cap.get(cv2.CAP_PROP_FPS)
            try:
                fps_val = float(fps)
            except Exception:
                fps_val = 0.0

            if not (fps_val and fps_val > 0):
                fps_val = float(DEFAULT_FPS)

            self.source_fps = fps_val
            self.source_frame_interval = 1.0 / fps_val
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            try:
                fps_val = float(fps)
            except Exception:
                fps_val = 0.0

            if not (fps_val and fps_val > 0):
                fps_val = float(DEFAULT_FPS)

            self.source_fps = fps_val
            self.source_frame_interval = 1.0 / fps_val

        return cap

    def start(self):
        """Opens the capture and starts the background reader thread."""
        self.cap = self._open_capture()
        self.running = True

        # Daemon thread: dies automatically with the main program, so a
        # crash can never leave a zombie reader behind.
        self.thread = threading.Thread(target=self._update)
        self.thread.daemon = True
        self.thread.start()
        return self

    # ─────────────────────── background reader ───────────────────────

    def _update(self):
        """Runs in the background thread until stop() flips self.running."""
        retries = 0

        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                # Local file reached its end and the caller asked NOT to
                # loop — stop cleanly instead of reconnecting.
                if self.is_file and not self.loop_file:
                    self.running = False
                    break

                if self.is_file and self.loop_file:
                    # EOF on a looping file: seek back to frame 0 and keep
                    # going. IMPORTANT: _frames_read is NOT reset — PTS
                    # must stay monotonically increasing across loops.
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                retries += 1
                if retries > RECONNECT_MAX_RETRIES:
                    self.running = False
                    break

                time.sleep(RECONNECT_DELAY_SEC)

                # try/except: _open_capture raises immediately if the
                # source is still unreachable — without catching, one
                # failed reconnect would kill this whole thread and
                # bypass RECONNECT_MAX_RETRIES entirely.
                self.cap.release()
                try:
                    self.cap = self._open_capture()
                except RuntimeError:
                    continue
                continue

            # ── successful read ──
            retries = 0

            if self.is_file:
                self._deliver_file_frame(frame)
            elif self.drop_live_frames:
                # live source: overwrite the whiteboard, dropping any
                # frame the consumer didn't get to in time — correct
                # when minimum latency matters more than smooth output.
                with self.lock:
                    self.frame = frame
            else:
                self._deliver_live_frame(frame)

    def _deliver_file_frame(self, frame):
        """
        FILE mode delivery: push (frame, pts) onto the lossless queue.

        Two pacing forces act here, covering both speed mismatches:
        - consumer FASTER than source fps -> the sleep below throttles
          reading down to real time, so playback speed stays correct;
        - consumer SLOWER than source fps -> queue.put() blocks when the
          queue is full, so the reader WAITS instead of overwriting.
          Zero frames dropped either way.
        """
        # Video-time timestamp for this frame, BEFORE incrementing:
        # frame 0 gets pts 0.0, frame 1 gets one interval, and so on.
        pts = self._frames_read * self.source_frame_interval
        self._frames_read += 1

        # Real-time pacing (only matters when we're reading faster than
        # the source fps — e.g. GPU inference breezing through a file).
        now = time.time()
        if self._last_read_time is not None:
            remaining = self.source_frame_interval - (now - self._last_read_time)
            if remaining > 0:
                time.sleep(remaining)
        self._last_read_time = time.time()

        # Blocking put with a short timeout in a loop, NOT a bare
        # put(): if stop() flips self.running while the queue is full,
        # a bare blocking put would deadlock this thread forever.
        while self.running:
            try:
                self.frame_queue.put((frame, pts), timeout=0.1)
                return
            except queue.Full:
                continue

    def _deliver_live_frame(self, frame):
        """
        Smooth live delivery: enqueue decoded frames in order with a
        wall-clock timestamp from the moment the frame reached the app.
        """
        timestamp = time.time()
        while self.running:
            try:
                self.frame_queue.put((frame, timestamp), timeout=0.1)
                return
            except queue.Full:
                continue

    # ───────────────────────── consumer side ─────────────────────────

    def read(self):
        """
        Consumer-facing method. Returns (frame, timestamp):

        - FILE mode:  the NEXT unseen frame in order + its video-time
          PTS in seconds. Lossless — every frame comes through exactly
          once. Waits up to ~10ms for one to arrive, then gives up with
          (None, None) so the caller's loop never hard-blocks.
        - LIVE default mode: the newest frame + wall-clock time.time().
        - LIVE smooth mode, if drop_live_frames=False: the next decoded
          live frame in order + wall-clock timestamp.
        - (None, None) when nothing is available / the stream stopped.
        """
        if self.is_file or not self.drop_live_frames:
            try:
                # Short timeout instead of get_nowait(): yields the CPU
                # while waiting, so main.py's retry-continue loop doesn't
                # spin at 100% between frames.
                return self.frame_queue.get(timeout=0.01)
            except queue.Empty:
                # Distinguish "reader finished AND queue fully drained"
                # (stream truly over -> None forever) from "reader alive,
                # next frame just not decoded yet" (transient None).
                return (None, None)

        # Low-latency live mode — same semantics as before, plus a timestamp
        with self.lock:
            if not self.running or self.frame is None:
                # Once the background thread has given up, stop handing
                # out the last cached frame as if it were live — that
                # would freeze analytics on a still image forever.
                return (None, None)
            return (self.frame.copy(), time.time())

    def is_finished(self):
        """
        True once the source is definitively over: the reader thread has
        stopped AND (for files) every queued frame has been consumed.
        Lets main.py exit cleanly at EOF instead of spinning forever.
        """
        if self.running:
            return False
        if self.is_file:
            return self.frame_queue.empty()
        return True

    def stop(self):
        """Cleanly shuts down the background thread and releases the capture."""
        self.running = False

        # Wait for the reader thread to actually exit before touching
        # self.cap — releasing mid-read is a race that can crash or hang.
        # (The reader can't deadlock on a full queue: _deliver_file_frame
        # re-checks self.running every 0.1s while trying to put.)
        if self.thread is not None:
            self.thread.join()

        if self.cap is not None:
            self.cap.release()

    # Context manager support: `with VideoIngestion(source) as cap:`
    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
