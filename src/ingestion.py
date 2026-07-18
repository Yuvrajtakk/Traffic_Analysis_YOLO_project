import os
import sys
import threading
import time

import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.thresholds import RECONNECT_MAX_RETRIES, RECONNECT_DELAY_SEC, DEFAULT_FPS


class VideoIngestion:
    def __init__(self, source, loop_file=True):
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

        # Normalize common protocol typos so malformed RTSP URLs still open.
        self.source = self._normalize_source(source)

        # this is True/False — is this an RTSP camera link or not?
        # isinstance(source, str) checks "is this a string?"
        # .startswith("rtsp://") checks if the string begins with rtsp://
        # both must be true for is_rtsp to be True
        self.is_rtsp = isinstance(self.source, str) and self.source.startswith("rtsp://")  

        # True only for local video files — NOT webcam (int) and NOT rtsp.
        # We need this to know whether "loop_file" should even apply.
        self.is_file = isinstance(self.source, str) and not self.is_rtsp      

        # self.cap will hold the actual OpenCV camera/video object once opened.
        # None means "not opened yet"
        self.cap = None

        # self.frame is our "whiteboard" — the latest frame, shared between
        # the background thread and the main program. Starts empty.
        self.frame = None           # the shared "latest frame" buffer

        # the "lock" — the rule that only one thread touches self.frame at a time
        self.lock = threading.Lock()  # protects self.frame from race conditions

        # a flag: is the background worker allowed to keep running?
        # we flip this to False when we want everything to stop.
        self.running = False

        # will hold the actual background worker (the Thread object) once started
        self.thread = None  #The dedicated process that handles video capture independently.
        # For local file playback pacing (None for webcam/RTSP)
        self.source_frame_interval = None
        # Wall-clock time of the last successful read (used only for local files)
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
        
        #Check if we are dealing with a live network camera (RTSP)
        if self.is_rtsp:
            
            # Force TCP transport instead of default UDP. 
            # UDP is fast but drops packets, which causes gray, smudged 
            # artifacts in the video. TCP ensures every pixel arrives intact.

            # os.environ allows our Python script to set a temporary environment variable for the operating system.
            # "OPENCV_FFMPEG_CAPTURE_OPTIONS" is a specific setting name. It tells OpenCV's internal video reader (which is powered by FFMPEG) to listen for special instructions.
            # "rtsp_transport;tcp" is the actual instruction. It forces the camera's video stream to travel over the TCP network protocol instead of the default UDP protocol.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            
            # Open the stream. We explicitly tell OpenCV to use the FFMPEG 
            # backend because it handles network streams much better than the defaults.
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            
        else:
            #For local webcams (0) or .mp4 files, just open
            #  it normally.
            cap = cv2.VideoCapture(self.source)

        # Safety check: Did the camera or file actually open successfully?
        # If not, crash loudly and tell us why, rather than failing silently later.
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {self.source}")
        # For local files, read the file's FPS and compute the per-frame interval
        if self.is_file:
            fps = cap.get(cv2.CAP_PROP_FPS)
            try:
                fps_val = float(fps)
            except Exception:
                fps_val = 0.0

            if fps_val and fps_val > 0:
                self.source_frame_interval = 1.0 / fps_val
            else:
                self.source_frame_interval = 1.0 / DEFAULT_FPS
        else:
            # For webcam or RTSP streams, do not throttle
            self.source_frame_interval = None

        # Hand back the successfully opened video object
        return cap

    def start(self):
        """Opens the capture and starts the background reader thread."""

        # Call our helper method to establish the actual connection to the camera/file
        self.cap = self._open_capture()

        # Set our control flag to True. The while loop in _update will run as long as this is True.
        self.running = True

        # set it as a daemon thread (so it dies automatically if the main
        # program exits unexpectedly). This is a safety measure to prevent zombie threads.

        # Create a new background worker. We tell it exactly which function to run 
        # by passing target=self._update (notice there are no parentheses after _update, 
        # because we are handing the function itself to the thread, not running it yet).
        self.thread = threading.Thread(target=self._update)

        # Mark this thread as a daemon. This ties its lifespan to the main program.
        # If the main program stops or crashes, the operating system will automatically 
        # kill this background thread so it doesn't become a zombie process.
        self.thread.daemon = True

        # Actually turn the thread on. This causes the _update() loop to begin executing 
        # simultaneously alongside our main program.
        self.thread.start()

        # Returning 'self' is a convenience that allows chaining method calls later.
        return self
    
    def _update(self):
        """
        Runs forever in the background thread. Reads frames as fast as
        possible and overwrites self.frame.
        """
        # Keep track of how many times we've failed to read a frame in a row
        retries = 0
        
        # 'while self.running:' means "keep doing this indented block of code 
        # forever, until someone changes self.running to False"
        while self.running:
            
            # Try to grab the next picture from the camera/file.
            # 'ret' will be True if successful, False if it failed/ended.
            # 'frame' holds the actual image pixels (if successful).
            ret, frame = self.cap.read()
            
            # 'if not ret:' means "if reading the frame failed"
            if not ret:

                # NEW: if this is a local file, it reached its end, AND the
                # caller explicitly asked us NOT to loop it — stop cleanly
                # instead of reconnecting. This is the only place loop_file
                # actually gets checked/used.
                if self.is_file and not self.loop_file:
                    self.running = False
                    break   
                
                retries += 1  # Add 1 to our failure counter
                
                # Have we failed too many times? (e.g., more than 5 times)
                if retries > RECONNECT_MAX_RETRIES:
                    self.running = False  # Give up, tell the loop to stop
                    break                 # 'break' forces us to exit the 'while' loop completely
                
                # Wait a moment (e.g., 1 second) before trying again
                time.sleep(RECONNECT_DELAY_SEC) 
                
                # Close the broken connection and try opening it again.
                # Wrapped in try/except: _open_capture() raises RuntimeError
                # immediately if the source is genuinely unreachable (e.g.
                # RTSP still down). Without this catch, that exception used
                # to kill this whole background thread on the FIRST failed
                # reconnect attempt, bypassing RECONNECT_MAX_RETRIES entirely.
                # Catching it here lets the retry counter above actually do
                # its job across multiple attempts, same as originally intended.
                self.cap.release()
                try:
                    self.cap = self._open_capture()
                except RuntimeError:
                    continue
                
                continue
                
            # If we get here, we successfully got a frame!
            else:
                retries = 0  # Reset our failure counter back to 0
                # Throttle local file playback to the source video's own FPS
                if self.source_frame_interval is not None:
                    now = time.time()
                    if self._last_read_time is not None:
                        elapsed = now - self._last_read_time
                        remaining = self.source_frame_interval - elapsed
                        if remaining > 0:
                            time.sleep(remaining)
                    # Update last read time after possible sleep
                    self._last_read_time = time.time()

                # Use the lock. This says "Hey main program, don't read the 
                # whiteboard while I am currently erasing and redrawing it!"
                with self.lock:
                    self.frame = frame  # Update the shared whiteboard with the new picture


    def read(self):
        """
        Consumer-facing method. Returns the most recent frame (or None if
        nothing's available yet / stream has stopped). NEVER blocks.
        """
        # Use the lock here too — same rule as before: don't touch the
        # whiteboard while someone else might be mid-write to it.
        with self.lock:

            # If the background thread has given up entirely (retries
            # exhausted, self.running flipped False), stop handing out
            # the last cached frame as if it were still live — that
            # silently freezes analytics on a still image forever,
            # which looks to every downstream module like every vehicle
            # in frame simultaneously stopped moving.
            if not self.running or self.frame is None:
                return None

            # .copy() creates a brand new, separate block of memory with
            # the same pixel data — NOT a reference to the same object.
            return self.frame.copy()

    def stop(self):
        """Cleanly shuts down the background thread and releases the capture."""

        # Flip the flag. The _update() loop checks "while self.running:"
        # on every iteration, so on its NEXT loop pass it'll see this is
        # False and exit on its own. We don't forcibly kill it — we ask
        # nicely and wait.
        self.running = False

        # .join() means: "pause HERE in the main program, and don't move
        # on to the next line, until that background thread has actually
        # finished and exited." Without this, we might try to release the
        # camera while the background thread is still mid-read() on it —
        # a race condition that can crash or hang.
        if self.thread is not None:
            self.thread.join()

        # Now that we're sure the background thread is fully stopped,
        # it's safe to release the camera/file handle.
        if self.cap is not None:
            self.cap.release()


    # Optional but good practice: context manager support so callers can do
    # `with VideoIngestion(source) as cap:`
    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
