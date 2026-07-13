"""
main.py

Wires every built module into one real-time loop:
ingestion -> tracker -> TrackManager -> 4 analytics modules -> event_recorder -> display.

Keyboard toggles gate whether each analytics module's check() is even
CALLED this frame — not just whether its output is shown. Turning a
switch OFF genuinely skips that module's work entirely.
"""

import time

import cv2

from src.ingestion import VideoIngestion
from src.tracker import YOLOTracker, TrackManager
from src.analytics.stationary import StationaryDetector
from src.analytics.wrong_way import WrongWayDetector
from src.analytics.hazards import HazardDetector
from src.analytics.congestion import CongestionDetector
from src.event_recorder import EventRecorder


SOURCE = "data/test_footage/sample.mp4"
WEIGHTS = "models/weights/best.pt"

MODULE_ORDER = (
    ("stationary", "S"),
    ("wrong_way", "W"),
    ("hazards", "H"),
    ("congestion", "C"),
)
KEY_TOGGLE_MAP = {
    ord("s"): "stationary",
    ord("w"): "wrong_way",
    ord("h"): "hazards",
    ord("c"): "congestion",
}


def toggle_module_state(module_state, key):
    module_name = KEY_TOGGLE_MAP.get(key)
    if module_name is None:
        return
    module_state[module_name] = not module_state[module_name]


def build_status_text(module_state):
    status_parts = []
    for module_name, label in MODULE_ORDER:
        status_parts.append(f"{label}:{'ON' if module_state[module_name] else 'OFF'}")
    return " ".join(status_parts)


def should_quit(key, window_name):
    if key == ord("q"):
        return True

    try:
        return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return False


def resize_frame_for_display(frame, max_width=1280, max_height=720):
    height, width = frame.shape[:2]
    if width <= max_width and height <= max_height:
        return frame

    scale = min(max_width / width, max_height / height)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


def main():
    # ================= PIECE 1: build every module =================
    # VideoIngestion opens the source and starts its background thread
    # immediately. From this point on, cap.read() always gives us
    # whatever the most recent frame is, without ever blocking.
    cap = VideoIngestion(SOURCE).start()

    # YOLOTracker has no .start() — the model is fully loaded into
    # memory the instant __init__ finishes. Nothing left to "start."
    tracker = YOLOTracker(weights_path=WEIGHTS)

    # ONE shared TrackManager — every analytics module below reads
    # from this SAME object, so none of them can ever disagree about
    # whether a given ID is still "alive."
    track_manager = TrackManager()

    # PRIME THE FRAME: CongestionDetector needs real frame_width and
    # frame_height to build its ROI polygon, and those numbers don't
    # exist until we've actually read one real frame. Keep asking
    # cap.read() until it stops returning None. This also guarantees
    # every frame we touch from here on is real, valid image data —
    # never garbage, never corrupted.
    first_frame = None
    while first_frame is None:
        first_frame = cap.read()
        if first_frame is None:
            time.sleep(0.05)  # don't spin the CPU at 100% while waiting

    # .shape gives (height, width, channels) — note the ORDER.
    frame_height, frame_width = first_frame.shape[:2]

    # Build the four analytics detectors — only NOW, after
    # frame_width/frame_height actually exist.
    stationary_detector = StationaryDetector(track_manager)
    wrong_way_detector = WrongWayDetector(track_manager)
    hazard_detector = HazardDetector()  # no track_manager — works off raw detections directly
    congestion_detector = CongestionDetector(track_manager, frame_width, frame_height)

    event_recorder = EventRecorder()

    # ================= PIECE 2: window + keyboard toggles =================
    window_name = "Traffic Dashboard"
    module_state = {
        "stationary": True,
        "wrong_way": True,
        "hazards": True,
        "congestion": True,
    }
    cv2.namedWindow(window_name)

    # ================= PIECE 3-7: the real-time loop =================
    while True:
        frame = cap.read()
        if frame is None:
            # background thread hasn't got a frame ready yet — skip
            # this spin entirely, try again next time round
            continue

        # ---- Piece 4: run YOLO + update shared tracking state ----
        # grabbed ONCE so every module this frame agrees on "what time
        # it is" — avoids tiny timestamp mismatches between modules
        now = time.time()
        detections = tracker.track(frame)
        track_manager.update(detections, timestamp=now)

        # ---- Piece 5: keyboard-toggle-gated analytics checks ----
        # start every result as an empty list FIRST. That way "switch
        # was OFF" and "switch was ON but nothing triggered" look
        # exactly the same to the code below — always a list, never
        # missing, never None.
        stationary_events = []
        wrong_way_events = []
        hazard_events = []
        congestion_events = []

        if module_state["stationary"]:
            stationary_events = stationary_detector.check(timestamp=now)

        if module_state["wrong_way"]:
            wrong_way_events = wrong_way_detector.check(timestamp=now)

        if module_state["hazards"]:
            # only this one needs the raw detections list — it has no
            # TrackManager, no history, it just reads what YOLO saw
            # THIS frame, directly
            hazard_events = hazard_detector.check(detections, timestamp=now)

        if module_state["congestion"]:
            congestion_events = congestion_detector.check(timestamp=now)

        # ---- Piece 6: hand fired events to event_recorder ----
        all_events = (
            stationary_events + wrong_way_events + hazard_events + congestion_events
        )

        for event in all_events:
            # saves an annotated .jpg immediately, and starts collecting
            # the next 2 seconds of "after" frames in the background
            event_recorder.trigger_event(
                frame,
                timestamp=now,
                event_type=event.get("class_name", "event"),
                metadata=event,
            )

        # the always-running rolling tape — runs EVERY frame no matter
        # what, so the "2 seconds BEFORE" an event always already exists
        event_recorder.add_frame(frame, timestamp=now)

        # ---- Piece 7: draw + display ----
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, det["class_name"], (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

        cv2.putText(
            frame,
            build_status_text(module_state),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        display_frame = resize_frame_for_display(frame)
        cv2.imshow(window_name, display_frame)

        # waitKey(1) pauses 1ms AND tells us which key was pressed —
        # also what actually makes the window repaint on screen
        key = cv2.waitKey(1) & 0xFF
        if should_quit(key, window_name):
            break
        if key in KEY_TOGGLE_MAP:
            toggle_module_state(module_state, key)

    # ================= cleanup — runs ONCE, after break =================
    cap.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()