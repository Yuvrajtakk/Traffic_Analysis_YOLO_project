"""
src/tracker.py

Wraps YOLOv8 + ByteTrack (via model.track()) and converts Ultralytics'
raw output into a clean, safe list of plain dictionaries. This is the
ONE place in the codebase that deals with results.boxes.id possibly
being None or shorter than results.boxes.xyxy — every module downstream
of this can assume clean, safe data.

Also contains TrackManager: shared per-ID history used by every
analytics module (stationary, wrong-way, hazards, congestion), so all
four agree on whether an ID is active, temporarily lost, or gone.
"""

import time

from ultralytics import YOLO

from config.thresholds import TRACK_BUFFER_FRAMES, DEFAULT_FPS


class YOLOTracker:
    def __init__(self, weights_path, confidence_threshold=0.4, device="cpu"):
        """
        weights_path: path to a .pt file (e.g. "models/weights/yolov8s_coco_stock.pt")
        confidence_threshold: ignore any detection below this confidence
        device: "cpu" or "0" (GPU index)
        """
        # Loads the model's weights from disk into memory, ready to run.
        self.model = YOLO(weights_path)

        self.confidence_threshold = confidence_threshold
        self.device = device

    def track(self, frame):
        """
        Runs detection + tracking on a single frame.
        Returns a list of dicts, one per confidently-tracked detection:
            {"id": int, "class_name": str, "confidence": float, "bbox": (x1,y1,x2,y2)}
        Detections with no assigned ID yet are SKIPPED (not included) —
        we only care about objects ByteTrack is confident enough to track.
        """
        # Runs YOLO detection + ByteTrack tracking together in one call.
        # persist=True is REQUIRED — it tells ByteTrack "remember previous
        # frames' tracks," otherwise every frame starts fresh with no memory,
        # and every object would get a brand new ID every single frame.
        results = self.model.track(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,   # suppresses Ultralytics' own console spam
        )

        # results is actually a LIST (Ultralytics supports batches of images),
        # but we only ever pass one frame at a time, so we always want
        # the first (and only) result in that list.
        result = results[0]

        # This is what we'll hand back — our clean, safe list. Starts empty.
        detections = []

        # GUARD CLAUSE: if result.boxes.id is None, it means ByteTrack has
        # zero confidently-tracked objects this frame (could be an empty
        # road, or every object is still too new/unconfirmed to have an ID
        # yet). Calling len() on None would crash, so we check FIRST and
        # bail out early with an empty list — nothing to process.
        if result.boxes.id is None:
            return detections

        # We now know result.boxes.id safely has a length. Loop by INDEX
        # (not by directly iterating one of the lists) so that xyxy[i],
        # id[i], cls[i], and conf[i] all stay correctly matched to the
        # SAME detection at every step — these are four parallel lists
        # that only make sense read together, position by position.
        for i in range(len(result.boxes.id)):

            # The persistent tracking ID ByteTrack assigned this object.
            # Comes back as a tensor value, so we convert to a plain int.
            track_id = int(result.boxes.id[i])

            # The predicted class as a NUMBER (e.g. 2), not yet a name.
            class_index = int(result.boxes.cls[i])

            # self.model.names is a dictionary built into the model that
            # maps {0: "person", 2: "car", ...} — converts the raw
            # number into the actual readable class name.
            class_name = self.model.names[class_index]

            # How confident YOLO is in this specific detection, 0.0-1.0.
            confidence = float(result.boxes.conf[i])

            # The box coordinates as a tensor: [x1, y1, x2, y2]
            # (top-left corner, bottom-right corner, in pixels).
            bbox_tensor = result.boxes.xyxy[i]

            # Convert every coordinate in the tensor into a plain Python
            # float, collected into a normal tuple — so nothing
            # downstream ever has to deal with tensor objects directly.
            bbox = tuple(float(coord) for coord in bbox_tensor)

            detection_dict = {
                "id": track_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": bbox,
            }
            detections.append(detection_dict)

        return detections


class TrackManager:
    """
    Shared memory across all analytics modules. Keeps a per-ID history
    of (timestamp, bbox) pairs so modules can answer questions like
    "has this ID been roughly still for 4 seconds?" without each module
    keeping its own separate, possibly-disagreeing copy of tracking state.
    """

    def __init__(self):
        # The main data store. Key = track ID (int).
        # Value = a dict holding that ID's history and bookkeeping info.
        # Starts completely empty — nothing has been tracked yet.
        #
        # Shape of one entry, for reference:
        # self.tracks[7] = {
        #     "history": [(t1, bbox1), (t2, bbox2), ...],
        #     "class_name": "car",
        #     "last_seen": t2,
        # }
        self.tracks = {}

    def update(self, detections, timestamp=None):
        """
        Called ONCE PER FRAME with the fresh list of detections from
        YOLOTracker.track(). Updates history for everything seen this
        frame, and purges anything that's been missing too long.
        """
        # If the caller didn't supply a timestamp, just use "right now."
        # Using real wall-clock time (not frame count) matches the
        # decision log's timestamp-based design — stays accurate even
        # if actual FPS drifts from any assumed value (RTSP is unstable).
        if timestamp is None:
            timestamp = time.time()

        # ── Step 1: record everything seen THIS frame ──────────────
        for det in detections:
            track_id = det["id"]

            # Have we ever seen this ID before? If not, open a brand
            # new "folder" for it with an empty history list.
            if track_id not in self.tracks:
                self.tracks[track_id] = {
                    "history": [],
                    "class_name": det["class_name"],
                    "last_seen": None,  # gets set for real just below
                }

            # Whether this ID is brand new or already existed, always
            # add one new (timestamp, bbox) entry — this is what builds
            # up the position-over-time timeline other modules will read.
            self.tracks[track_id]["history"].append((timestamp, det["bbox"]))

            # Stamp "the last time we actually saw this ID" with now.
            self.tracks[track_id]["last_seen"] = timestamp

        # ── Step 2: purge anything missing too long ─────────────────
        # Convert the frame-count buffer into an approximate seconds
        # value, since we're operating in real time, not frame counts.
        buffer_seconds = TRACK_BUFFER_FRAMES / DEFAULT_FPS

        # We CANNOT delete dictionary keys while looping over the same
        # dictionary with .items() — Python raises a RuntimeError if the
        # dict's size changes mid-iteration. So: first just COLLECT the
        # IDs we want to delete into a plain list, leaving self.tracks
        # untouched during this loop...
        to_purge = []
        for track_id, info in self.tracks.items():
            if (timestamp - info["last_seen"]) > buffer_seconds:
                to_purge.append(track_id)

        # ...THEN, in a completely separate loop (dictionary iteration
        # is fully finished by now), it's safe to actually delete them.
        for track_id in to_purge:
            del self.tracks[track_id]

    def get_active_ids(self, timestamp=None):
        """
        Returns a list of track IDs currently ACTIVE — meaning seen in
        roughly the last 1 frame (not just "somewhere in the buffer").
        Useful for modules that only care about what's visible RIGHT NOW,
        like congestion counting (you don't want to count a car that's
        been occluded for the last second as "currently in the ROI").
        """
        if timestamp is None:
            timestamp = time.time()

        # One frame's worth of time, in seconds. We allow a little
        # slack (1.5x) so tiny natural timing jitter doesn't wrongly
        # exclude something that was genuinely seen this frame.
        one_frame_duration = 1.0 / DEFAULT_FPS
        active = []

        for track_id, info in self.tracks.items():
            if (timestamp - info["last_seen"]) <= one_frame_duration * 1.5:
                active.append(track_id)

        return active

    def get_history(self, track_id):
        """
        Returns the full (timestamp, bbox) history list for one ID,
        or an empty list if that ID doesn't exist (e.g. already purged,
        or never existed at all).

        Returning [] instead of None matters: calling code can safely
        do `for t, bbox in get_history(id):` without ever needing an
        extra "if history is not None" check first — looping over an
        empty list just runs zero times, no crash either way.
        """
        # .get(track_id, {}) : if track_id isn't a key, hand back an
        # empty dict {} instead of crashing with a KeyError.
        # .get("history", []) : then look up "history" inside whatever
        # we got — real data, or [] if we're working with that empty
        # fallback dict.
        return self.tracks.get(track_id, {}).get("history", [])