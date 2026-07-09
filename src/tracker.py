"""
src/tracker.py

Wraps YOLOv8 + ByteTrack (via model.track()) and converts Ultralytics'
raw output into a clean, safe list of plain dictionaries. This is the
ONE place in the codebase that deals with results.boxes.id possibly
being None or shorter than results.boxes.xyxy — every module downstream
of this can assume clean, safe data.
"""

from ultralytics import YOLO


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
        # SAME detection at every step.
        for i in range(len(result.boxes.id)):

            # The persistent tracking ID ByteTrack assigned this object.
            # Comes back as a tensor value, so we convert to a plain int.
            track_id = int(result.boxes.id[i])

            # The predicted class as a NUMBER (e.g. 2), not yet a name.
            class_index = int(result.boxes.cls[i])

            # self.model.names is a dictionary built into the model that
            # maps {0: "person", 2: "car", ...} — this converts the raw
            # number into the actual readable class name.
            class_name = self.model.names[class_index]

            # How confident YOLO is in this specific detection, 0.0-1.0.
            confidence = float(result.boxes.conf[i])

            # The box coordinates as a tensor: [x1, y1, x2, y2]
            # (top-left corner, bottom-right corner, in pixels).
            bbox_tensor = result.boxes.xyxy[i]

            # Convert every coordinate in the tensor into a plain Python
            # float, and collect them into a normal tuple — so nothing
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