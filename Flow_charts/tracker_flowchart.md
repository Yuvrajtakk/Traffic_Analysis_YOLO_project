---
tags: [traffic-monitoring, tracker, flowchart, yolo, bytetrack, track-manager]
---

# `src/tracker.py` — `YOLOTracker` + `TrackManager` Flowchart

## How this one differs from `ingestion.py`

No threads here. No lock. Everything in this file runs on whichever
thread calls it, one call at a time, in sequence — `YOLOTracker.track()`
gets called once per frame by your main loop, and its return value
(`detections`) is immediately handed to `TrackManager.update()`. There
is no background worker quietly running in parallel.

What *is* shared, though, is `self.tracks` inside `TrackManager` — one
dictionary that `update()` writes to and `get_active_ids()` /
`get_history()` read from, persisting **across every call**, for the
entire lifetime of the object. That persistence is the whole reason
this class exists — without it, every module downstream would have to
remember tracking history itself, and they'd inevitably disagree.

| Color | Meaning |
|---|---|
| Blue | `YOLOTracker` methods |
| Green | `TrackManager` methods |
| Orange | The part of a loop that runs **once per item**, not once per frame |
| Red | Guard clause / early return |
| Teal | `self.tracks` — the persistent shared dictionary itself |
| Purple | Constants imported from `config/thresholds.py` |
| Amber | Annotation — a "why," not a step |

```mermaid
flowchart TD

%% ===================== LEGEND =====================
subgraph LEGEND["Legend"]
    direction LR
    L1["YOLOTracker method"]:::yoloClass
    L2["TrackManager method"]:::tmClass
    L3["Per-item loop body"]:::loopBody
    L4["Guard clause / early return"]:::guardPath
    L5["self.tracks - shared dict"]:::sharedState
    L6["Constant from thresholds.py"]:::constant
    L7["Annotation"]:::note
end

%% ===================== GLOBAL CONSTANTS =====================
subgraph THRESH["config/thresholds.py - Global Constants"]
    GC1["TRACK_BUFFER_FRAMES = 30"]
    GC2["DEFAULT_FPS = 25"]
end

%% ===================== YOLOTracker.__init__ =====================
subgraph YOLO_INIT["YOLOTracker.__init__"]
    YI1(["YOLOTracker(weights_path, confidence_threshold, device) called"])
    YI2["self.model = YOLO(weights_path)"]
    YI3["self.confidence_threshold = confidence_threshold"]
    YI4["self.device = device"]
    YI5(["tracker object ready, model loaded in memory"])

    YI1 --> YI2 --> YI3 --> YI4 --> YI5
end

%% ===================== YOLOTracker.track =====================
subgraph YOLO_TRACK["YOLOTracker.track(frame) - called once per frame"]
    YT1(["track(frame) called"])
    YT2["results = self.model.track(frame, conf, device, tracker='bytetrack.yaml', persist=True)"]
    YT3["result = results[0]"]
    YT4["detections = empty list"]
    YT5{"result.boxes.id is None ?"}
    YT6(["return detections - empty, nothing confidently tracked yet"])
    YT7["i = 0"]
    YT8{"i < len(result.boxes.id) ?"}
    YT9["track_id = int(result.boxes.id[i])"]
    YT10["class_index = int(result.boxes.cls[i])"]
    YT11["class_name = self.model.names[class_index]"]
    YT12["confidence = float(result.boxes.conf[i])"]
    YT13["bbox_tensor = result.boxes.xyxy[i]"]
    YT14["bbox = tuple of plain floats from bbox_tensor"]
    YT15["detection_dict = id, class_name, confidence, bbox"]
    YT16["detections.append(detection_dict)"]
    YT17["i += 1"]
    YT18(["return detections"])

    YT1 --> YT2 --> YT3 --> YT4 --> YT5
    YT5 -- yes --> YT6
    YT5 -- no --> YT7 --> YT8
    YT8 -- yes --> YT9 --> YT10 --> YT11 --> YT12 --> YT13 --> YT14 --> YT15 --> YT16 --> YT17
    YT17 --> YT8
    YT8 -- no --> YT18
end

%% ===================== TrackManager.__init__ =====================
subgraph TM_INIT["TrackManager.__init__"]
    MI1(["TrackManager() called"])
    MI2["self.tracks = empty dict"]
    MI3(["manager ready, zero tracks stored"])

    MI1 --> MI2 --> MI3
end

%% ===================== TrackManager.update =====================
subgraph TM_UPDATE["TrackManager.update(detections, timestamp) - called once per frame"]
    MU1(["update(detections, timestamp) called"])
    MU2{"timestamp is None ?"}
    MU3["timestamp = time.time()"]
    MU4["use the timestamp passed in"]
    MU5["j = 0"]
    MU6{"j < len(detections) ?"}
    MU7["track_id = detections[j]['id']"]
    MU8{"track_id already in self.tracks ?"}
    MU9["create new entry: history=empty list, class_name, last_seen=None"]
    MU10["entry already exists, nothing to create"]
    MU11["self.tracks[track_id]['history'].append((timestamp, bbox))"]
    MU12["self.tracks[track_id]['last_seen'] = timestamp"]
    MU13["j += 1"]
    MU14["buffer_seconds = TRACK_BUFFER_FRAMES divided by DEFAULT_FPS"]
    MU15["to_purge = empty list"]
    MU16["ids_list = list(self.tracks.keys())"]
    MU17["k = 0"]
    MU18{"k < len(ids_list) ?"}
    MU19{"timestamp minus last_seen greater than buffer_seconds ?"}
    MU20["to_purge.append(track_id)"]
    MU21["k += 1"]
    MU22["p = 0"]
    MU23{"p < len(to_purge) ?"}
    MU24["del self.tracks[to_purge[p]]"]
    MU25["p += 1"]
    MU26(["update() finished for this frame"])
    MUNOTE["cannot delete dict keys while looping over the same dict - collect first in a plain list, delete in a second separate pass"]:::note

    MU1 --> MU2
    MU2 -- yes --> MU3
    MU2 -- no --> MU4
    MU3 --> MU5
    MU4 --> MU5
    MU5 --> MU6
    MU6 -- yes --> MU7 --> MU8
    MU8 -- no --> MU9
    MU8 -- yes --> MU10
    MU9 --> MU11
    MU10 --> MU11
    MU11 --> MU12 --> MU13
    MU13 --> MU6
    MU6 -- no --> MU14
    MU14 --> MU15 --> MU16 --> MU17 --> MU18
    MU18 -- yes --> MU19
    MU19 -- yes --> MU20
    MU19 -- no --> MU21
    MU20 --> MU21
    MU21 --> MU18
    MU18 -- no --> MU22 --> MU23
    MU23 -- yes --> MU24 --> MU25
    MU25 --> MU23
    MU23 -- no --> MU26
    MUNOTE -.-> MU16
    MUNOTE -.-> MU22
end

%% ===================== TrackManager.get_active_ids =====================
subgraph TM_ACTIVE["TrackManager.get_active_ids(timestamp)"]
    GA1(["get_active_ids(timestamp) called"])
    GA2{"timestamp is None ?"}
    GA3["timestamp = time.time()"]
    GA4["use the timestamp passed in"]
    GA5["one_frame_duration = 1 divided by DEFAULT_FPS"]
    GA6["active = empty list"]
    GA7["ids_list = list(self.tracks.keys())"]
    GA8["m = 0"]
    GA9{"m < len(ids_list) ?"}
    GA10{"timestamp minus last_seen less-or-equal one_frame_duration times 1.5 ?"}
    GA11["active.append(track_id)"]
    GA12["m += 1"]
    GA13(["return active"])

    GA1 --> GA2
    GA2 -- yes --> GA3
    GA2 -- no --> GA4
    GA3 --> GA5
    GA4 --> GA5
    GA5 --> GA6 --> GA7 --> GA8 --> GA9
    GA9 -- yes --> GA10
    GA10 -- yes --> GA11
    GA10 -- no --> GA12
    GA11 --> GA12
    GA12 --> GA9
    GA9 -- no --> GA13
end

%% ===================== TrackManager.get_history =====================
subgraph TM_HISTORY["TrackManager.get_history(track_id)"]
    GH1(["get_history(track_id) called"])
    GH2["entry = self.tracks.get(track_id, empty dict)"]
    GH3(["return entry.get('history', empty list)"])

    GH1 --> GH2 --> GH3
end

%% ===================== SHARED STATE ANCHOR =====================
TRACKS_DICT["self.tracks - lives for the lifetime of the TrackManager object"]:::sharedState

%% ===================== CROSS-WIRING =====================
GC1 -.->|used at| MU14
GC2 -.->|used at| MU14
GC2 -.->|used at| GA5

YT18 -.->|return value becomes the detections argument of| MU1

MU9 -.->|writes| TRACKS_DICT
MU11 -.->|writes| TRACKS_DICT
MU12 -.->|writes| TRACKS_DICT
MU24 -.->|writes deletes| TRACKS_DICT
GA7 -.->|reads| TRACKS_DICT
GH2 -.->|reads| TRACKS_DICT

%% ===================== STYLES =====================
classDef yoloClass fill:#1e3a5f,stroke:#60a5fa,stroke-width:2px,color:#e5e7eb
classDef tmClass fill:#1e4620,stroke:#4ade80,stroke-width:2px,color:#e5e7eb
classDef loopBody fill:#5a3a1e,stroke:#fb923c,stroke-width:3px,color:#fff
classDef guardPath fill:#5a1e1e,stroke:#f87171,stroke-width:2px,color:#fff
classDef sharedState fill:#0f3d3d,stroke:#2dd4bf,stroke-width:3px,color:#e5e7eb
classDef constant fill:#2d2d3a,stroke:#a78bfa,stroke-width:2px,color:#e5e7eb
classDef note fill:#4a3f0f,stroke:#facc15,stroke-width:1px,color:#fde68a

class YI1,YI2,YI3,YI4,YI5 yoloClass
class YT1,YT2,YT3,YT4,YT7,YT8,YT18 yoloClass
class YT9,YT10,YT11,YT12,YT13,YT14,YT15,YT16,YT17 loopBody
class YT5,YT6 guardPath

class MI1,MI2,MI3 tmClass
class MU1,MU2,MU3,MU4,MU5,MU6,MU13,MU14,MU15,MU16,MU17,MU18,MU21,MU22,MU23,MU25,MU26 tmClass
class MU7,MU8,MU9,MU10,MU11,MU12,MU19,MU20,MU24 loopBody

class GA1,GA2,GA3,GA4,GA5,GA6,GA7,GA8,GA9,GA12,GA13 tmClass
class GA10,GA11 loopBody
class GH1,GH3 tmClass
class GH2 sharedState

class GC1,GC2 constant
class MUNOTE note
class TRACKS_DICT sharedState
```

## Points worth being able to explain out loud

1. **Why is there no lock anywhere in this file, when `ingestion.py`
   was full of them?** Because `ingestion.py` genuinely has two
   threads racing over the same variable. `tracker.py` assumes it is
   only ever called from one place, sequentially — your main loop
   calls `track()`, then immediately calls `update()` with that
   result, then moves to the next frame. Nothing runs concurrently
   with `self.tracks` being mutated. If you ever changed that
   assumption (e.g. ran inference on a separate thread from the main
   display loop), `TrackManager` would need the exact same locking
   treatment `VideoIngestion` has. Good "what would break this"
   question to be ready for.

2. **The two-pass purge (`MU16`→`MU21` collect, then `MU22`→`MU25`
   delete) is a genuine Python constraint, not a style choice.**
   Mutating a dict's size while iterating over it with `.items()`
   raises `RuntimeError: dictionary changed size during iteration`.
   The annotation node (`MUNOTE`, amber) is there specifically because
   this is exactly the kind of thing an interviewer asks you to
   explain: "why not just delete inline as you find it?"

3. **`YT18 -.-> MU1`** is the one edge that shows how these two
   classes are meant to be wired together in `main.py`, even though
   neither class imports or calls the other directly. `YOLOTracker`
   doesn't know `TrackManager` exists, and vice versa — the contract
   between them is just "a list of dicts shaped like `{id, class_name,
   confidence, bbox}` flows from one to the other." That's a
   deliberate decoupling: you could swap in a different tracker
   entirely and `TrackManager` wouldn't need to change, as long as it
   still hands over dicts in that shape.

---

*No `<br/>` tags, no emoji glyphs anywhere in this file's node labels —
both caused silent text loss in your Excalidraw export of the previous
diagram. If this one renders clean end-to-end, that confirms the
theory; if something's still missing, tell me exactly which node so we
can narrow down the real cause.*
