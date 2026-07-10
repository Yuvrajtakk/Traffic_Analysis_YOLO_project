---
tags: [traffic-monitoring, stationary, flowchart, analytics, guard-clauses]
---

# `src/analytics/stationary.py` — `StationaryDetector` Flowchart

## What's different about this one

`ingestion.py` was two threads racing over shared memory. `tracker.py`
was sequential logic building up a shared dictionary. This file is
neither — it's almost entirely **guard clauses**. Four separate "this
vehicle doesn't qualify, move to the next one" exits, and only ONE
narrow path that actually survives all four gates and fires an event.

That shape — mostly red skip-paths, one thin green success-path — is
the whole personality of this function, and it's worth being able to
see that at a glance, not just read it top to bottom.

The other thing worth seeing explicitly: this file **borrows** state
(`self.track_manager.tracks`, owned by `TrackManager`, read-only from
here) but also **owns** its own private state (`self.last_triggered`,
created and only ever touched by this class). Two different kinds of
"memory," two different colors.

| Color | Meaning |
|---|---|
| Blue | Normal control flow |
| Green | The success path — an event actually gets built and fired |
| Teal | `self.track_manager.tracks` — borrowed, read-only here |
| Magenta | `self.last_triggered` — owned by this class, read AND written here |
| Purple | Constant from `config/thresholds.py` |
| Amber | Annotation — a design decision, not a step |

```mermaid
flowchart TD

%% ===================== LEGEND =====================
subgraph LEGEND["Legend"]
    direction LR
    LG1["Normal control flow"]:::mainFlow
    LG2["Success path - event fires"]:::success
    LG3["Borrowed state - read only"]:::borrowed
    LG4["Owned state - read and write"]:::local
    LG5["Constant from thresholds.py"]:::constant
    LG6["Annotation"]:::note
end

%% ===================== GLOBAL CONSTANTS =====================
subgraph THRESH["config/thresholds.py - Global Constants"]
    GC1["STATIONARY_DURATION_SEC"]
    GC2["STATIONARY_PIXEL_THRESHOLD"]
    GC3["VEHICLE_CLASSES"]
    GC4["EVENT_COOLDOWN_SEC"]
end

%% ===================== __init__ =====================
subgraph INIT["StationaryDetector.__init__"]
    I1(["StationaryDetector(track_manager) called"])
    I2["self.track_manager = track_manager"]
    I3["self.last_triggered = empty dict"]
    I4(["detector ready"])

    I1 --> I2 --> I3 --> I4
end

%% ===================== _get_centroid =====================
subgraph CENTROID["_get_centroid(bbox) - helper"]
    GCD1(["_get_centroid(bbox) called"])
    GCD2["x1, y1, x2, y2 = bbox"]
    GCD3["cx = (x1 + x2) divided by 2"]
    GCD4["cy = (y1 + y2) divided by 2"]
    GCD5(["return (cx, cy)"])

    GCD1 --> GCD2 --> GCD3 --> GCD4 --> GCD5
end

%% ===================== _distance =====================
subgraph DIST["_distance(p1, p2) - helper"]
    D1(["_distance(p1, p2) called"])
    D2["x1, y1 = p1"]
    D3["x2, y2 = p2"]
    D4(["return sqrt of dx squared plus dy squared"])

    D1 --> D2 --> D3 --> D4
end

%% ===================== check =====================
subgraph CHECK["check(timestamp) - called once per frame"]
    C1(["check(timestamp) called"])
    C2{"timestamp is None ?"}
    C3["timestamp = time.time()"]
    C4["use timestamp passed in"]
    C5["events = empty list"]
    C6["ids_list = list of self.track_manager.tracks.keys()"]
    C7["n = 0"]
    C8{"n < len(ids_list) ?"}
    C9["track_id = ids_list[n]"]
    C10["info = self.track_manager.tracks[track_id]"]
    C11{"info class_name in VEHICLE_CLASSES ?"}
    C12["history = info history list"]
    C13["window = entries from history within STATIONARY_DURATION_SEC of now"]
    C14{"len(window) less than 2 ?"}
    C15["oldest_timestamp = window[0] timestamp"]
    C16{"now minus oldest_timestamp less than STATIONARY_DURATION_SEC times 0.9 ?"}
    C17["centroids = get_centroid of every bbox in window"]
    C18["max_distance = biggest distance from centroids[0] to any centroid"]
    C19{"max_distance less-or-equal STATIONARY_PIXEL_THRESHOLD ?"}
    C20["last_time = self.last_triggered.get(track_id, 0)"]
    C21{"now minus last_time less than EVENT_COOLDOWN_SEC ?"}
    C22["event = build dict with id, class_name, bbox, timestamp"]
    C23["events.append(event)"]
    C24["self.last_triggered[track_id] = timestamp"]
    C25["n += 1 - advance to next ID"]
    C26(["return events"])
    CNOTE["design decision: loop over tracks.keys(), NOT get_active_ids(). get_active_ids() only returns IDs seen in the last ~1.5 frames - a vehicle briefly occluded by a passing truck would be skipped that frame, silently gapping its stationary streak even though it never moved"]:::note

    C1 --> C2
    C2 -- yes --> C3
    C2 -- no --> C4
    C3 --> C5
    C4 --> C5
    C5 --> C6 --> C7 --> C8
    C8 -- yes --> C9 --> C10 --> C11
    C11 -- no, not a vehicle --> C25
    C11 -- yes --> C12 --> C13 --> C14
    C14 -- yes, too few points --> C25
    C14 -- no --> C15 --> C16
    C16 -- yes, not enough history yet --> C25
    C16 -- no --> C17 --> C18 --> C19
    C19 -- no, moved too much --> C25
    C19 -- yes --> C20 --> C21
    C21 -- yes, still on cooldown --> C25
    C21 -- no --> C22 --> C23 --> C24 --> C25
    C25 --> C8
    C8 -- no --> C26
    CNOTE -.-> C6
end

%% ===================== SHARED / OWNED STATE ANCHORS =====================
BORROWED_TRACKS["self.track_manager.tracks - OWNED by TrackManager, read only here"]:::borrowed
LOCAL_TRIGGERED["self.last_triggered - OWNED by this StationaryDetector instance"]:::local

%% ===================== CROSS-WIRING =====================
GC1 -.->|used at| C13
GC1 -.->|used at| C16
GC2 -.->|used at| C19
GC3 -.->|used at| C11
GC4 -.->|used at| C21

I2 -.->|points at, never copies| BORROWED_TRACKS
I3 -.->|creates| LOCAL_TRIGGERED

BORROWED_TRACKS -.->|read by| C6
BORROWED_TRACKS -.->|read by| C10
LOCAL_TRIGGERED -.->|read by| C20
C24 -.->|writes to| LOCAL_TRIGGERED

C17 -.->|calls| GCD1
C18 -.->|calls| D1

%% ===================== STYLES =====================
classDef mainFlow fill:#1e3a5f,stroke:#60a5fa,stroke-width:2px,color:#e5e7eb
classDef success fill:#1e4620,stroke:#4ade80,stroke-width:3px,color:#e5e7eb
classDef borrowed fill:#0f3d3d,stroke:#2dd4bf,stroke-width:3px,color:#e5e7eb
classDef local fill:#4a1e3a,stroke:#f472b6,stroke-width:3px,color:#e5e7eb
classDef constant fill:#2d2d3a,stroke:#a78bfa,stroke-width:2px,color:#e5e7eb
classDef note fill:#4a3f0f,stroke:#facc15,stroke-width:1px,color:#fde68a

class I1,I2,I3,I4 mainFlow
class GCD1,GCD2,GCD3,GCD4,GCD5 mainFlow
class D1,D2,D3,D4 mainFlow
class C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C25,C26 mainFlow
class C22,C23,C24 success
class GC1,GC2,GC3,GC4 constant
class CNOTE note
class BORROWED_TRACKS borrowed
class LOCAL_TRIGGERED local
```

## Points worth being able to answer out loud

1. **Count the exits into `C25`.** Four separate reasons converge on
   the exact same "move to next ID" node: not a vehicle, not enough
   history points, history not old enough yet, moved too far. That's
   the visual proof of "mostly guard clauses" — four different doors,
   all leading to the same hallway.

2. **`BORROWED_TRACKS` has two arrows going *into* `C6` and `C10`, and
   zero arrows coming *out* of this file back into it.** This module
   never mutates `self.track_manager.tracks` — only `TrackManager.update()`
   does that. If you ever caught yourself writing
   `self.track_manager.tracks[track_id] = something` inside
   `stationary.py`, that would be a real design violation worth
   catching — it would mean this module secretly started owning state
   it's only supposed to borrow.

3. **The design-decision note (`CNOTE`) attaches to `C6`, not to a
   later node**, because the decision is made the moment you choose
   *which* iterable to loop over — everything downstream just follows
   from that one choice. If Ankit sir asks "why not just filter for
   active tracks first," this is the node to point at.

4. **Notice what `C13`'s window filter does NOT do**: it doesn't check
   whether the history is *continuous* — just whether each entry's
   timestamp falls within the last `STATIONARY_DURATION_SEC`. If a
   vehicle was seen at `t=1s`, occluded, then reappeared at `t=4.8s`,
   both points land in the same window, and the max-distance check
   would treat them as adjacent even though 3.8 seconds of unknown
   movement happened in between. Is that a bug, or is it fine given
   what "stationary" is trying to measure? Worth having an answer
   ready — I'd genuinely like to hear your take before I give mine.

---

*Once you've opened this in Obsidian: does `CNOTE` (the design-decision
annotation) render and connect visibly to `C6`? That's the node most
likely to get visually lost given how many arrows converge on this
subgraph — good thing to double check.*
