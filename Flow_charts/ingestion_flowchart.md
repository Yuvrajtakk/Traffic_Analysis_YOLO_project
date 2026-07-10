---
tags: [traffic-monitoring, ingestion, flowchart, video-capture, threading]
---

# `src/ingestion.py` — `VideoIngestion` Class Flowchart

## How to read this diagram

This isn't a single sequential flow — it's **two things happening at
once**, plus one function that gets called by *both*. The colors encode
that directly:

| Color | Meaning |
|---|---|
| 🔵 Blue | Runs on the **main thread** (`__init__`, `start()`, `read()`, `stop()`) |
| 🟢 Green | Runs on the **background thread** (`_update()`'s loop body) |
| 🟠 Orange | Inside `with self.lock:` — the critical section, mutually exclusive |
| 🟦 Teal | `_open_capture()` — shared code, called by *either* thread depending on context |
| 🔴 Red | Termination / error paths |
| 🟣 Purple | Constants imported from `config/thresholds.py` |

The **dashed arrows** are "uses/imports," not control flow. The
**thick arrows** are the two moments concurrency actually kicks in:
`start()` forking a new thread, and `stop()` blocking until that
thread has actually finished.

```mermaid
flowchart TD

%% ===================== LEGEND =====================
subgraph LEGEND["🎨 Legend"]
    direction LR
    L1["Main Thread"]:::mainThread
    L2["Background Thread"]:::bgThread
    L3["Locked / Critical Section"]:::lockZone
    L4["Shared fn — either thread"]:::sharedCode
    L5["Termination / Error"]:::errorPath
    L6["Constant from thresholds.py"]:::constant
end

%% ===================== GLOBAL CONSTANTS =====================
subgraph THRESH["⚙️ config/thresholds.py — Global Constants"]
    GC1["RECONNECT_MAX_RETRIES = 5"]
    GC2["RECONNECT_DELAY_SEC = 1 sec"]
end

%% ===================== INIT =====================
subgraph INIT["__init__ source, loop_file=True — MAIN THREAD"]
    I0(["VideoIngestion(source, loop_file) called"])
    I1["self.raw_source = source<br/>self.loop_file = loop_file<br/>self.source = source"]
    I2{"isinstance(source, str) AND<br/>starts with 'rtsp://' ?"}
    I3["self.is_rtsp = True"]
    I4["self.is_rtsp = False"]
    I5{"isinstance(source, str) AND<br/>NOT is_rtsp ?"}
    I6["self.is_file = True"]
    I7["self.is_file = False"]
    I8["self.cap = None<br/>self.frame = None<br/>self.lock = threading.Lock()<br/>self.running = False<br/>self.thread = None"]
    I9(["Object constructed —<br/>NOT yet connected to source"])

    I0 --> I1 --> I2
    I2 -- yes --> I3
    I2 -- no --> I4
    I3 --> I5
    I4 --> I5
    I5 -- yes --> I6
    I5 -- no --> I7
    I6 --> I8
    I7 --> I8
    I8 --> I9
end

%% ===================== OPEN CAPTURE (SHARED) =====================
subgraph OPEN["_open_capture — callable by MAIN thread (start) OR BACKGROUND thread (reconnect)"]
    O1{"self.is_rtsp ?"}
    O2["os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']<br/>= 'rtsp_transport;tcp'"]
    O3["cv2.VideoCapture(source, cv2.CAP_FFMPEG)"]
    O4["cv2.VideoCapture(source)"]
    O5{"cap.isOpened() ?"}
    O6["raise RuntimeError<br/>'Could not open source'"]
    O7(["return cap"])

    O1 -- yes --> O2 --> O3 --> O5
    O1 -- no --> O4 --> O5
    O5 -- no --> O6
    O5 -- yes --> O7
end

%% ===================== START (MAIN THREAD) =====================
subgraph START["start — MAIN THREAD"]
    S1(["start() called"])
    S2["self.cap = self._open_capture()"]
    S3["self.running = True"]
    S4["self.thread = Thread(target=self._update)<br/>self.thread.daemon = True"]
    S5["self.thread.start()"]
    S6(["return self"])

    S1 --> S2 --> S3 --> S4 --> S5 --> S6
end

%% ===================== BACKGROUND THREAD LOOP =====================
subgraph UPDATE["_update — FOREVER LOOP on BACKGROUND THREAD"]
    U1(["thread begins executing"])
    U2["retries = 0"]
    U3{"while self.running ?"}
    U4["ret, frame = self.cap.read()"]
    U5{"ret == True ?"}
    U6["retries = 0"]
    U7["🔒 ACQUIRE LOCK"]
    U8["self.frame = frame"]
    U9["🔓 RELEASE LOCK"]
    U10{"self.is_file AND<br/>NOT self.loop_file ?"}
    U11["self.running = False"]
    U12(["break → thread exits"])
    U13["retries += 1"]
    U14{"retries > RECONNECT_MAX_RETRIES ?"}
    U15["time.sleep(RECONNECT_DELAY_SEC)"]
    U16["self.cap.release()"]
    U17["self.cap = self._open_capture()"]

    U1 --> U2 --> U3
    U3 -- no --> U12
    U3 -- yes --> U4 --> U5
    U5 -- yes --> U6 --> U7 --> U8 --> U9 --> U3
    U5 -- no --> U10
    U10 -- yes --> U11 --> U12
    U10 -- no --> U13 --> U14
    U14 -- yes --> U11
    U14 -- no --> U15 --> U16 --> U17 --> U3
end

%% ===================== READ (MAIN THREAD / CONSUMER) =====================
subgraph READ["read — called anytime by main program — whichever thread calls it"]
    R1(["read() called"])
    R2["🔒 ACQUIRE LOCK"]
    R3{"self.frame is None ?"}
    R4["return None"]
    R5["return self.frame.copy()"]
    R6["🔓 RELEASE LOCK<br/>(on either return path)"]

    R1 --> R2 --> R3
    R3 -- yes --> R4 --> R6
    R3 -- no --> R5 --> R6
end

%% ===================== STOP (MAIN THREAD) =====================
subgraph STOP["stop — MAIN THREAD"]
    P1(["stop() called"])
    P2["self.running = False"]
    P3["self.thread.join()<br/>BLOCKS here until background thread exits"]
    P4["self.cap.release()"]
    P5(["fully shut down"])

    P1 --> P2 --> P3 --> P4 --> P5
end

%% ===================== CROSS-WIRING =====================
GC1 -.->|imported, used at| U14
GC2 -.->|imported, used at| U15

I9 -.->|.start() called next| S1
S2 -.->|calls| O1
S5 ==>|forks a NEW thread| U1
U17 -.->|re-opens via| O1
P3 ==>|must wait for| U12

%% ===================== STYLES =====================
classDef mainThread fill:#1e3a5f,stroke:#60a5fa,stroke-width:2px,color:#e5e7eb
classDef bgThread fill:#1e4620,stroke:#4ade80,stroke-width:2px,color:#e5e7eb
classDef lockZone fill:#5a3a1e,stroke:#fb923c,stroke-width:3px,color:#fff
classDef sharedCode fill:#0f3d3d,stroke:#2dd4bf,stroke-width:2px,color:#e5e7eb
classDef errorPath fill:#5a1e1e,stroke:#f87171,stroke-width:2px,color:#fff
classDef constant fill:#2d2d3a,stroke:#a78bfa,stroke-width:2px,color:#e5e7eb

class I0,I1,I2,I3,I4,I5,I6,I7,I8,I9 mainThread
class S1,S2,S3,S4,S5,S6 mainThread
class P1,P2,P3,P4,P5 mainThread
class R1 mainThread
class U1,U2,U3,U4,U5,U6,U10,U13,U14,U15,U16,U17 bgThread
class U7,U8,U9,R2,R3,R4,R5,R6 lockZone
class O1,O2,O3,O4,O5,O7 sharedCode
class O6,U11,U12 errorPath
class GC1,GC2 constant
```

## Things this diagram makes visually obvious that reading the code top-to-bottom doesn't

1. **`_open_capture()` sits in its own lane for a reason.** It's the
   *only* function in the whole file that both threads call — once
   from `start()` (main thread, at startup) and repeatedly from
   `_update()` (background thread, on every reconnect). Same code,
   different callers, different times. That's why it's teal, not blue
   or green.

2. **The lock zone is small on purpose.** Look at how little of the
   diagram is actually orange. `_update()` does the *expensive* work
   (`cap.read()`, at `U4`) **outside** the lock, and only grabs it for
   the cheap, instant assignment `self.frame = frame`. If the lock
   wrapped the whole loop instead, `read()` would block for the
   entire duration of a camera read — defeating the entire point of
   threading it in the first place.

3. **`U12` (thread exit) has three separate paths leading into it** —
   clean shutdown from `stop()` propagating through `while
   self.running`, the `loop_file=False` EOF fix, and retries exhausted.
   All three are different *reasons*, but they converge on the exact
   same exit point. Worth being able to name all three if asked "how
   does this thread ever stop?"

4. **`P3 ==> U12`** is the single most important edge in this whole
   diagram for explaining thread safety. `stop()` cannot proceed to
   `cap.release()` until `_update()` has physically reached `U12` —
   that's what `.join()` buys you, drawn as an explicit dependency
   rather than left implicit in prose.

---

*Drop this file directly into your Obsidian vault — Mermaid renders
natively via Obsidian's built-in Markdown engine, no community plugin
required.*
