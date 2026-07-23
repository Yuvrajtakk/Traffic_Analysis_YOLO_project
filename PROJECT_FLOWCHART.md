# Traffic Analysis YOLO Project — Architecture & Flowcharts

This document breaks down the system architecture of the real-time YOLOv8 traffic monitoring pipeline. The architecture is presented **piece-by-piece**, building up from individual components to the complete system.

> **How to read these diagrams on GitHub:** GitHub renders Mermaid diagrams with built-in zoom & pan controls (a small widget appears in the corner). For the smaller diagrams below, you likely won't need them — but they're there for the larger ones.

---

## 1. Data Ingestion & Tracking Pipeline

The system starts by ingesting video frames in a **background thread** to prevent blocking. The main loop grabs the latest frame, runs YOLOv8 + ByteTrack to detect and track objects, and stores the results in a shared `TrackManager`.

```mermaid
flowchart TD
    subgraph EXT["📹 Video Sources"]
    direction LR
        X_FILE[/"Local test video"/]
        X_RTSP[/"Live RTSP camera feed"/]
    end

    subgraph ING["🔄 Video Ingestion — Background Thread"]
    direction TB
        I_INIT["VideoIngestion starts"]
        I_UPD{{"_update loop runs forever"}}
        I_RETQ{"Read frame from source"}
        I_LOCKW["Store latest frame in shared buffer"]
    end

    subgraph MAIN["🧠 Main Processing Loop"]
    direction TB
        M_READ["Grab newest frame — never blocks"]
        M_TRACK["Run YOLOv8s + ByteTrack inference"]
        M_TMUPD["Update TrackManager with detections"]
    end

    subgraph TRK["📋 Shared Track State"]
    direction TB
        TM_APP["Append timestamp + bbox to each vehicle's history"]
        TM_PURGE["Purge stale IDs that haven't been seen recently"]
    end

    X_FILE --> I_INIT
    X_RTSP --> I_INIT
    I_INIT ==> I_UPD
    I_UPD --> I_RETQ
    I_RETQ -->|"success"| I_LOCKW --> I_UPD

    I_LOCKW -.->|"shared buffer"| M_READ
    M_READ --> M_TRACK --> M_TMUPD
    M_TMUPD ==> TM_APP --> TM_PURGE

    classDef srcC fill:#E3F2FD,stroke:#1565C0,color:#0D47A1
    classDef ingC fill:#DBEAFE,stroke:#2563EB,color:#1E3A8A
    classDef mainC fill:#FFF8E1,stroke:#F59E0B,color:#7C2D12
    classDef trkC fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20

    class X_FILE,X_RTSP srcC
    class I_INIT,I_UPD,I_RETQ,I_LOCKW ingC
    class M_READ,M_TRACK,M_TMUPD mainC
    class TM_APP,TM_PURGE trkC
```

**What's happening here:**
- A **daemon thread** continuously reads frames from the video source and stores the latest one in a thread-safe buffer.
- The **main loop** grabs a copy of the latest frame (never waits), runs YOLO detection + ByteTrack tracking, and feeds the results into the `TrackManager`.
- The `TrackManager` keeps a history of every tracked vehicle's position over time, and cleans up old IDs that disappeared.

---

## 2. Analytics Modules

Once the `TrackManager` is updated, the data is passed to **four independent analytics modules**. Each module watches for a different traffic condition. Three modules read from the `TrackManager`'s vehicle history; the **Hazard** module is different — it reads raw YOLO detections directly because fire/smoke/accident have no stable identity to track.

```mermaid
flowchart TD
    subgraph TRK["📋 Shared Track State"]
    direction LR
        TM_HIST["Vehicle History Buffer"]
    end

    subgraph ANA["🔍 Analytics Modules — each can be toggled ON/OFF via keyboard"]
    direction TB
        M_GATE_S{"Stationary ON?"}
        S_CHECK["🅿️ StationaryDetector<br>Has a vehicle stayed parked for 4+ seconds?"]

        M_GATE_W{"Wrong-Way ON?"}
        W_CHECK["🔄 WrongWayDetector<br>Is a vehicle moving against the expected flow direction?"]

        M_GATE_H{"Hazards ON?"}
        H_CHECK["🔥 HazardDetector<br>Is Fire, Smoke, or Accident confidently detected?"]

        M_GATE_C{"Congestion ON?"}
        C_CHECK["🚗 CongestionDetector<br>Are there more vehicles in the ROI than the capacity?"]
    end

    RAW["Raw YOLO Detections<br>— no tracking needed"] -.->|"direct feed"| H_CHECK

    TM_HIST -.->|"read history"| S_CHECK
    TM_HIST -.->|"read history"| W_CHECK
    TM_HIST -.->|"read active IDs"| C_CHECK

    M_GATE_S -->|"Yes"| S_CHECK
    M_GATE_W -->|"Yes"| W_CHECK
    M_GATE_H -->|"Yes"| H_CHECK
    M_GATE_C -->|"Yes"| C_CHECK

    S_CHECK -->|"🔔 FIRE EVENT"| MERGE["Merge All Fired Events"]
    W_CHECK -->|"🔔 FIRE EVENT"| MERGE
    H_CHECK -->|"🔔 FIRE EVENT"| MERGE
    C_CHECK -->|"🔔 FIRE EVENT"| MERGE

    classDef trkC fill:#E8F5E9,stroke:#2E7D32,color:#1B5E20
    classDef staC fill:#FFEDD5,stroke:#EA580C,color:#7C2D12
    classDef wwC fill:#FEE2E2,stroke:#DC2626,color:#7F1D1D
    classDef hazC fill:#FEF9C3,stroke:#CA8A04,color:#713F12
    classDef conC fill:#DCFCE7,stroke:#16A34A,color:#14532D
    classDef rawC fill:#F3E5F5,stroke:#7B1FA2,color:#4A148C
    classDef mergeC fill:#E0F7FA,stroke:#00838F,color:#004D40

    class TM_HIST trkC
    class M_GATE_S,S_CHECK staC
    class M_GATE_W,W_CHECK wwC
    class M_GATE_H,H_CHECK hazC
    class M_GATE_C,C_CHECK conC
    class RAW rawC
    class MERGE mergeC
```

**What's happening here:**
- Each module can be **toggled ON or OFF** at runtime using keyboard keys (`s`, `w`, `h`, `c`). When a module is OFF, its `check()` is never called — saving compute.
- **Stationary** checks if a vehicle's centroid has barely moved over a time window.
- **Wrong-Way** computes the cosine similarity between a vehicle's trajectory and the expected flow direction — negative = going against traffic.
- **Hazard** uses a flicker-tolerant persistence check: it requires 3 confident detections out of the last 5 frames before firing, so brief false positives don't trigger events.
- **Congestion** simply counts vehicles inside a defined polygon and compares against a capacity threshold.

---

## 3. Event Recorder & Output

When any analytics module fires an event, the system immediately records it. A **rolling pre-event buffer** ensures we always have the 2 seconds of video *before* the event occurred, and a background thread writes the resulting `.mp4` clip and `.jpg` still to disk without blocking the main loop.

```mermaid
flowchart TD
    subgraph MAIN["🧠 Main Loop"]
    direction TB
        M_ANYEV{"Any module fired an event?"}
        M_TRIGGER["Call event_recorder.trigger_event()"]
        M_ADDF["Call event_recorder.add_frame()<br>— runs EVERY frame, not just events"]
    end

    subgraph REC["💾 Event Recorder"]
    direction TB
        E_APP["Append frame to rolling ring buffer"]
        E_TRIM["Trim frames older than 2 seconds"]

        E_JPG["📸 Save annotated JPG immediately"]
        E_DISP["Collect 2 seconds of 'after' frames"]
        E_WRITE["🎬 Background Thread: stitch pre+after into MP4"]
    end

    subgraph EXT["📂 Disk Outputs"]
    direction LR
        X_OUT[/"outputs/events/<br>JPG stills + MP4 clips"/]
    end

    M_ANYEV -->|"Yes"| M_TRIGGER
    M_ANYEV -->|"No — still feed the buffer"| M_ADDF
    M_TRIGGER --> M_ADDF

    M_ADDF ==> E_APP --> E_TRIM
    M_TRIGGER ==> E_JPG
    M_TRIGGER ==> E_DISP

    E_TRIM -.->|"pre-event frames ready"| E_DISP
    E_DISP ==>|"daemon thread"| E_WRITE

    E_JPG --> X_OUT
    E_WRITE --> X_OUT

    classDef mainC fill:#FFF8E1,stroke:#F59E0B,color:#7C2D12
    classDef recC fill:#FCE7F3,stroke:#DB2777,color:#831843
    classDef outC fill:#ECEFF1,stroke:#607D8B,color:#37474F,stroke-dasharray:4 3

    class M_ANYEV,M_TRIGGER,M_ADDF mainC
    class E_APP,E_TRIM,E_JPG,E_DISP,E_WRITE recC
    class X_OUT outC
```

**What's happening here:**
- `add_frame()` runs on **every single frame**, feeding a rolling 2-second ring buffer. This way, when an event fires, the "before" footage already exists.
- `trigger_event()` saves the annotated JPG **immediately**, then starts collecting 2 seconds of "after" frames.
- Once the after-window is full, a **daemon thread** stitches pre + after frames into an MP4 clip and writes it to disk — the main loop never blocks on I/O.

---

## 4. Complete System Overview

This diagram shows **the entire system in one view** — from startup to the real-time processing loop. It is a simplified, readable version that focuses on the *flow of data* rather than individual function signatures.

<!--
HOW TO READ THIS DIAGRAM:
- Solid arrows (→) = control flow / data flow
- Thick arrows (⇒) = object construction or thread spawn
- Dotted arrows (⇢) = config/data reads
- Colors match the subsystem: amber=main, blue=ingestion, green=tracking, 
  orange/red/yellow/green=analytics, pink=recorder, purple=config, cyan=tuning
-->

```mermaid
flowchart TD

%% ═══════ STARTUP ═══════
subgraph STARTUP["🚀 STARTUP SEQUENCE"]
direction TB
    S1(["python main.py"])
    S2["Auto-detect: probe RTSP stream"]
    S3{"Live stream reachable?"}
    S4["Use RTSP live feed"]
    S5["Fallback to sample.mp4"]
    S6["Create shared LiveConfig"]
    S7["Start VideoIngestion bg thread"]
    S8["Load YOLOv8s model onto GPU/CPU"]
    S9["Create shared TrackManager"]
    S10["Wait for first real frame"]
    S11{"Calibrate zones?"}
    S12["Run calibration UI on first frame"]
    S13["Skip — use existing zones"]
    S14{"Start monitoring?"}
    S15["Build 4 analytics detectors"]
    S16(["Exit"])
end

S1 --> S2 --> S3
S3 -->|"Yes"| S4 --> S6
S3 -->|"No"| S5 --> S6
S6 --> S7 --> S8 --> S9 --> S10
S10 --> S11
S11 -->|"n"| S12 --> S14
S11 -->|"y"| S13 --> S14
S14 -->|"n"| S16
S14 -->|"y"| S15

%% ═══════ FRAME LOOP ═══════
subgraph LOOP["🔁 REAL-TIME FRAME LOOP — runs every frame"]
direction TB
    L1["Grab latest frame from ingestion thread"]
    L2{"Frame available?"}
    L3["Run YOLOv8 + ByteTrack detection"]
    L4["Update TrackManager with new detections"]
    L5["Run toggled-ON analytics modules"]
    L6{"Any events fired?"}
    L7["Save JPG + start collecting clip"]
    L8["Feed frame into rolling 2s pre-buffer"]
    L9["Draw boxes, arrows, status overlay"]
    L10["Handle keyboard input"]
end

S15 --> L1
L1 --> L2
L2 -->|"No — skip"| L1
L2 -->|"Yes"| L3 --> L4 --> L5 --> L6
L6 -->|"Yes"| L7 --> L8
L6 -->|"No"| L8
L8 --> L9 --> L10 --> L1

%% ═══════ SIDE SYSTEMS ═══════
subgraph SIDE["⚙️ SUPPORTING SYSTEMS"]
direction TB
    CFG["📋 Config Layer<br>thresholds.py — frozen defaults<br>LiveConfig — mutable, read fresh each frame"]
    TUN["🎛️ Tuning Panel<br>Trackbar sliders write into LiveConfig<br>Changes take effect next frame"]
    REC["💾 Event Recorder<br>Rolling 2s buffer + background MP4 writer"]
    OUT[/"📂 outputs/events/<br>JPG + MP4 per event"/]
end

%% ═══════ KEYBOARD ═══════
subgraph KEYS["⌨️ KEYBOARD CONTROLS"]
direction LR
    K1["d = toggle dashboard"]
    K2["t = toggle tuning panel"]
    K3["s w h c = toggle modules"]
    K4["p = print config snapshot"]
    K5["q = quit"]
end

%% ═══════ CONNECTIONS ═══════
CFG -.->|"read fresh"| L3
CFG -.->|"read fresh"| L5
TUN ==>|"writes into"| CFG
L7 ==> REC
L8 ==> REC
REC --> OUT
KEYS --> L10

%% ═══════ STYLES ═══════
classDef startC fill:#E8EAF6,stroke:#3949AB,color:#1A237E
classDef loopC fill:#FFF8E1,stroke:#F59E0B,stroke-width:2px,color:#7C2D12
classDef sideC fill:#F3E5F5,stroke:#7B1FA2,color:#4A148C
classDef keyC fill:#E0F2F1,stroke:#00897B,color:#004D40
classDef outC fill:#ECEFF1,stroke:#607D8B,color:#37474F,stroke-dasharray:4 3

class S1,S2,S3,S4,S5,S6,S7,S8,S9,S10,S11,S12,S13,S14,S15,S16 startC
class L1,L2,L3,L4,L5,L6,L7,L8,L9,L10 loopC
class CFG,TUN,REC sideC
class K1,K2,K3,K4,K5 keyC
class OUT outC
```

**How to read this diagram:**
- **🚀 Startup Sequence (top):** The program auto-detects whether a live RTSP stream is available. If not, it falls back to a test video file. It then loads the YOLO model, waits for the first frame, optionally runs the zone calibration tool, and builds all analytics modules.
- **🔁 Frame Loop (middle):** Every frame goes through: grab → detect → track → analyze → record → draw → handle keys → repeat.
- **⚙️ Supporting Systems (right):** The config layer provides thresholds that are read fresh every frame. The tuning panel writes directly into `LiveConfig` so changes are instant. The event recorder runs its own background thread for disk I/O.
- **⌨️ Keyboard Controls (bottom):** All module toggles and display controls are handled via single-key presses.

---

## 5. One-Frame Lifecycle (Runtime Sequence)

This sequence diagram shows the **exact order of operations** within a single frame — which component calls which, and what data flows between them.

```mermaid
sequenceDiagram
    autonumber
    participant BG as 🔄 Ingestion Thread
    participant M as 🧠 Main Loop
    participant Y as 🎯 YOLOTracker
    participant TM as 📋 TrackManager
    participant AN as 🔍 Analytics (x4)
    participant LC as ⚙️ LiveConfig
    participant R as 💾 EventRecorder
    participant WT as 🎬 Clip Writer Thread

    Note over BG: Runs forever in background
    BG->>BG: cap.read() → overwrite shared frame buffer
    M->>BG: Grab newest frame copy (never blocks)
    M->>M: Stamp current time as shared timestamp
    M->>Y: track(frame)
    Y->>LC: Read YOLO confidence threshold (fresh)
    Y->>Y: model.track — YOLOv8s + ByteTrack
    Y-->>M: Return list of detections (id, class, conf, bbox)
    M->>TM: update(detections, timestamp)
    TM->>TM: Append history + purge stale IDs
    M->>AN: check() for each toggled-ON module
    AN->>TM: Read vehicle history / active IDs
    AN->>LC: Read tunables (fresh every call)
    AN-->>M: Return event lists (tagged by module)
    M->>R: trigger_event() for each fired event → save JPG
    M->>R: add_frame() — always, feeds 2s rolling buffer
    R->>WT: When 2s of after-frames collected → write MP4
    WT-->>R: MP4 saved to outputs/events/
    M->>M: Draw boxes, arrows, status text → display
    M->>M: Handle keyboard input → loop back
```

---

## Color Legend

| Color | Subsystem |
|---|---|
| 🟦 Blue | `src/ingestion.py` — VideoIngestion threaded capture |
| 🟨 Amber | `main.py` — entry point, startup, real-time loop |
| 🟩 Green | `src/tracker.py` — YOLOTracker + TrackManager |
| 🟧 Orange | `src/analytics/stationary.py` |
| 🟥 Red | `src/analytics/wrong_way.py` |
| 🟡 Yellow | `src/analytics/hazards.py` |
| 💚 Green | `src/analytics/congestion.py` |
| 🩷 Pink | `src/event_recorder.py` |
| 🟪 Purple | `config/` — thresholds.py + LiveConfig |
| 🩵 Cyan | `src/tuning_panel.py` |

**Edge styles:** thick `==>` = object construction or thread spawn · thin `-->` = control/data flow · dotted `-.->` = config/data read.
