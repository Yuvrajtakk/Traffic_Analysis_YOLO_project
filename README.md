# Traffic Analysis YOLO Project

Real-time traffic monitoring pipeline: YOLOv8 (custom-trained, 10-class)
+ ByteTrack for detection/tracking, four independent analytics modules,
and automatic before/after event clip recording — built to run against
either a live RTSP camera feed or a local test video.

## Pipeline

```
ingestion.py  --frames-->  tracker.py  --tracked detections-->  TrackManager
                                                                    |
                                        (shared state, every module reads the same truth)
                                                                    |
                    ┌───────────────┬───────────────┬───────────────┬───────────────┐
                    |               |               |               |
              stationary.py    wrong_way.py     hazards.py    congestion.py
                    └───────────────┴───────────────┴───────────────┘
                                            |
                                    event_recorder.py (before/after clip + still frame)
                                            |
                                    main.py (draws boxes, displays, keyboard controls)
```

## Model

- 10-class custom YOLOv8 model: `Car, Truck, Bus, Fire, Smoke, Bike,
  Animal, Person, Accident, Obj_On_Road`
- Trained at `imgsz=960` — inference **must** run at the same size
  (`config/thresholds.py: MODEL_IMGSZ`), or accuracy silently degrades
  with no error.
- Final validated metrics (100 epochs): mAP50 0.721, mAP50-95 0.452.
  Weakest classes: Bike (0.506 mAP50), Person (0.554), Fire (0.578) —
  small/thin objects at the current elevated camera angle. Strongest:
  Obj_On_Road (0.929), Accident (0.887), Smoke (0.859).
- Weights expected at `models/weights/new_best.pt` (not checked into
  git — see `.gitignore`).

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Drop the trained weights file at `models/weights/new_best.pt`.
`models/weights/` always holds the current best model as `new_best.pt`;
older versions live in `models/weights/archive/` if kept.

## Running

```bash
python main.py
```

At startup, the app asks:

```text
Input source — type 'file' or 'live':
```

Type `live` for the RTSP feed, or press Enter / type `file` for the
local demo-file mode. The app starts processing immediately; use the
keyboard controls below to show or hide the dashboard and tuning panel.

### Keyboard controls (focus must be on the Traffic Dashboard window)

| Key | Effect |
|---|---|
| `d` | Toggle Traffic Dashboard window open/closed |
| `t` | Toggle Tuning Panel window open/closed |
| `s` | Toggle Stationary Vehicle detection on/off |
| `w` | Toggle Wrong-Way detection on/off |
| `h` | Toggle Hazard (Fire/Smoke/Accident) detection on/off |
| `c` | Toggle Congestion detection on/off |
| `p` | Print current live-tuned threshold values to the console |
| `q` | Quit |

### Zone / ROI Calibration

```bash
python tools/calibrate_zones.py data/test_footage/sample.mp4
```

Keyboard input (`n` / `s` / `q`) must be pressed while the **Zone
Calibration** video window itself has focus, not the terminal.

## Live Tuning Panel

Every numeric threshold that's worth adjusting while watching the live
feed is exposed as a trackbar in the **Tuning Panel** window, backed by
`config/live_config.py`. Moving a slider takes effect on the very next
frame — no restart needed:

- YOLO detection confidence
- Stationary: duration (sec), pixel-movement tolerance, area-change tolerance
- Wrong-way: duration (sec), cosine-similarity threshold
- Hazard: confidence, persistence duration (sec)
- Congestion: capacity (vehicle count)

Once you land on values worth keeping, press `p` to print them to the
console, then copy them into `config/thresholds.py`'s defaults so
they persist as the new startup baseline.

**Not** live-tunable by design (structural, not threshold-tuning):
`VEHICLE_CLASSES`/`HAZARD_CLASSES` (taxonomy), `WRONG_WAY_ZONES` and
`CONGESTION_ROI_POLYGON_NORM` (per-camera geometry calibration), `DEBUG`.

## Module Architecture

- **`src/ingestion.py`** — threaded video capture (webcam/file/RTSP),
  auto-reconnect with retry limit, never blocks the main loop.
- **`src/tracker.py`** — `YOLOTracker` (YOLO + ByteTrack wrapper) and
  `TrackManager` (shared per-ID position history every analytics
  module reads from, so they can never disagree about track state).
- **`src/analytics/stationary.py`** — flags a vehicle stopped in place
  for N seconds (anchor-point re-fire logic tuned for handheld-camera
  jitter).
- **`src/analytics/wrong_way.py`** — flags a vehicle moving against
  the expected traffic direction, using **per-zone flow vectors**
  (multi-lane aware — see Known Limitations).
- **`src/analytics/hazards.py`** — flags Fire/Smoke/Accident/
  Obj_On_Road/Animal classes persisting confidently across frames.
- **`src/analytics/congestion.py`** — snapshot count of vehicles
  currently inside a calibrated ROI polygon.
- **`src/geometry.py`** — shared point-in-polygon helper used by both
  `congestion.py` and `wrong_way.py`'s zones.
- **`src/event_recorder.py`** — always-on rolling pre-event buffer +
  threaded before/after `.mp4` + `.jpg` writer, one independent
  collection per event so overlapping events never interfere.
- **`config/thresholds.py`** — every startup default and structural
  setting, single source of truth.
- **`config/live_config.py`** — the mutable object the tuning panel
  writes into; every analytics module reads live values from here.
- **`src/tuning_panel.py`** — the trackbar UI wired to `LiveConfig`.
- **`tools/calibrate_zones.py`** — click points on one video frame and
  print normalized polygons for `WRONG_WAY_ZONES` and
  `CONGESTION_ROI_POLYGON_NORM`.

## Recent Fixes

- Congestion now fires on threshold crossing instead of repeating every
  frame while crowded.
- Wrong-way cosine threshold now requires clearly opposite movement;
  the current test-video zones use a top/bottom split, not final traced
  camera geometry.
- Hazard persistence now tolerates flicker with a 3-of-5-frame rule.
- Detection labels now show confidence and track ID.
- `Obj_On_Road` and `Animal` now route through `hazards.py`.

## Deployment

This is a desktop OpenCV application using a `cv2.imshow` window, not a
web app. The intended deliverable is the GitHub repo plus a short demo
recording, not a cloud or Streamlit deployment, because the GUI display
window cannot run in a browser-hosted environment without a full rewrite
of the display layer.

## Known Limitations (current, by design — not bugs)

- **Wrong-way zone flow is approximate.** `WRONG_WAY_ZONES` in
  `thresholds.py` is now traced against real test footage, but the
  flow-vector direction is still an approximation for the test camera's
  angle and has not been verified against the real deployed camera yet.
- **Congestion ROI is placeholder geometry.** Same situation —
  `CONGESTION_ROI_POLYGON_NORM` is a rough rectangle, not the real
  road boundary for the deployed camera.
- **Hazard classes still depend on model quality** — the code now
  smooths flicker, but a confident repeated misclassification can still
  become an event.
- **Domain shift**: the model is trained predominantly on ground-level
  / vehicle-mounted footage, but tested from an elevated handheld
  camera angle — expect some misclassification tied to this gap, not a
  code bug.

## Roadmap

See `PROJECT_HANDOFF.md` for full session history and the next-phase
plan (per-camera zone/ROI calibration tooling, repo packaging,
Obj_On_Road handling decision, broader flicker-tolerant hazard logic).
