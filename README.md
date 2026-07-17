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

## Running

Edit `SOURCE` and `WEIGHTS` at the top of `main.py`:

```python
SOURCE = "rtsp://<camera-ip>:<port>/"   # or a local file: "data/test_footage/sample.mp4"
WEIGHTS = "models/weights/new_best.pt"
```

```bash
python main.py
```

Two windows open:

- **Traffic Dashboard** — the live feed with detection boxes and
  module on/off status.
- **Tuning Panel** — trackbars for every live-tunable threshold (see
  below).

### Keyboard controls (focus must be on the Traffic Dashboard window)

| Key | Effect |
|---|---|
| `s` | Toggle Stationary Vehicle detection on/off |
| `w` | Toggle Wrong-Way detection on/off |
| `h` | Toggle Hazard (Fire/Smoke/Accident) detection on/off |
| `c` | Toggle Congestion detection on/off |
| `p` | Print current live-tuned threshold values to the console |
| `q` | Quit |

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
- **`src/analytics/hazards.py`** — flags Fire/Smoke/Accident classes
  persisting confidently across frames.
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

## Known Limitations (current, by design — not bugs)

- **Wrong-way zones are placeholder geometry.** `WRONG_WAY_ZONES` in
  `thresholds.py` is a rough vertical half-split, not a traced lane
  boundary. Real deployment needs a human tracing the actual lane
  edges for the specific camera.
- **Congestion ROI is placeholder geometry.** Same situation —
  `CONGESTION_ROI_POLYGON_NORM` is a rough rectangle, not the real
  road boundary for the deployed camera.
- **Hazard persistence has zero tolerance for detection flicker** — a
  single missed frame resets the streak, even if the real hazard never
  stopped. A "confident in 8 of last 10 frames" version would be more
  forgiving; deliberately deferred.
- **Domain shift**: the model is trained predominantly on ground-level
  / vehicle-mounted footage, but tested from an elevated handheld
  camera angle — expect some misclassification tied to this gap, not a
  code bug.
- **`Obj_On_Road` and `Animal`** are real trained classes with no
  analytics module built around them yet (see
  `UNUSED_TRAINED_CLASSES` in `thresholds.py`).

## Roadmap

See `PROJECT_HANDOFF.md` for full session history and the next-phase
plan (per-camera zone/ROI calibration tooling, repo packaging,
Obj_On_Road handling decision, broader flicker-tolerant hazard logic).
