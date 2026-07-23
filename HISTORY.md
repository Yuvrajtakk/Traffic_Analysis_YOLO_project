# Development History & Journal

This document preserves the detailed, step-by-step development history of the Traffic Analysis YOLO Project, originally written during the project's inception. It serves as a historical record of the challenges faced, datasets rebuilt, and the training process.

---

## Phase 1 — Origin & Planning

### The brief
Build a real-time traffic monitoring system with an input toggle between file playback and a live camera feed, running on a custom-trained object detector, and layer four analytics modules on top of the raw detections — **stationary vehicle detection**, **wrong-way detection**, **hazard detection**, and **congestion detection**. The detector itself needed nine classes: **Bus, Car, Bike, Person, Animal, Fire, Smoke, Accident, Object_on_road**. (A tenth class, Truck, got added later — more on that in Phase 2.)

### What I knew going in
Before touching this project, my actual background was: seven classical ML algorithms built from scratch, an MNIST digit recognizer hitting 98.15% with an SVM, and comfort with Python/sklearn/numpy/matplotlib. I also had one directly relevant prior project — a YOLOv8 + ByteTrack player-tracking system I'd built and deployed on Streamlit — so I wasn't starting from zero on the tracking side. But I had **zero formal background in object detection or YOLO theory specifically**, and that gap is what the first real chunk of work went into.

### Learning the YOLO family before writing a line of project code
I ran a structured teaching session to actually understand the architecture instead of just calling `ultralytics.YOLO()` blind. The teaching contract was strict: build every mechanism by hand before touching library shortcuts, tie every new idea back to something I already understood from the SVM/MNIST work, and answer a checkpoint question in my own words before moving on.

What actually got built by hand in that process:
- Grid-based detection vs. the old two-stage sliding-window approach (explained via a "cardboard tube scanning a park" vs. "standing on a hill seeing the whole grid at once" analogy).
- IoU derived by hand and implemented from scratch, visualized across four overlap cases.
- YOLOv1's output tensor shape `(7,7,30)` simulated directly in PyTorch.
- YOLOv2's fixes explained in sequence, not in isolation.
- YOLOv3's Feature Pyramid Network — three detection scales (13×13, 26×26, 52×52) with 3 anchors each. 

After three separate research reports on the full v1–YOLO26 timeline, I restructured my own learning into three tiers:

- **Tier 1 (build from scratch):** v1, v2, v3, v8 — the actual backbone of debugging intuition.
- **Tier 2 (understand deeply, don't rebuild):** v4, v5, v10.
- **Tier 3 (one-paragraph mental model only):** v6, v7, v9, v11, v12, YOLO-NAS, YOLO-World, YOLO26.

### Why the specific architecture choices
- **YOLOv8s, not Nano:** justified by the hazard classes (Fire/Smoke/Accident/Object_on_road) being visually organic and cluttered — they need more backbone capacity to learn reliably.
- **Raw Ultralytics + ByteTrack, not a wrapper library:** I deliberately chose the un-wrapped, more manual path over something like the `supervision` library, trading a faster build for full debuggability.

---

## Phase 2 — Dataset Journey (v1 → v2)

### v1: how the data actually got sourced
This is worth stating plainly: **I did not manually annotate any of the 9 classes myself.** Vehicles could be pulled straight from COCO (already annotated), fire/smoke datasets already exist on Roboflow, and the animal class also exists in COCO. So v1 was built by pulling ~600 already-annotated images per class from existing Roboflow datasets and merging them into one combined project.

### What went wrong
The Car class became a dumping ground. I labeled Bus for actual buses, but everything else on four wheels — cars, taxis, and (critically) **trucks** — got merged under the single Car label. Mixing trucks, taxis, and cars under one label taught the model an inconsistent definition of "Car."

Pulling the real class counts directly from the Roboflow API confirmed the theory hard:

| Class | Images | Old mAP50 |
|---|---:|---:|
| Bike | 3,623 | 0.648 |
| Person | 2,594 | 0.781 |
| **Car** | **2,440** | **0.485 (worst)** |

Car had the *second-most* images in the whole dataset and still scored worst by a wide margin. 

### v2: the rebuild
Rather than starting a new Roboflow project, v2 was built as a new **version**. I sourced two Universe datasets:
- **Aerial Person Detection** (VisDrone-based), 7,015 images, overhead/drone angle.
- **Overhead Vehicle Detection**, 4,679 images, also overhead-angle.

`van`, `truck`, `tricycle`, and `awning-tricycle` were deliberately **excluded** from this pull — those are exactly the heavy-vehicle shapes that caused the original Car contamination.

### The Truck decision
The guidance I settled on was to **not** merge Bus and Truck together — collapsing two visually distinct shapes under one label is exactly the mistake that broke Car. Truck sourcing landed on two datasets for roughly 1,000 new Truck images.

### Augmentation and final shape
Training moved to **`imgsz=960`** for the v2 run. Final class taxonomy stood at **10 classes**: Bus, Car, Bike, Person, Animal, Fire, Smoke, Accident, Obj_On_Road, Truck.

The final, fully merged dataset (v1 + v2 + Truck sources combined) landed at **10,329 images** and **65,741 total annotations** — an average of 6.4 objects per image. 

---

## Phase 3 — Training, the Real Grind

### Local vs. Colab
My machine has an RTX 4050 (6GB confirmed). This forces `batch=2` or `4` at `imgsz=640` locally and risks OOM. The actual decision came down to one concrete test: I timed a single local epoch and did the math to project the full run, which came out to roughly **100 hours locally versus roughly 20 hours on Colab**. That projected gap is what settled it in Colab's favor.

### Checkpoint strategy
Colab's free tier caps a session at roughly 4–5 hours. That forced the checkpoint strategy directly: `patience=30` (so early stopping wouldn't trigger from normal epoch-to-epoch noise), `resume=True`, and `save_period=5` so a checkpoint existed every 5 epochs, not just at the end.

### Final metrics
The 100-epoch run on the rebuilt v2/10-class dataset finished cleanly. Final overall: **mAP50 = 0.721**, **mAP50-95 = 0.452**.

The training arc: epoch 23 (mAP50-95 0.380) → epoch 48 (0.436) → epoch 66 (**0.452, the actual best result**) → epoch 100 (0.449, final checkpoint). Most of the real improvement happened in the first two-thirds of the run.

Full per-class breakdown on the final validated model:

| Class | mAP50 | mAP50-95 |
|---|---:|---:|
| Obj_On_Road | 0.929 | 0.655 |
| Accident | 0.887 | 0.712 |
| Smoke | 0.859 | 0.547 |
| Animal | 0.801 | 0.462 |
| Car | 0.729 | 0.463 |
| Bus | 0.716 | 0.487 |
| Truck | 0.654 | 0.409 |
| Fire | 0.578 | 0.284 |
| Person | 0.554 | 0.267 |
| Bike | 0.506 | 0.237 |

Car climbed a long way from the old, contaminated-label 0.485 up to 0.729 — direct evidence the v1→v2 dataset rebuild actually fixed the problem it targeted.

Separately, a real domain-shift gap was measured on a validation check: Person detection, which validated at 96.7% on its own training distribution, and Bike similarly strong at baseline, both fell off a cliff — Bike to roughly 0%, Person to roughly 3% — once tested against footage from a different camera angle. That's a real, demonstrated generalization gap, not a hypothesis.

---

## Phase 4 — Building the Pipeline, Piece by Piece

### The shape of the architecture
`ingestion → tracker → TrackManager → four analytics modules → event_recorder → display`

One `TrackManager` sits as the single shared source of truth for every track's state — active, lost-in-buffer (occluded but not yet given up on), or purged — so all four analytics modules agree on what's still "alive" at any given moment. `hazards.py` is the one module that deliberately does **not** use `TrackManager`: fire, smoke, and accident detections don't have a meaningful stable identity to track frame-to-frame the way a vehicle does, so it works directly off raw per-frame detections instead.

### Each analytics module's real logic
- **`stationary.py`** — flags a vehicle whose centroid stays within a small pixel tolerance for a continuous window of `STATIONARY_DURATION_SEC`, read off the shared `TrackManager` history. It deliberately allows triggering once the observed window spans **≥90%** of the configured duration.
- **`wrong_way.py`** — flags a vehicle whose trajectory vector's cosine similarity to an authorized flow-vector direction drops below a threshold for a continuous `WRONG_WAY_DURATION_SEC` window.
- **`hazards.py`** — the one module working directly off raw per-frame detections rather than `TrackManager`. Uses a persistence/confidence-streak check with flicker tolerance.
- **`congestion.py`** — a pure per-frame snapshot question, "how many vehicles are inside the ROI right now," with no time-persistence logic.
- **`event_recorder.py`** — a timestamp-based (not frame-count-based) rolling deque buffer, specifically because RTSP FPS drifts in practice. Disk writes for MP4/JPG clips are offloaded to a background thread so the main loop never blocks on file I/O.

---

## Phase 5 — Testing Against Reality

### Systematic, isolated per-module testing
Testing ran module-by-module, deliberately in order of complexity — congestion first, then stationary, then wrong-way, hazard last. 

Real fixes that came out of this phase:
- **A frozen-frame chain-reaction bug**, found on a real (mobile-hotspot) RTSP run. `_open_capture()` raised a `RuntimeError` on a truly dead connection, and that exception wasn't caught inside the reconnect block — so it killed the entire background ingestion thread on the *first* failed reconnect attempt.
- **A stationary-event dedup bug**, found via a deliberate loop-test. `EVENT_COOLDOWN_SEC` was a single flat cooldown reused across all four modules — correct for short conditions like hazards, but wrong for a condition like "stationary" that can validly persist for tens of seconds.

---

## Phase 6 — Deployment / Finishing Phase

The **live tuning panel** and keyboard module toggles (`s`/`w`/`h`/`c`/`q`/`p`) replaced the original trackbar UI entirely. This was built and treated deliberately as a **developer convenience tool, not a permanent configuration editor** — there's no auto-persistence back into `config/thresholds.py`.

**Deployment decision:** shipped as a desktop OpenCV application using `cv2.imshow()`, deliberately *not* as a cloud or Streamlit deployment. `cv2.imshow()` needs a real local display/HighGUI context that a cloud target simply doesn't provide.

---

## Phase 7 — Reflection / What Was Learned

- **Verify a claim independently before building on it.** The VRAM figure (assumed 4.5GB on paper, confirmed 6GB via a direct `nvidia-smi` run).
- **Simple, explainable code beat clever code, consistently.** Keyboard toggles won over trackbars once trackbars caused a real platform-specific crash. A flat time-based event buffer beat a frame-count one specifically because it stays correct under FPS drift.
