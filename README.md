# Real-Time YOLOv8 Traffic Monitoring System

**Author:** Yuvraj Tak | **Internship:** AI/ML Intern, Watsoo Express Pvt. Ltd.
**Mentor:** Ankit Gupta | **Repo:** github.com/Yuvrajtakk/Traffic_Analysis_YOLO_project

## TL;DR — Read This If You're Short on Time

A real-time traffic monitoring pipeline: a custom-trained 10-class
YOLOv8s detector + ByteTrack tracking, feeding four independent
analytics modules that each watch for a different real-world traffic
condition, with automatic before/after clip recording whenever one
fires.

**Detects:** Bus, Car, Bike, Person, Animal, Fire, Smoke, Accident,
Obj_On_Road, Truck.

**What it does with those detections:**

| Module | Triggers when... |
|---|---|
| Stationary Vehicle | a vehicle stays parked/stopped for 4+ seconds |
| Wrong-Way | a vehicle moves against the expected traffic direction for 4+ seconds |
| Hazard | Fire, Smoke, or Accident is confidently detected and persists |
| Congestion | vehicle count in the monitored road area exceeds capacity |

Every trigger saves an annotated still image **and** a short before/after
video clip automatically — no manual review needed to catch an event.

**Final model accuracy:** mAP50 0.721, mAP50-95 0.452 (best result was
at epoch 66 of a 100-epoch run; final checkpoint essentially matched
it). Strongest class: Obj_On_Road (0.929 mAP50). Weakest: Bike (0.506
mAP50) — a small-object/elevated-camera-angle limitation, not a data
problem (full breakdown in Phase 3 below).

### How to run it
```bash
pip install -r requirements.txt
python main.py
```
It'll ask: `Input source — type 'file' or 'live':`
- `file` → runs against a bundled test clip
- `live` → connects to a live RTSP camera feed

### Keyboard controls
(Click the small control window that opens first, then press keys —
OpenCV only receives keypresses when one of its windows has focus.)

| Key | Effect |
|---|---|
| `d` | Show/hide the main video dashboard |
| `t` | Show/hide the live tuning panel (dev tool — adjusts detection thresholds on the fly, doesn't save changes permanently) |
| `s` | Toggle Stationary detection on/off |
| `w` | Toggle Wrong-Way detection on/off |
| `h` | Toggle Hazard detection on/off |
| `c` | Toggle Congestion detection on/off |
| `p` | Print current tuning values to console |
| `q` | Quit |

Saved events land in `outputs/events/` — one `.jpg` still and one
`.mp4` clip per triggered event, named by which module fired it, which
class, which tracked vehicle ID, and the exact timestamp, e.g.
`stationary_Car_id3_182052_716.jpg`.

### If you just want to see it work, not read the whole story
Everything below this line is the full, honest development journal —
research, dataset rebuilds, training obstacles, testing, and every real
bug hit along the way. It's long on purpose, written as a personal
record of how this was actually built. Skip to whichever phase
interests you, or stop here — the section above covers everything
needed to actually run and evaluate the project.

---

# The Full Development Journal

This is the real story of how this project got built — not the cleaned-up version. I'm writing it in chronological phases, pulling the actual numbers, filenames, and decisions from my own conversation history as I worked through it. Where something was genuinely unclear or showed up differently in two different sessions, I flagged it with **❓** instead of quietly picking one version — those have since been resolved and folded in below; see the note at the very end.

---

## Phase 1 — Origin & Planning

### The brief
Ankit sir's brief was: build a real-time traffic monitoring system with an input toggle between file playback and a live camera feed, running on a custom-trained object detector, and layer four analytics modules on top of the raw detections — **stationary vehicle detection**, **wrong-way detection**, **hazard detection**, and **congestion detection**. The detector itself needed nine classes: **Bus, Car, Bike, Person, Animal, Fire, Smoke, Accident, Object_on_road**. (A tenth class, Truck, got added later — more on that in Phase 2.)

### What I knew going in
Before touching this project, my actual background was: seven classical ML algorithms built from scratch, an MNIST digit recognizer hitting 98.15% with an SVM, and comfort with Python/sklearn/numpy/matplotlib. I also had one directly relevant prior project — a YOLOv8 + ByteTrack player-tracking system I'd built and deployed on Streamlit — so I wasn't starting from zero on the tracking side. But I had **zero formal background in object detection or YOLO theory specifically**, and that gap is what the first real chunk of work went into.

### Learning the YOLO family before writing a line of project code
I ran a structured teaching session (first with Gemini, then continued and reset with Claude) to actually understand the architecture instead of just calling `ultralytics.YOLO()` blind. The teaching contract was strict: build every mechanism by hand before touching library shortcuts, tie every new idea back to something I already understood from the SVM/MNIST work, and answer a checkpoint question in my own words before moving on.

What actually got built by hand in that process:
- Grid-based detection vs. the old two-stage sliding-window approach (explained via a "cardboard tube scanning a park" vs. "standing on a hill seeing the whole grid at once" analogy) — the old approach's two real failure modes were speed (~2000 forward passes per image) and context blindness (isolated crops misreading background as objects).
- IoU derived by hand and implemented from scratch, visualized across four overlap cases.
- YOLOv1's output tensor shape `(7,7,30)` simulated directly in PyTorch.
- YOLOv2's fixes explained in sequence, not in isolation: batch norm replacing dropout in every conv layer (dropout and batch norm actually *fight* each other — dropout's random zeroing skews the batch statistics batch norm depends on, adding noise on noise instead of extra regularization), and a pre-detection classifier fine-tune at 448×448 for 10 epochs so the backbone adjusts to the larger input resolution *before* detection complexity gets added on top.
- YOLOv3's Feature Pyramid Network — three detection scales (13×13, 26×26, 52×52) with 3 anchors each. I got the "why merge deep into shallow" checkpoint wrong on the first pass (said shallow layers "can't capture small objects" — actually backwards): shallow layers have fine spatial resolution but weak semantics (few conv ops, only edges/corners so far), deep layers have strong semantics but have downsampled away the spatial detail small objects need. The FPN merge gives the high-resolution shallow map access to the deep map's semantic strength without losing localization precision.

After three separate research reports on the full v1–YOLO26 timeline (which disagreed with each other specifically on YOLOv12 — one called Area Attention a real breakthrough, the other flagged mAP drops and up to 3× slower CPU inference vs. YOLO11 in production), I restructured my own learning into three tiers instead of treating all ~15 versions equally, specifically to keep this useful for an internship rather than becoming a research rabbit hole:

- **Tier 1 (build from scratch):** v1, v2, v3, v8 — the actual backbone of debugging intuition.
- **Tier 2 (understand deeply, don't rebuild):** v4 (Bag of Freebies/Specials — Mosaic aug, CIoU loss, SPP, PANet — absorbed into everything downstream), v5 (the PyTorch/Ultralytics transition v8's codebase inherits from), v10 (NMS-free dual assignment, the direction the family is heading).
- **Tier 3 (one-paragraph mental model only):** v6, v7, v9, v11, v12, YOLO-NAS, YOLO-World, YOLO26.

Separately, I also built a full set of 14 standalone practical notebooks (one per YOLO version, v1 through YOLO26) as a hands-on comparison exercise, and ran a Roboflow number-plate-recognition dataset through training as a first, low-stakes practice rep with the Roboflow platform itself before touching the real 9-class dataset. Both were deliberately kept separate from the actual traffic project — practice reps, not project deliverables.

### Why the specific architecture choices
- **YOLOv8s, not Nano:** justified by the hazard classes (Fire/Smoke/Accident/Object_on_road) being visually organic and cluttered — they need more backbone capacity to learn reliably. This wasn't a speed decision; there was plenty of speed headroom either way.
- **Raw Ultralytics + ByteTrack, not a wrapper library:** I deliberately chose the un-wrapped, more manual path over something like the `supervision` library, trading a faster build for full debuggability — every tensor, every `None`-check, every index is something I actually wrote and understand.

---

## Phase 2 — Dataset Journey (v1 → v2)

### v1: how the data actually got sourced
This is worth stating plainly because it surprised Ankit sir when I explained it: **I did not manually annotate any of the 9 classes myself.** When I first proposed doing full annotation, Ankit sir corrected me directly — vehicles could be pulled straight from COCO (already annotated), fire/smoke datasets already exist on Roboflow, and the animal class also exists in COCO. So v1 was built by pulling ~600 already-annotated images per class from existing Roboflow datasets and merging them into one combined project (`traffic-monitoring-9class`).

### What went wrong
The Car class became a dumping ground. I labeled Bus for actual buses, but everything else on four wheels — cars, taxis, and (critically) **trucks** — got merged under the single Car label. In my own words to Ankit sir at the time: *"I annotated bus label for buses and all the other vehicles as car which includes truck, car, taxi etc which leads to blurry prediction."* Two other real symptoms showed up during testing: the model would fire a false Accident event whenever a specific red-dressed woman walked near any vehicle, and at one point it detected my own face as an Animal.

Pulling the real class counts directly from the Roboflow API (not guessed) confirmed the theory hard:

| Class | Images | Old mAP50 |
|---|---:|---:|
| Bike | 3,623 | 0.648 |
| Person | 2,594 | 0.781 |
| **Car** | **2,440** | **0.485 (worst)** |
| Obj_On_Road | 1,209 | 0.934 |
| Animal | 1,200 | 0.766 |
| Fire | 1,104 | 0.575 |
| Bus | 987 | 0.839 |
| Accident | 884 | 0.854 |
| Smoke | 772 | 0.869 |

Car had the *second-most* images in the whole dataset — more than double Bus — and still scored worst by a wide margin. That's not a data-volume problem, it's a data-consistency problem: mixing trucks, taxis, and cars under one label taught the model an inconsistent definition of "Car."

A second, separate bug surfaced around the same time: the dataset had been generated at 640×640, but inference was running at `imgsz=1280`. Training and inference were happening at two different resolutions — likely a real contributor to weak small-object performance (Bike, Fire) on top of the label-mixing issue.

### v2: the rebuild
Rather than starting a new Roboflow project, v2 was built as a new **version** inside the existing `traffic-monitoring-9class` project — versions support exactly this kind of iteration. I sourced two Universe datasets I'd already forked into my workspace:

- **Aerial Person Detection** (VisDrone-based), 7,015 images, overhead/drone angle — matching my actual camera's elevated viewpoint.
- **Overhead Vehicle Detection**, 4,679 images, also overhead-angle, kept in reserve as a backup Car source.

From the VisDrone source specifically, the class remap and pull counts were:

| Source class | Maps to | Pulled |
|---|---|---:|
| car | Car | 700 |
| bus | Bus | 700 |
| bicycle | Bike | 400 |
| motor | Bike | 400 |
| pedestrian | Person | 300 |
| people | Person | 300 |

`van`, `truck`, `tricycle`, and `awning-tricycle` were deliberately **excluded** from this pull — those are exactly the heavy-vehicle shapes that caused the original Car contamination, and importing them under Car would have undone the fix before it started. `motor` (motorcycle) was mapped into Bike rather than dropped, since on Indian traffic footage motorcycles/scooters are the dominant two-wheeler, not bicycles. `ignored regions` and `others` were discarded outright — not real object classes.

### The Truck decision
Partway through, I went back to Ankit sir with the contamination problem and asked directly: *"Sir may I include truck as different class beside car, bike and bus? Or I can merge heavy vehicles together and label it as bus or truck."* His reply confirmed the annotation approach (pre-annotated data, no manual labeling needed) — and he approved the Truck class itself, so this went into the taxonomy as a fully confirmed 10th class, not a provisional one.

The guidance I settled on was to **not** merge Bus and Truck together, even though Ankit sir floated that as an alternative — collapsing two visually distinct shapes under one label is exactly the mistake that broke Car. Truck sourcing landed on two datasets: `kmec/truck-detection` (1,424 images total, 700 pulled) and `khulna-university/truck-detection` (1,800 images total, 300 pulled), for roughly 1,000 new Truck images — sized to land near Bus's original count so the new class wouldn't start starved.

A triage plan was also drawn up (not confirmed executed) for the *existing* contaminated Car images: run a stock COCO-pretrained YOLO over the old 2,440 Car images, and flag anything it confidently calls "truck" (>0.5 confidence) as a relabel candidate, rather than manually re-reviewing all 2,440 images blind.

### Augmentation and final shape
Roboflow's own augmentation settings were never actually in play — the real training ran outside Roboflow, in Colab, directly through the `ultralytics` library, so augmentation had to be set in the `model.train()` call itself:

```
hsv_h=0.02, hsv_s=0.7, hsv_v=0.5   # more color/lighting variation across merged sources
mosaic=1.0                          # kept on
degrees=10                          # birds-eye cameras tilt slightly; small rotation tolerance
copy_paste=0.3                      # boosts rare classes (Fire, Smoke, Accident) without new raw images
```

Training moved to **`imgsz=960`** for the v2 run (the working folder was literally named `v2_960_100epoch_real`) — a middle ground addressing the earlier 640-train/1280-inference mismatch. Final class taxonomy stood at **10 classes**: Bus, Car, Bike, Person, Animal, Fire, Smoke, Accident, Obj_On_Road, Truck.

The final, fully merged dataset (v1 + v2 + Truck sources combined) landed at **10,329 images** and **65,741 total annotations** — an average of 6.4 objects per image. Final per-class annotation counts:

| Class | Annotations |
|---|---:|
| Car | 23,183 |
| Person | 18,060 |
| Bike | 11,480 |
| Bus | 3,923 |
| Truck | 3,211 |
| Accident | 1,599 |
| Obj_On_Road | 1,209 |
| Animal | 1,200 |
| Fire | 1,104 |
| Smoke | 772 |

Car, Person, and Bike ended up far ahead of the rest by volume — largely a byproduct of the dense, multi-object VisDrone-style overhead sources used for the v2 rebuild, where a single image can contain dozens of cars/people/bikes at once, versus the hazard classes (Fire, Smoke, Accident) which are inherently one-or-few-objects-per-image. That volume imbalance is worth keeping in mind against the final per-class accuracy numbers in Phase 3 — Bike ending up the weakest class despite having the third-highest instance count is exactly the kind of result that volume alone doesn't explain.

---

## Phase 3 — Training, the Real Grind

### Local vs. Colab
My machine has an RTX 4050. The project plan originally assumed ~4.5GB of usable VRAM; running `nvidia-smi` directly corrected that to **6GB confirmed** (6,141 MiB), a real example of not trusting an assumption on paper. Even with the corrected number, 6GB forces `batch=2` or `4` at `imgsz=640` locally and risks OOM, versus Colab's free T4 giving 14,913 MiB (~15GB) and a comfortable `batch=16`.

Ankit sir had directly told me to "train locally if your laptop supports it," which I took seriously rather than just defaulting to Colab — so I set an explicit fallback trigger instead of an open-ended "try and see": if `batch=4` at `imgsz=640` threw an OOM error, or a single epoch took noticeably long (over ~10–15 minutes), fall back to Colab. `imgsz=416` locally was also considered as a lower-memory compromise. I looked seriously at combining both — a real distributed setup splitting one training job across my local GPU and Colab's GPU simultaneously via something like a Ray cluster — and rejected it: my dataset was small (500–600 images per class per source), the model wasn't huge, and home-internet-to-Colab latency plus the fragility of a coordinator server just weren't worth it for a project on a 1–2 day ship clock. The actual decision came down to one concrete test: I timed a single local epoch, did the math to project the full run, and it came out to roughly **100 hours locally versus roughly 20 hours on Colab** for the same target epoch count. That projected gap — not an open-ended "try and see" — is what settled it in Colab's favor; the local GPU was kept for fast smoke tests on a tiny data slice instead, to catch bugs in the training script before burning a full Colab session on something broken.

### Session limits and checkpoint strategy
Colab's free tier caps a session at roughly 4–5 hours before disconnecting. That forced the checkpoint strategy directly: `patience=30` (so early stopping wouldn't trigger from normal epoch-to-epoch noise), `resume=True`, and `save_period=5` so a checkpoint existed every 5 epochs, not just at the end.

### The checkpoint-chaos saga
This is the part of the story that cost the most real hours, and it's worth telling straight rather than smoothing over.

Training bounced across a string of Google accounts as each one's Colab compute ran out — named in the transcripts I have: the original owner **takcommunity99**, then **yuvrajtak651**, then a third account, **imuv**. Across the full saga (including the later stretch finishing epoch 74 → 100), the real total was closer to **~5 account switches**, not just the three named above.

The `yuvrajtak651` run was the good one — it went the full stretch from epoch 23 to 49 with **zero disconnects**, climbing cleanly the whole way:

| Epoch | mAP50 | mAP50-95 |
|---|---:|---:|
| 23 | 0.649 | 0.380 |
| 30 | 0.678 | 0.408 |
| 40 | 0.696 | 0.425 |
| 48 | 0.710 | 0.436 |

It died mid-epoch 49 (511 of 937 batches in), with `patience=30` never even close to triggering — still gaining every few epochs.

Then the actual chaos: switching to the third account (`imuv`), I mounted Drive, checked for `last.pt` at the expected path, found it, and resumed — except training restarted from **epoch 21**, not 48. The root cause turned out to be two separate Drive folder objects that looked interchangeable but weren't: a standalone `v2_960_100epoch_real` folder shared directly at 2:49 PM (the real one, from `yuvrajtak651`'s clean run), and a different, nested `v2_960_100epoch_real` folder living inside a shared `traffic_project` folder from `takcommunity99`, shared later at 8:25 PM — and stuck around epoch 20. I'd resumed from the wrong one. In my own words at the time: *"i have traverse all the folder but couldn't find epoches more than 20... please help me?? i wasted so many hours and energy."*

The actual fix was to stop manually browsing folders and **search Drive by filename** directly (`last.pt`, `epoch45.pt`) instead — that's what surfaced the real folder, still sitting on the `yuvrajtak651` account, containing checkpoints from epoch 25 through 45. I shared it fresh to `imuv`, renamed the new shortcut to avoid yet another name collision, and — critically — verified the *real* latest checkpoint by running `os.listdir()` on the weights folder and reading the actual epoch-numbered filenames present, rather than trusting any `last.pt` file's modified timestamp (mtimes proved unreliable across shared-Drive, multi-account contexts). That became a locked-in rule for the rest of the project: **always confirm via `os.listdir()`, never trust `last.pt`'s mtime.**

### Final metrics
The 100-epoch run on the rebuilt v2/10-class dataset finished cleanly — 100/100 epochs, no more checkpoint hunting needed. Final overall: **mAP50 = 0.721**, **mAP50-95 = 0.452**.

The training arc is worth stating plainly rather than implying all 100 epochs contributed equally: epoch 23 (mAP50-95 0.380) → epoch 48 (0.436) → epoch 66 (**0.452, the actual best result**) → epoch 100 (0.449, final checkpoint, essentially flat since ~66). Most of the real improvement happened in the first two-thirds of the run; the last third mostly held steady rather than climbing further — normal behavior, not a sign anything went wrong. The deployed model is effectively the epoch-66 result, carried forward to the epoch-100 checkpoint with negligible drift.

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

Bike and Person are the two weakest classes, and notably that's *not* a data-volume problem — both have high instance counts (2,023 and 2,621 respectively). It lines up with the small-object-at-elevated-angle hypothesis: at 960px input, from the camera's overhead viewpoint, bikes and people occupy a genuinely small footprint in the frame, which is a harder detection problem than the class label alone suggests. Worth attention in a future retraining phase, not an immediate fix. Car, by contrast, climbed a long way from the old, contaminated-label 0.485 up to 0.729 — direct evidence the v1→v2 dataset rebuild actually fixed the problem it targeted.

Separately, a real domain-shift gap was measured on a validation check: Person detection, which validated at 96.7% (29 of 30 known-Person validation images correctly detected) on its own training distribution, and Bike similarly strong at baseline, both fell off a cliff — Bike to roughly 0%, Person to roughly 3% — once tested against footage from a different camera angle. The training data was mostly ground-level/street-level shots; the real deployment camera is handheld and elevated. That's a real, demonstrated generalization gap, not a hypothesis — small objects (Bike, Person) at an elevated angle appear to be the classes most exposed to it.

---

## Phase 4 — Building the Pipeline, Piece by Piece

### The shape of the architecture
`ingestion → tracker → TrackManager → four analytics modules → event_recorder → display`

One `TrackManager` sits as the single shared source of truth for every track's state — active, lost-in-buffer (occluded but not yet given up on), or purged — so all four analytics modules agree on what's still "alive" at any given moment, instead of each module keeping its own private, potentially inconsistent bookkeeping. `hazards.py` is the one module that deliberately does **not** use `TrackManager`: fire, smoke, and accident detections don't have a meaningful stable identity to track frame-to-frame the way a vehicle does, so it works directly off raw per-frame detections instead.

### Building main.py, piece by piece
This was built incrementally with real review at each step, not dumped in all at once. The final loop breaks into seven pieces: build every module and prime the first real frame (needed so `congestion.py` can build its ROI polygon against real `frame_width`/`frame_height`); set up the window; run YOLO tracking and update the shared `TrackManager`; run each analytics module's `check()` gated behind a toggle, always initializing results to an empty list first so a module being off looks identical, code-wise, to a module firing nothing; hand any fired events to `event_recorder` while feeding *every* frame (not just event frames) into its rolling pre-event buffer; draw boxes and labels; display.

Real bugs hit along the way, kept honest rather than smoothed over:
- A PowerShell command that ran `New-Item ...` and `git add .` on the same line with no separator silently created zero of the intended files — confusing early `git status` output until the two commands were split apart.
- A Windows-specific `cv2.getTrackbarPos` crash (`NULL window` error) — the window object existed in Python before the OS had actually finished registering it on screen. First patched with an extra `cv2.waitKey(1)` tick right after window creation; later, the entire trackbar UI was scrapped in favor of keyboard toggles (`s`/`w`/`h`/`c` to flip stationary/wrong-way/hazards/congestion, `q` to quit, `p` to print a tuning snapshot), which eliminated the whole bug class at once and also caught a stray, accidentally-duplicated `cv2.createTrackbar("Congestion", ...)` line in the process.
- Video that looked like stop-motion during local testing — not a bug. CPU-only inference (no GPU in that test context) couldn't keep pace with the background ingestion thread reading frames as fast as the file allowed, so the displayed frame was always hundreds of frames stale by the time it painted. This matched an earlier measured ~2.5 FPS CPU benchmark and was left as-is (a known, explainable limitation of CPU-only testing) rather than "fixed," aside from an optional cosmetic `time.sleep()` if the choppiness was distracting.
- A stray `tempCodeRunnerFile.py` (a leftover VS Code Code Runner artifact) had actually been committed to git — deleted and gitignored.
- `README.md` existed in the repo but was committed at 0 bytes — flagged and filled in later.
- A real event-labeling bug: `event_type` was being set to the *detected class name* (e.g. "Car"), not the module that fired the event — so once more than one analytics module was active, there was no way to tell from a filename or log line whether a "Car" event came from Stationary or Congestion. Fixed by tagging each event dict with its source module before merging results, renaming files to `{module}_{class_name}_id{track_id}_{HHMMSS_ms}` format, and adding structured `[EVENT TRIGGERED]` / `[EVENT SAVED]` console logging.
- A slow-import scare that turned out not to be a crash at all: the project lived inside a OneDrive-synced folder, and OneDrive's cloud-placeholder file checks made `import torch` (thousands of files) take 30 seconds to 2+ minutes. What looked like a hang was actually me hitting Ctrl+C mid-import — moving `venv/` outside OneDrive was the real fix.
- The actual reason Python had to be pinned to 3.10/3.11 wasn't Python itself — it was `lapx`, the compiled C-extension linear-assignment solver ByteTrack uses internally for its Hungarian-algorithm matching. PyPI only ships prebuilt wheels for a narrow band of Python versions at a time; anything newer falls back to compiling from source, which on Windows needs Visual Studio Build Tools most people don't have installed.
- A low-risk structural note, flagged but deliberately not acted on: `TrackManager` and `YOLOTracker` share one `tracker.py` file, which imports `ultralytics` at module load time — so importing `TrackManager` alone for a "pure logic" unit test drags in the entire `ultralytics`/`torch` stack anyway. Not a bug, just a future cleanup candidate (splitting the file) that wasn't worth doing mid-project with nothing actually broken.

### Each analytics module's real logic
- **`stationary.py`** — flags a vehicle whose centroid stays within a small pixel tolerance for a continuous window of `STATIONARY_DURATION_SEC`, read off the shared `TrackManager` history. It deliberately allows triggering once the observed window spans **≥90%** of the configured duration (e.g., 3.6s of a 4.0s window) instead of demanding the literal full window — avoiding an artificial extra delay. This design choice got directly validated during testing: a unit test that assumed a strict full-4.0s threshold failed, and the failure turned out to be the *test's* wrong assumption, not the detector's logic.
- **`wrong_way.py`** — flags a vehicle whose trajectory vector's cosine similarity to an authorized flow-vector direction drops below a threshold for a continuous `WRONG_WAY_DURATION_SEC` window, off the same shared `TrackManager` history. It evolved from a single global flow vector to per-zone, multi-lane zones — multi-lane wrong-way detection was explicitly named as a next-phase coding target in one handoff and was later confirmed shipped.
- **`hazards.py`** — the one module working directly off raw per-frame detections rather than `TrackManager`, since fire/smoke/accident/animal/debris don't have a stable trackable identity the way a vehicle does. Uses a persistence/confidence-streak check with flicker tolerance: a short gap in detection doesn't kill an in-progress streak (avoiding false negatives from smoke flickering in and out of detection), but a large enough gap does reset it.
- **`congestion.py`** — the simplest of the four logically: a pure per-frame snapshot question, "how many vehicles are inside the ROI right now," with no time-persistence logic at all. Needs real `frame_width`/`frame_height` from the first successfully-read frame to build its ROI polygon.
- **`event_recorder.py`** — a timestamp-based (not frame-count-based) rolling deque buffer, specifically because RTSP FPS drifts in practice and a frame-count buffer would silently become the wrong real-world duration. Disk writes for MP4/JPG clips are offloaded to a background thread so the main loop never blocks on file I/O. A real robustness gap was found and fixed here: events triggered close to shutdown were losing their video clip entirely, because the "after" frame-collection window never got a chance to finish before the program exited — fixed by adding a `flush_pending()` call, verified afterward by confirming all 24 test events produced both a JPG and an MP4.

---

## Phase 5 — Testing Against Reality

### First contact with the real camera
The first real-world RTSP attempt, against the actual guest-house camera, surfaced two separate problems: a network-side stream timeout, and an OpenCV-variant dependency conflict (a headless-opencv install trap).

### The phone-as-RTSP-camera trick
With the real camera unreachable, I built a working substitute: downloaded a relevant YouTube clip, and used VLC on my phone to convert and stream it out as a live RTSP feed to a local loopback address (`rtsp://127.0.0.1:8554/mystream`). That gave the pipeline a genuine, controllable RTSP source to test the *real* code path against — reconnect logic, TCP transport, decode handling — without depending on physical camera access. It worked, and it turned out to be the actual backbone of the entire testing phase that followed.

### Systematic, isolated per-module testing
Testing ran module-by-module, deliberately in order of complexity — congestion first (simplest, a pure per-frame count), then stationary (needs a vehicle held in frame long enough), then wrong-way (most fiddly, dependent on correct zone/flow-vector setup), hazard last (rarest usable footage). At each stage, the other three modules were toggled off via the keyboard shortcuts so failures could be attributed to the right module. Test footage was sourced from targeted YouTube searches per module — CCTV traffic-jam footage for congestion, hard-shoulder-breakdown footage for stationary, wrong-way-driver CCTV compilations for wrong-way, and highway-incident compilations plus specific searches like "truck tire blowout" and "deer crossing" for the hazard classes.

Real fixes that came out of this phase:
- **The event-type/filename confusion** described in Phase 4.
- **A frozen-frame chain-reaction bug**, found on a real (mobile-hotspot) RTSP run. The h264 decode errors visible in the console (`error while decoding MB...`) are normal packet-loss noise on an imperfect mobile connection — not a defect, and already mitigated by forcing TCP transport over UDP. But underneath that noise sat a genuine bug: `_open_capture()` raised a `RuntimeError` on a truly dead connection, and that exception wasn't caught inside the reconnect block — so it killed the entire background ingestion thread on the *first* failed reconnect attempt, instead of allowing the configured retry budget to run. Worse, `read()` had no way to know the thread had died, so it just kept re-serving the same cached frame forever. Because every vehicle's centroid then appeared perfectly frozen, `stationary.py` — correctly, given what it was being fed — interpreted that as every vehicle simultaneously stopping, and fired repeatedly. About 106 event files piled up in a 3–4 minute window before this was diagnosed. I'd actually reasoned my own way to the right root cause before getting confirmation: *"i guess in 3 to 4 minutes there are 106 videos recorded... i guess in that video is stationary video when connection broken cause accident videos are working."* The fix wrapped the reconnect's `_open_capture()` call in a `try/except`, and taught `read()` to also check the thread's running flag before handing back a frame.
- **A stationary-event dedup bug**, found via a deliberate loop-test: a parked-SUV clip set to loop kept re-firing a stationary event roughly every 2–9 seconds for as long as it stayed in frame, not once per loop as first assumed. Root cause: `EVENT_COOLDOWN_SEC` was a single flat cooldown reused across all four modules — correct for short, bursty conditions like hazards or congestion, but wrong for a condition like "stationary" that can validly persist for tens of seconds. With an ~18.5s dwell time and a 2s cooldown, 4–5 events per pass was mathematically expected, not a fluke; the ~40.7s gap observed between clusters was the loop boundary itself. A streak-scoped, one-shot-per-incident redesign (fire once, then suppress entirely until the vehicle genuinely starts moving again) was proposed as the real long-term fix, rather than just lengthening the cooldown arbitrarily.
- **Domain-shift sensitivity**, confirmed rather than assumed (see Phase 3's final metrics) — treated as an honest, documented limitation rather than something to keep chasing with code, since no model generalizes perfectly outside its training distribution and this one was outside it on two axes at once (camera angle and handheld shake).

### The zone/ROI calibration tool
Built specifically because flow-vector directions and lane zones needed to be traced and set by hand against real footage, not guessed in code. Getting it right took genuine trial and error — at one point tracing what turned out to be only one lane instead of two, and guessing the wrong-way flow-vector direction incorrectly more than once — before landing on correct zones through direct observation of real footage rather than assumption.

---

## Phase 6 — Deployment / Finishing Phase

The **live tuning panel** and keyboard module toggles (`s`/`w`/`h`/`c`/`q`/`p`) replaced the original trackbar UI entirely. This was built and treated deliberately as a **developer convenience tool, not a permanent configuration editor** — there's no auto-persistence back into `config/thresholds.py`. Changing a threshold for good still means editing that file directly; the panel is for live experimentation and debugging, not for shipping config changes.

The **input-mode toggle** (file vs. live) was built to satisfy the original brief's explicit requirement directly.

The dashboard/tuning-panel's window-visibility behavior was shaped by a real OpenCV constraint: `cv2.waitKey()` — the call that both captures keypresses and lets a HighGUI window actually repaint — only functions against a live window, which constrained how visibility toggling could work.

**Final repo cleanup** removed the stray `tempCodeRunnerFile.py` artifact (gitignored against recurrence) and filled in the previously-empty `README.md`.

**Deployment decision:** shipped as a desktop OpenCV application using `cv2.imshow()`, deliberately *not* as a cloud or Streamlit deployment. `cv2.imshow()` needs a real local display/HighGUI context that a cloud target simply doesn't provide — a desktop app was the only architecturally honest choice given that constraint, not a fallback.

---

## Phase 7 — Reflection / What Was Learned

A few lessons showed up more than once across this project, in different disguises each time:

- **Verify a claim independently before building on it.** The VRAM figure (assumed 4.5GB on paper, confirmed 6GB via a direct `nvidia-smi` run) and the Colab checkpoint saga (never trust `last.pt`'s modified timestamp across shared-Drive, multi-account contexts — enumerate the real epoch-numbered files with `os.listdir()` instead) are the two clearest, most expensive examples of this.
- **Checkpoint discipline matters as much as the training run itself.** The multi-account chaos cost real hours specifically because two differently-pathed Drive folders looked interchangeable and weren't — a problem entirely orthogonal to model quality.
- **Simple, explainable code beat clever code, consistently.** Keyboard toggles won over trackbars once trackbars caused a real platform-specific crash. A flat time-based event buffer beat a frame-count one specifically because it stays correct under FPS drift, even though it's the slightly less "obvious" first design to reach for.
- **Don't trust a tool's own success message.** Several points in the process involved independently re-verifying a claim rather than accepting it at face value — a weights URL typed from memory during the multi-notebook YOLO build (caught wrong, corrected against the real release tag), and a full byte-for-byte, file-by-file diff of a delivered zip when its authenticity was in doubt, rather than trusting a hash alone.

**Known limitations, honestly stated as of the last documented state:**
- Zone/flow-vector geometry was hand-calibrated against specific test footage and needs reverification against the real guest-house camera's actual mounting angle once it's reachable.
- Hazard-class reliability (Fire/Smoke/Accident/Animal) still depends entirely on underlying model/data quality, not pipeline logic — and the Car-class contamination triage (flagging old mislabeled Car images against a stock COCO model) was still open as of the last handoff I have, even though Truck itself is now a confirmed, integrated class.
- The model's domain-shift sensitivity to camera angle and stabilization is a demonstrated gap, not a hypothetical one — training data skewed ground-level/street-angle, the real deployment camera is handheld and elevated.
- `Obj_On_Road` and `Animal` are real trained classes with no analytics logic built around them at all — detected, but functionally inert in the pipeline as of the last documented state.
- Wrong-way and congestion zone geometry both still ship as placeholder polygons (a rough half-frame split, not traced lane boundaries) — real deployment needs a human tracing the actual boundaries against the deployed camera's real frame.

---

## Summary Table

| | |
|---|---|
| **Final overall mAP50** | 0.721 |
| **Final overall mAP50-95** | 0.452 (best at epoch 66; epoch 100 final was 0.449, essentially flat since ~66) |
| **Weakest original class (v1, pre-rebuild)** | Car — 0.485 mAP50 (label contamination with trucks/taxis) |
| **Weakest final-model class** | Bike — 0.506 mAP50 / 0.237 mAP50-95 (small-object-at-elevated-angle, not a data-volume problem) |
| **Strongest final-model class** | Obj_On_Road — 0.929 mAP50 / 0.655 mAP50-95 |
| **Final class taxonomy** | 10 classes: Bus, Car, Bike, Person, Animal, Fire, Smoke, Accident, Obj_On_Road, Truck (Truck approved by Ankit sir) |
| **Training resolution** | imgsz=960 (v2 dataset) |
| **Architecture** | ingestion → YOLOv8s+ByteTrack tracker → shared TrackManager → 4 analytics modules (stationary, wrong-way, hazard, congestion) → threaded event_recorder → OpenCV display |
| **Detector** | YOLOv8s (chosen for backbone capacity on organic/cluttered hazard classes) |
| **Tracker** | Raw Ultralytics + ByteTrack (chosen over wrapper libraries for debuggability) |
| **Dataset platform** | Roboflow (merged ~10 source datasets across v1/v2, mostly pre-annotated) |
| **Final dataset size** | 10,329 images, 65,741 annotations (6.4 objects/image avg.) |
| **Training platform** | Google Colab (T4, ~15GB VRAM) — chosen over local after a single-epoch timing test projected ~100 local hours vs. ~20 Colab hours for the full run; local RTX 4050 (6GB) kept for fast smoke tests / fallback |
| **Checkpoint strategy** | `patience=30`, `resume=True`, `save_period=5` |
| **Colab accounts used to reach 100 epochs** | ~5 account switches across the full saga (named: takcommunity99, yuvrajtak651, imuv) |
| **Deployment** | Desktop OpenCV app (`cv2.imshow`), file/live input toggle, keyboard-driven module toggles + live tuning panel |
| **Tech stack** | Python, YOLOv8 (Ultralytics), ByteTrack, OpenCV, Roboflow, Google Colab |
