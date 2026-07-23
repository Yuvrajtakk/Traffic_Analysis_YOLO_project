# Real-Time YOLOv8 Traffic Monitoring System

A real-time traffic monitoring pipeline built with a custom-trained 10-class YOLOv8s detector and ByteTrack. The system runs four independent analytics modules to detect real-world traffic conditions in real-time, automatically saving an annotated still image and a video clip whenever an event fires.

## 🚀 What It Does

The system detects **10 object classes**: Bus, Car, Bike, Person, Animal, Fire, Smoke, Accident, Obj_On_Road, Truck.

Those detections feed into four analytics modules:

| Module | Triggers when... |
|---|---|
| **Stationary Vehicle** | A vehicle stays parked/stopped for 4+ seconds. |
| **Wrong-Way** | A vehicle moves against the expected traffic direction for 4+ seconds. |
| **Hazard** | Fire, Smoke, or Accident is confidently detected and persists. |
| **Congestion** | Vehicle count in a monitored ROI polygon exceeds capacity. |

Every trigger saves an annotated still image **and** a short before/after video clip automatically — no manual review needed to catch an event.

## ⚙️ How It Works (Architecture)

The system is built on a highly modular, multi-threaded architecture to ensure zero dropped frames during real-time RTSP ingestion. The pipeline flows as follows:

`Ingestion → YOLOv8s+ByteTrack → TrackManager → Analytics Modules → Event Recorder`

> [!TIP]
> **Explore the Architecture Flowcharts!** 
> 
> We have mapped out the entire codebase using interactive Mermaid diagrams. You can view how each piece interacts step-by-step, or zoom into the complete master flowchart!
> 👉 **[View the Architecture Flowcharts here (PROJECT_FLOWCHART.md)](PROJECT_FLOWCHART.md)** *(GitHub supports native zooming and panning!)*

## 🛠️ How to Run It

### Installation
```bash
pip install -r requirements.txt
python main.py
```
The script will prompt you: `Input source — type 'file' or 'live':`
- `file` → runs against a bundled test clip.
- `live` → connects to a live RTSP camera feed.

### Keyboard Controls
When the OpenCV window opens, click on it to ensure it has focus, then use these keys to control the system on the fly:

| Key | Effect |
|---|---|
| `d` | Show/hide the main video dashboard. |
| `t` | Show/hide the live tuning panel (developer tool to adjust detection thresholds on the fly without restarting). |
| `s` | Toggle **Stationary** detection on/off. |
| `w` | Toggle **Wrong-Way** detection on/off. |
| `h` | Toggle **Hazard** detection on/off. |
| `c` | Toggle **Congestion** detection on/off. |
| `p` | Print current tuning values to the console. |
| `q` | Quit the application. |

**Outputs:**
Saved events land in the `outputs/events/` folder. They are named explicitly by the module that fired, the class, the track ID, and the timestamp (e.g., `stationary_Car_id3_182052_716.jpg`).

---

## 🏆 Achievements

* **High Performance Model:** Achieved a final model accuracy of **mAP50: 0.721** and **mAP50-95: 0.452** at epoch 66/100.
* **Top Performing Classes:** `Obj_On_Road` reached 0.929 mAP50, and `Accident` reached 0.887 mAP50.
* **Complex Data Curation:** Successfully merged ~10 source datasets into a cohesive 10,329-image dataset containing 65,741 annotations.
* **Robust Multi-Threading:** The `VideoIngestion` thread runs asynchronously to keep the main YOLO inference loop fed with fresh frames without blocking, resulting in zero stale-frame bugs during heavy load.

## 🏗️ How It Was Built

The project began by deeply studying the YOLO architecture family (from v1 to v8) to understand grid-based detection, FPN merges, and IoU constraints. The **YOLOv8s** architecture was explicitly chosen for its backbone capacity to handle organic and cluttered hazard classes (Fire, Smoke, Accident) without sacrificing real-time speed.

**The Dataset Rebuild:**
Early on, the `Car` class suffered from heavy label contamination (trucks, taxis, and cars were merged into one label), resulting in a poor 0.485 mAP50. The entire dataset was rebuilt using VisDrone and Overhead Vehicle datasets, deliberately isolating a new `Truck` class to prevent confusion. This rebuild caused the `Car` class accuracy to skyrocket to 0.729 mAP50.

**The Training Process:**
Training was performed in Google Colab (T4 GPU, 15GB VRAM) at `imgsz=960` (up from 640) to better match the inference resolution of 1280. A strict checkpointing strategy (`patience=30`, `save_period=5`) was used to survive Colab's strict session limits over the 100-epoch run.

> [!NOTE]  
> If you are interested in the granular, step-by-step history of this project's development (including dead ends, specific bugs, and Roboflow metrics), please read the **[HISTORY.md](HISTORY.md)** file.

## ⚠️ Limitations

- **Small Objects at Angles:** `Bike` (0.506 mAP50) and `Person` (0.554 mAP50) struggle slightly. This is primarily a domain-shift limitation; small objects at elevated, overhead camera angles are significantly harder to detect than street-level imagery. 
- **Domain-Shift Sensitivity:** Testing revealed a generalization gap when moving from street-level training data to a handheld, elevated test camera.
- **Manual Zone Calibration:** Wrong-way and congestion ROI polygons currently require manual tracing/calibration against the deployment camera's real frame angle to function accurately.
