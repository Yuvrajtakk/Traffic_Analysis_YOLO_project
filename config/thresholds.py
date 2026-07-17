"""
Centralized tunable thresholds for the traffic monitoring system.
Every module imports from here instead of hardcoding values —
so tuning behavior means editing one file, not hunting through five.

NOTE ON LIVE TUNING: the numeric thresholds also mirrored in
config/live_config.py (STATIONARY_*, WRONG_WAY_DURATION/COSINE,
HAZARD_*, CONGESTION_CAPACITY, YOLO_CONFIDENCE_THRESHOLD) are used
here only as STARTUP defaults. Once the program is running, the
live tuning panel (src/tuning_panel.py) overrides them in memory via
config/live_config.py's LiveConfig object — editing this file changes
tomorrow's default, not right now's live value. Everything else here
(taxonomy, zone geometry, dev flags, infra settings) is NOT
live-tunable by design — see each section's note.
"""

DEBUG = False  # flip to True to enable targeted diagnostic prints

# ── Module 0: YOLO Detection ─────────────────────────────────────────
# MODEL_IMGSZ MUST match the imgsz the weights were actually TRAINED
# at (v2_960_100epoch_real used imgsz=960). Running inference at a
# different resolution doesn't error — it just quietly hurts accuracy,
# since the model never saw objects at that scale during training.
MODEL_IMGSZ = 960
YOLO_CONFIDENCE_THRESHOLD = 0.4  # live-tunable — see live_config.py

# ── Video / Ingestion ──────────────────────────────────────────────
# Assumed frame rate ONLY used as a fallback when the source doesn't
# reliably report its own FPS (some RTSP streams lie about this).

DEFAULT_FPS = 25

RECONNECT_MAX_RETRIES = 5   
RECONNECT_DELAY_SEC = 1     

# ── Tracking (ByteTrack) ────────────────────────────────────────────
# How many frames an ID can stay "alive" while unmatched/occluded
# before ByteTrack (and our TrackManager) considers it gone for good.
# ByteTrack's own internal default is 30 — decide if you match it or override.
TRACK_BUFFER_FRAMES = 30   

# ── Module I: Stationary Vehicle Detection ─────────────────────────
# Both values below are live-tunable — see live_config.py. These are
# just the startup defaults.
STATIONARY_DURATION_SEC = 5   
# Raised from 5px after measuring real steady-state bbox jitter on
# handheld (non-tripod) footage: observed noise floor was ~4-9px,
# with occasional spikes to ~10px. 5px sat inside that noise floor,
# causing false re-fires. 15px gives safe margin above the measured
# ceiling while still being tight enough to catch genuine movement.
STATIONARY_PIXEL_THRESHOLD = 15
# A vehicle whose bbox area changes by more than 25% within the
# stationary window is considered moving (toward/away from camera),
# even if its centroid barely shifted in x/y. Catches depth-axis
# motion that pure pixel-distance can't see.
STATIONARY_AREA_CHANGE_THRESHOLD = 0.25
# ^ this is your epsilon — accounts for bounding-box area changes, not true stillness

# ── Module II: Wrong-Way Detection (multi-zone) ─────────────────────
# UPGRADED from a single global AUTHORIZED_FLOW_VECTOR (the old v1
# limitation) to a list of ZONES, each with its own expected flow
# direction — this is what makes two-lane / opposite-direction traffic
# and 4-way intersections work correctly, instead of flagging the
# "opposite lane" as wrong-way just because it points the other way.
#
# WRONG_WAY_DURATION_SEC and WRONG_WAY_COSINE_THRESHOLD are
# live-tunable — see live_config.py. These are just startup defaults.
WRONG_WAY_DURATION_SEC = 5
WRONG_WAY_SMOOTHING_WINDOW = 10
WRONG_WAY_COSINE_THRESHOLD = 0.5

# Each zone: a polygon in NORMALIZED (0-1) image coordinates — same
# convention as CONGESTION_ROI_POLYGON_NORM below — plus the (x, y)
# direction traffic is EXPECTED to flow for any vehicle whose centroid
# falls inside that polygon.
#
# PLACEHOLDER GEOMETRY — same status as CONGESTION_ROI_POLYGON_NORM:
# this is a rough vertical half-split (left half / right half of
# frame), not a manually traced lane boundary. Real deployment needs
# a human looking at an actual sample frame from the real camera and
# tracing the real lane edges as polygon points before these zones are
# meaningfully accurate. Two zones shown here as a working multi-lane
# EXAMPLE — add/remove/reshape zones to match the real intersection.
WRONG_WAY_ZONES = [
    {
        "polygon": [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)],
        "flow_vector": (1.0, 0.0),    # left half of frame: traffic expected moving RIGHT
    },
    {
        "polygon": [(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)],
        "flow_vector": (-1.0, 0.0),   # right half of frame: traffic expected moving LEFT
    },
]

# Used ONLY if a vehicle's centroid falls outside EVERY zone above
# (e.g. zones don't cover the full frame, or a vehicle briefly strays
# off-road). Deliberately falls back to a sane default rather than
# silently skipping wrong-way checks entirely for that vehicle.
WRONG_WAY_DEFAULT_FLOW_VECTOR = (1.0, 0.0)

# ── Module III: Environmental Hazards ───────────────────────────────
# Both live-tunable — see live_config.py. Startup defaults only.
HAZARD_CONFIDENCE_THRESHOLD = 0.25  
HAZARD_PERSISTENCE_SEC = 1        

# ── Module IV: Congestion / Density ─────────────────────────────────
# Live-tunable — see live_config.py. Startup default only.
CONGESTION_CAPACITY = 5  
# Default ROI polygon in NORMALIZED (0-1) image coordinates — a rough
# placeholder covering most of the frame. Real deployment requires
# manually tracing the actual road boundary for each specific camera
# (see congestion.py's KNOWN LIMITATION note).
CONGESTION_ROI_POLYGON_NORM = [
    (0.05, 0.35),
    (0.95, 0.35),
    (0.95, 0.95),
    (0.05, 0.95),
] 

# ── Event Recording ──────────────────────────────────────────────────
PRE_EVENT_SEC = 2    
POST_EVENT_SEC = 2   
EVENT_COOLDOWN_SEC = 2  
HAZARD_EVENT_COOLDOWN_SEC = 30
# Hazard events (Fire/Smoke/Accident) use a longer cooldown than other
# modules. A misclassification can persist for minutes on a single
# object, and firing every 2 seconds during that time floods the
# events folder. 30s still catches a genuine hazard quickly while
# limiting repeat noise from a stuck false positive.

# ── Vehicle class taxonomy (used across multiple modules) ──────────
# NOT live-tunable — taxonomy, not a numeric threshold.
#
# FIXED: 'Truck' was previously missing here even though it's a real,
# trained class in the model (10-class dataset: Car, Truck, Bus, Fire,
# Smoke, Bike, Animal, Person, Accident, Obj_On_Road). Every truck in
# frame was silently invisible to stationary/wrong-way/congestion
# detection until now — those modules only ever check membership in
# this list, so a missing entry means total silent exclusion, no error.
VEHICLE_CLASSES = ['Car', 'Bike', 'Bus', 'Truck']
HAZARD_CLASSES = ['Fire', 'Smoke', 'Accident']

# Not currently consumed by any module yet — 'Obj_On_Road' (debris/
# object blocking the road) and 'Animal' are real trained classes with
# no analytics logic built around them. Worth a deliberate decision in
# a future phase: fold Obj_On_Road into HAZARD_CLASSES, or give it its
# own module? Flagging here rather than silently ignoring it.
UNUSED_TRAINED_CLASSES = ['Obj_On_Road', 'Animal']
