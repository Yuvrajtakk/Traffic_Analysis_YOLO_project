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
# Cosine values mean:
#   +1 = same direction as the expected flow
#    0 = sideways / unclear
#   -1 = opposite direction
# Wrong-way should only trigger when movement is clearly opposite, so
# the default is negative. A positive value like 0.487 is still mostly
# aligned and must NOT be treated as wrong-way.
WRONG_WAY_COSINE_THRESHOLD = -0.3

# Each zone: a polygon in NORMALIZED (0-1) image coordinates — same
# convention as CONGESTION_ROI_POLYGON_NORM below — plus the (x, y)
# direction traffic is EXPECTED to flow for any vehicle whose centroid
# falls inside that polygon.
#
# Traced against real test footage. The per-zone flow-vector direction
# is still an approximation for this specific test video's camera angle
# near the vanishing point: accurate enough for the stated task, not
# perfectly tuned, needs re-verification against the real deployed
# camera.
WRONG_WAY_ZONES = [{'polygon': [(0.482, 0.3181), (0.0039, 0.6222), (0.0023, 0.9917), (0.4375, 0.9972), (0.4875, 0.4083), (0.5047, 0.3458), (0.5047, 0.3194)], 'flow_vector': (0.0, -1.0)}, {'polygon': [(0.5219, 0.3278), (0.5133, 0.3583), (0.5227, 0.5458), (0.5953, 0.9958), (0.9984, 0.9944), (0.9938, 0.6875), (0.8187, 0.4625), (0.5625, 0.3333)], 'flow_vector': (0.0, 1.0)}]
# Used ONLY if a vehicle's centroid falls outside EVERY zone above
# (e.g. zones don't cover the full frame, or a vehicle briefly strays
# off-road). Deliberately falls back to a sane default rather than
# silently skipping wrong-way checks entirely for that vehicle.
WRONG_WAY_DEFAULT_FLOW_VECTOR = (1.0, 0.0)

# ── Module III: Environmental Hazards ───────────────────────────────
# Both live-tunable — see live_config.py. Startup defaults only.
HAZARD_CONFIDENCE_THRESHOLD = 0.25  
HAZARD_PERSISTENCE_SEC = 1        
HAZARD_FLICKER_WINDOW_FRAMES = 5
HAZARD_FLICKER_MIN_CONFIDENT_FRAMES = 3

# ── Module IV: Congestion / Density ─────────────────────────────────
# Live-tunable — see live_config.py. Startup default only.
CONGESTION_CAPACITY = 5  
# ROI polygon in NORMALIZED (0-1) image coordinates, traced against
# the current real test footage.
CONGESTION_ROI_POLYGON_NORM = [
    (0.4875, 0.3292), (0.0031, 0.675), (0.0031, 0.975), (0.9977, 0.9722),
    (0.9969, 0.6903), (0.6281, 0.4028), (0.5547, 0.3361), (0.4891, 0.3292),
] 

# ── Event Recording ──────────────────────────────────────────────────
PRE_EVENT_SEC = 2    
POST_EVENT_SEC = 2   
EVENT_COOLDOWN_SEC = 2  
HAZARD_EVENT_COOLDOWN_SEC = 30
# Hazard events use a longer cooldown than other modules. A
# misclassification can persist for minutes on a single object, and
# firing every 2 seconds during that time floods the events folder.
# 30s still catches a genuine hazard quickly while limiting repeat
# noise from a stuck false positive. This same cooldown is fine for
# Obj_On_Road and Animal because they use the same event path.

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

# Detected and drawn on-screen like any other class, but intentionally
# NOT wired into hazards.py's event/recording pipeline — Ankit sir's
# brief only specifies analytics tasks for Fire/Smoke/Accident
# (stationary/wrong-way/congestion cover the vehicle classes).
# Animal is a standard detection class (grouped with Bus/Car/Bike/
# Person in the brief); Object_on_road has no task mapped to it either.
DETECTION_ONLY_CLASSES = ['Animal', 'Obj_On_Road']

# All trained classes now have an explicit route:
# vehicles go through vehicle analytics, and Fire/Smoke/Accident/
# Obj_On_Road/Animal go through hazards.py's flicker + persistence +
# cooldown logic.
UNUSED_TRAINED_CLASSES = []
