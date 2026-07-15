"""
Centralized tunable thresholds for the traffic monitoring system.
Every module imports from here instead of hardcoding values —
so tuning behavior means editing one file, not hunting through five.
"""

DEBUG = False  # flip to True to enable targeted diagnostic prints

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

# ── Module II: Wrong-Way Detection ──────────────────────────────────
WRONG_WAY_DURATION_SEC = 5    
WRONG_WAY_SMOOTHING_WINDOW = 10 
WRONG_WAY_COSINE_THRESHOLD = 0.5  # Adjust based on your needs
AUTHORIZED_FLOW_VECTOR = (1.0, 0.0)

# ── Module III: Environmental Hazards ───────────────────────────────
HAZARD_CONFIDENCE_THRESHOLD = 0.25  
HAZARD_PERSISTENCE_SEC = 1        

# ── Module IV: Congestion / Density ─────────────────────────────────
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
VEHICLE_CLASSES = ['Car','Bike', 'Bus']   
HAZARD_CLASSES = ['Fire','Smoke','Accident']    