"""
Centralized tunable thresholds for the traffic monitoring system.
Every module imports from here instead of hardcoding values —
so tuning behavior means editing one file, not hunting through five.
"""

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
STATIONARY_PIXEL_THRESHOLD = 5  
# ^ this is your epsilon — accounts for bounding-box jitter, not true stillness

# ── Module II: Wrong-Way Detection ──────────────────────────────────
WRONG_WAY_DURATION_SEC = 5    
WRONG_WAY_SMOOTHING_WINDOW = 10 

# ── Module III: Environmental Hazards ───────────────────────────────
HAZARD_CONFIDENCE_THRESHOLD = 0.25  
HAZARD_PERSISTENCE_SEC = 1        

# ── Module IV: Congestion / Density ─────────────────────────────────
CONGESTION_CAPACITY = 5   

# ── Event Recording ──────────────────────────────────────────────────
PRE_EVENT_SEC = 2    
POST_EVENT_SEC = 2   
EVENT_COOLDOWN_SEC = 2  

# ── Vehicle class taxonomy (used across multiple modules) ──────────
VEHICLE_CLASSES = ['car','bike', 'bus']   
HAZARD_CLASSES = ['fire','smoke','accident']    