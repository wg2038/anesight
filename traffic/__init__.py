"""traffic — Apple Silicon optimized video analytics package.

Modules:
    engine    — Inference backend abstraction (CoreML/ANE > MPS > CUDA > CPU) + ByteTrack
    capture   — Threaded video capture with latest-frame policy for live sources
    writer    — Video output with VideoToolbox H.264 hardware encoding
    analytics — Directional tripwires, polygon zones, heatmaps, speed estimation
    annotate  — HUD and overlay rendering
    logio     — Session logging (CSV events + JSON summary) and snapshots
"""

from traffic.engine import Detection, InferenceEngine, Backend, select_backend
from traffic.capture import VideoCaptureThreaded
from traffic.writer import VideoSink
from traffic.analytics import (DirectionalLine, PolygonZone, MotionHeatmap,
                               SpeedEstimator, TrackBook, SuppressNested)
from traffic.logio import SessionLogger
from traffic.perf import PerfProfile, get_profile, set_qos
from traffic.alerts import Alert, AlertBus
from traffic.parking import ParkingSlot, SlotManager, parse_slot, parse_slot_grid
from traffic.factory import (ProximityMonitor, RestrictedZone, is_armed,
                             parse_armed_hours)

__all__ = [
    "Detection", "InferenceEngine", "Backend", "select_backend",
    "VideoCaptureThreaded", "VideoSink",
    "DirectionalLine", "PolygonZone", "MotionHeatmap", "SpeedEstimator",
    "TrackBook", "SuppressNested", "SessionLogger",
    "PerfProfile", "get_profile", "set_qos",
    "Alert", "AlertBus",
    "ParkingSlot", "SlotManager", "parse_slot", "parse_slot_grid",
    "ProximityMonitor", "RestrictedZone", "is_armed", "parse_armed_hours",
]
