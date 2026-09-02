#!/usr/bin/env python3
"""Performance modes: eco (energy saving) / balanced / turbo (full Apple Silicon power).

eco     — lightest model + processing paced to the source's real-time rate (the actual
          energy lever: a 12 fps camera needs ~12 inferences/s, not 130) + macOS
          UTILITY QoS so the OS prefers efficiency cores for the orchestration work.
balanced — the tuned defaults (yolov8s @ 512x960 ANE, unpaced).
turbo   — biggest accuracy (yolov8m @ 704x1280), unpaced, highest output bitrate,
          USER_INTERACTIVE QoS to hold performance cores.

Explicit CLI flags always win over mode defaults: the mode only fills in what the
user did not specify.
"""

import ctypes
import platform
from dataclasses import dataclass, field


@dataclass
class PerfProfile:
    name: str
    model: str | None = None          # default weights
    imgsz: str | None = None          # default canvas
    skip_frames: int = 0
    realtime: bool = False            # pace file sources to native fps
    bitrate_m: int = 8                # output H.264 bitrate (Mbps)
    display_fps: float = 30.0         # GUI refresh cap
    qos: str | None = None            # macOS QoS class


PROFILES = {
    "eco": PerfProfile(
        name="eco", model="yolov8n.pt", imgsz="512x960", skip_frames=0,
        realtime=True, bitrate_m=5, display_fps=15.0, qos="utility",
    ),
    "balanced": PerfProfile(
        name="balanced", model="yolov8s.pt", imgsz="512x960", skip_frames=0,
        realtime=False, bitrate_m=8, display_fps=30.0, qos=None,
    ),
    "turbo": PerfProfile(
        name="turbo", model="yolov8m.pt", imgsz="704x1280", skip_frames=0,
        realtime=False, bitrate_m=12, display_fps=30.0, qos="user_interactive",
    ),
}


def get_profile(mode: str) -> PerfProfile:
    return PROFILES.get(mode, PROFILES["balanced"])


_QOS_CLASSES = {
    "background": 0x09,
    "utility": 0x11,
    "user_initiated": 0x19,
    "user_interactive": 0x21,
}


def set_qos(qos: str | None) -> bool:
    """Set the calling thread's macOS QoS class (P/E-core scheduling hint).

    Threads spawned afterwards inherit it. Returns True on success.
    """
    if not qos or qos not in _QOS_CLASSES:
        return False
    if platform.system() != "Darwin":
        return False
    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        rc = libc.pthread_set_qos_class_self_np(_QOS_CLASSES[qos], 0)
        return rc == 0
    except Exception:
        return False
