#!/usr/bin/env python3
"""Smoke tests for capture + writer I/O layers. Run: python tests/test_io.py"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from traffic import VideoCaptureThreaded, VideoSink

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


print("\n[VideoCaptureThreaded]")
src = "test.mp4" if Path("test.mp4").exists() else "samples/video1.mp4"
cap = VideoCaptureThreaded(src)
check("opened", cap.width > 0 and cap.height > 0, f"{cap.width}x{cap.height}")
check("not live", not cap.is_live)
ok, frame = cap.read(timeout=10)
check("read frame", ok and frame is not None and frame.shape[0] == cap.height)
n = 1
while n < 30:
    ok, f = cap.read(timeout=10)
    if not ok:
        break
    n += 1
check("read 30 frames", n == 30, f"got {n}")
check("seek", cap.seek(5))
ok, f5 = cap.read(timeout=10)
check("read after seek", ok and f5 is not None)
cap.release()
print("  (capture released)")

print("\n[VideoSink — cv2 mp4v]")
with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "test_cv2.mp4"
    sink = VideoSink(out, 12.0, (320, 240), encoder="cv2")
    for i in range(24):
        img = np.full((240, 320, 3), (i * 10 % 255, 100, 100), dtype=np.uint8)
        sink.write(img)
    sink.close()
    check("cv2 output exists", out.exists() and out.stat().st_size > 1000)
    probe = cv2.VideoCapture(str(out))
    frames_out = 0
    while True:
        ok, _ = probe.read()
        if not ok:
            break
        frames_out += 1
    probe.release()
    check("cv2 output decodable, 24 frames", frames_out == 24, f"got {frames_out}")

print("\n[VideoSink — videotoolbox/auto]")
with tempfile.TemporaryDirectory() as td:
    out = Path(td) / "test_vt.mp4"
    try:
        sink = VideoSink(out, 12.0, (320, 240), encoder="auto")
        backend = sink.backend
        for i in range(24):
            sink.write(np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8))
        sink.close()
        check(f"auto encoder ({backend}) wrote file", out.exists() and out.stat().st_size > 1000)
        probe = cv2.VideoCapture(str(out))
        frames_out = 0
        while True:
            ok, _ = probe.read()
            if not ok:
                break
            frames_out += 1
        probe.release()
        check(f"{backend} output decodable", frames_out == 24, f"got {frames_out}")
    except Exception as e:
        print(f"  SKIP videotoolbox: {e}")

print(f"\n{'=' * 40}\nRESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
