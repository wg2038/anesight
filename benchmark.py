#!/usr/bin/env python3
"""
benchmark.py — Apple Silicon benchmarking suite for the YOLO traffic pipeline.

  --bench detect  Ultralytics predict across models/backends (raw detector speed)
  --bench track   The real pipeline engine (InferenceEngine + ByteTrack) per backend

Examples:
  python benchmark.py                          # track bench, yolov8s, all backends
  python benchmark.py --bench detect --imgsz 512x960
  python benchmark.py --track-model yolov8n.pt --backends coreml,mps
"""

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2


def load_frames(path: str, n: int):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break
        frames.append(frame)
    cap.release()
    return frames


def parse_imgsz(s: str):
    if "x" in str(s).lower():
        h, w = str(s).lower().split("x")
        return [int(h), int(w)]
    return [int(s), int(s)]


def chip_info():
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                          capture_output=True, text=True).stdout.strip()
    ram = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True).stdout.strip()) / 2**30
    return chip, f"{ram:.0f}GB", platform.mac_ver()[0]


def bench_detect(models, frames, imgsz, warmup=8):
    from ultralytics import YOLO
    rows = []
    for m in models:
        if not Path(m).exists():
            print(f"  [SKIP] {m} not found")
            continue
        model = YOLO(m)
        for i in range(warmup):
            model.predict(frames[i % len(frames)], imgsz=imgsz, conf=0.45,
                          classes=[0, 1, 2, 3, 5, 7], verbose=False)
        t0 = time.perf_counter()
        n = len(frames)
        for i in range(n):
            model.predict(frames[i % n], imgsz=imgsz, conf=0.45,
                          classes=[0, 1, 2, 3, 5, 7], verbose=False)
        dt = time.perf_counter() - t0
        rows.append((m, dt / n * 1000, n / dt))
    return rows


def bench_track(backends, frames, imgsz, weights, warmup=6):
    from traffic import InferenceEngine
    rows = []
    for be in backends:
        try:
            engine = InferenceEngine(weights=weights, imgsz=imgsz, backend=be,
                                     classes=[0, 1, 2, 3, 5, 7],
                                     tracker_cfg="custom_bytetrack.yaml")
            for i in range(warmup):
                engine.track(frames[i % len(frames)])
            t0 = time.perf_counter()
            n = len(frames)
            for i in range(n):
                engine.track(frames[i % n])
            dt = time.perf_counter() - t0
            rows.append((f"{weights} @ {engine.describe()}", dt / n * 1000, n / dt))
            engine._trackers.clear()
        except Exception as e:
            print(f"  [FAIL] {be}: {type(e).__name__}: {str(e)[:100]}")
    return rows


def main():
    ap = argparse.ArgumentParser(description="Apple Silicon YOLO benchmark")
    ap.add_argument("--bench", choices=["detect", "track"], default="track")
    ap.add_argument("--models", nargs="+", default=["yolov8s.pt"],
                    help="models for --bench detect")
    ap.add_argument("--track-model", default="yolov8s.pt")
    ap.add_argument("--backends", nargs="+", default=["coreml", "mps", "cpu"])
    ap.add_argument("--imgsz", type=str, default="512x960")
    ap.add_argument("--frames", type=int, default=80)
    ap.add_argument("--source", type=str, default="test.mp4")
    ap.add_argument("--json", type=str, default=None)
    args = ap.parse_args()
    imgsz = parse_imgsz(args.imgsz)

    chip, ram, macos = chip_info()
    print(f"\n{'=' * 78}")
    print(f" Apple Silicon Benchmark | {chip} | {ram} | macOS {macos}")
    print(f" bench={args.bench} | imgsz={imgsz} | frames={args.frames}")
    print(f"{'=' * 78}\n")

    frames = load_frames(args.source, args.frames)
    if not frames:
        print(f"[ERROR] No frames from {args.source}")
        return 1
    print(f"Source: {args.source} {frames[0].shape[1]}x{frames[0].shape[0]} | {len(frames)} frames\n")

    if args.bench == "detect":
        rows = bench_detect(args.models, frames, imgsz)
    else:
        rows = bench_track(args.backends, frames, imgsz, args.track_model)

    for label, ms, fps in rows:
        print(f"  {label:<52s} {ms:7.1f} ms/frame  {fps:7.1f} FPS")
    if rows:
        best = max(rows, key=lambda r: r[2])
        print(f"\n  BEST: {best[0]}  @ {best[2]:.1f} FPS ({best[1]:.1f} ms/frame)")

    if args.json:
        Path(args.json).write_text(json.dumps({
            "chip": chip, "imgsz": imgsz, "bench": args.bench,
            "results": [{"config": l, "ms": round(ms, 2), "fps": round(f, 2)}
                        for l, ms, f in rows]}, indent=2))
        print(f"  Saved: {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
