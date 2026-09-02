#!/usr/bin/env python3
"""
run_multistream.py — Multi-stream traffic surveillance engine for Apple Silicon.

N video feeds (files, webcams, RTSP URLs) processed round-robin through ONE compiled
CoreML/ANE model with an independent ByteTrack state per stream. Live sources run
with a latest-frame policy so slow feeds never stall fast ones. Output is a tiled
grid with per-camera HUD, tripwires and counters.

Examples:
  python run_multistream.py                              # defaults: test.mp4 + samples
  python run_multistream.py cam1.mp4 cam2.mp4 rtsp://ip/cam
  python run_multistream.py --model yolov8n.pt --imgsz 512x960 --cols 2
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from traffic import (DirectionalLine, InferenceEngine, MotionHeatmap, SpeedEstimator,
                     SuppressNested, TrackBook, VideoCaptureThreaded,
                     get_profile, set_qos)
from traffic.annotate import CLASS_META, draw_box, draw_line, draw_trail

SUPPORTED = [0, 1, 2, 3, 5, 7]
VEHICLE_CLASSES = ["car", "bus", "truck"]
TWO_WHEELER_CLASSES = ["motorcycle", "bicycle"]  # e-bikes surface as either class


class StreamState:
    def __init__(self, idx, source, engine, args):
        self.idx = idx
        try:
            self.reader = VideoCaptureThreaded(source, name=Path(str(source)).name)
        except RuntimeError as e:
            print(f"[ERROR] Stream {idx}: {e}")
            raise
        self.name = self.reader.name
        self.w, self.h = self.reader.width, self.reader.height
        self.book = TrackBook(vote_window=args.vote_window)
        self.sup = SuppressNested(args.containment)
        sw, sh = self.w / 1280.0, self.h / 674.0
        self.line = DirectionalLine((int(80 * sw), int(480 * sh)),
                                    (int(760 * sw), int(480 * sh)),
                                    "STOP-LINE", classes=VEHICLE_CLASSES)
        self.heatmap = MotionHeatmap(self.w, self.h) if args.heatmap else None
        self.speed = SpeedEstimator(mpp=args.mpp) if args.mpp else None
        self.fps = 0.0
        self.last_canvas = None
        self.frames = 0
        self.t_prev = time.perf_counter()

    def step(self, engine, args):
        ok, frame = self.reader.read(timeout=0.05)
        if not ok or frame is None:
            return self.last_canvas
        dets = self.sup(engine.track(frame, stream_id=self.idx))
        in_counts = {}
        for d in dets:
            meta = CLASS_META.get(d.cls_id)
            if meta is None:
                continue
            thr = min(args.conf, meta["min_conf"]) if "min_conf" in meta else args.conf
            if d.conf < thr:
                continue
            cls_name = meta["name"]
            in_counts[cls_name] = in_counts.get(cls_name, 0) + 1
            _, prev_pt, pt = self.book.update_track(d, cls_name)
            ev = self.line.update(d.track_id, prev_pt, pt, cls_name)
            if ev:
                print(f"  [CAM{self.idx}] #{d.track_id} {cls_name} {ev['direction'].upper()} "
                      f"(total {self.line.total_in + self.line.total_out})", flush=True)
            if self.speed is not None:
                self.speed.update(d.track_id, d.cx, d.cy)
        if self.heatmap is not None:
            self.heatmap.update(dets)
            frame = self.heatmap.overlay(frame, 0.4)
        draw_line(frame, self.line)
        for d in dets:
            meta = CLASS_META.get(d.cls_id)
            if meta is None:
                continue
            thr = min(args.conf, meta["min_conf"]) if "min_conf" in meta else args.conf
            if d.conf < thr:
                continue
            draw_trail(frame, self.book.trails[d.track_id], meta["color"])
            kmh = f" {self.speed.latest[d.track_id][1]:.0f}km/h" \
                if self.speed and self.speed.latest.get(d.track_id, (None, None))[1] else ""
            draw_box(frame, d, meta["name"], meta["color"],
                     f"CAM{self.idx} #{d.track_id} {meta['name']}{kmh}")

        now = time.perf_counter()
        inst = 1.0 / max(now - self.t_prev, 1e-6)
        self.t_prev = now
        self.fps = 0.9 * self.fps + 0.1 * inst if self.frames > 2 else inst
        self.frames += 1
        banner = (f"CAM{self.idx} {self.name} | {self.fps:4.1f} FPS | "
                  f"P:{in_counts.get('person', 0)} "
                  f"V:{sum(in_counts.get(k, 0) for k in VEHICLE_CLASSES)} "
                  f"TW:{sum(in_counts.get(k, 0) for k in TWO_WHEELER_CLASSES)} "
                  f"| crossed: {self.line.total_in + self.line.total_out}")
        cv2.rectangle(frame, (0, 0), (self.w, 30), (18, 20, 22), -1)
        cv2.putText(frame, banner, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 230, 230), 1, cv2.LINE_AA)
        self.last_canvas = frame
        return frame


def tile(canvases, cols, tw=640, th=360):
    cells = []
    for c in canvases:
        if c is None:
            c = np.zeros((th, tw, 3), dtype=np.uint8)
        cells.append(cv2.resize(c, (tw, th)))
    rows = []
    for i in range(0, len(cells), cols):
        row = cells[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros((th, tw, 3), dtype=np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def main():
    ap = argparse.ArgumentParser(description="Multi-stream surveillance (Apple Silicon)")
    ap.add_argument("sources", nargs="*", default=None, help="Files / webcam ids / rtsp urls")
    ap.add_argument("--mode", default="balanced", choices=["eco", "balanced", "turbo"],
                    help="eco: n 模型+实时节奏 | turbo: m 模型+大画布")
    ap.add_argument("--stream1", default=None, help="(compat) first stream")
    ap.add_argument("--stream2", default=None, help="(compat) second stream")
    ap.add_argument("--model", default=None, help="YOLO weights; mode default if omitted")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "coreml", "mps", "cuda", "cpu"])
    ap.add_argument("--imgsz", default=None)
    ap.add_argument("--conf", type=float, default=0.45)
    ap.add_argument("--cols", type=int, default=2, help="Grid columns")
    ap.add_argument("--heatmap", action="store_true")
    ap.add_argument("--mpp", type=float, default=None, help="Meters/pixel for km/h speeds")
    ap.add_argument("--vote-window", type=int, default=15)
    ap.add_argument("--containment", type=float, default=0.70)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--no-gui", action="store_true", help="Headless: no preview window")
    ap.add_argument("--realtime", action="store_true", default=None,
                    help="Pace file sources at their native FPS (eco mode default)")
    ap.add_argument("--save", default=None, help="Save grid recording to this mp4")
    args = ap.parse_args()

    profile = get_profile(args.mode)
    if args.model is None:
        args.model = profile.model
    if args.imgsz is None:
        args.imgsz = profile.imgsz
    if args.realtime is None:
        args.realtime = profile.realtime
    set_qos(profile.qos)

    sources = args.sources or []
    if args.stream1:
        sources.append(args.stream1)
    if args.stream2:
        sources.append(args.stream2)
    if not sources:
        sources = ["test.mp4", "samples/video1.mp4"]
    sources = [int(s) if s.isdigit() else s for s in sources]

    print("=" * 78)
    print(f" Multi-Stream Surveillance — {len(sources)} streams [{args.mode.upper()} MODE]"
          f" | model {args.model}")
    for i, s in enumerate(sources):
        print(f"   CAM{i}: {s}")
    print("=" * 78)

    imgsz = ([int(v) for v in args.imgsz.lower().split("x")]
             if "x" in args.imgsz.lower() else [int(args.imgsz)] * 2)
    engine = InferenceEngine(weights=args.model, imgsz=imgsz, backend=args.backend,
                             classes=SUPPORTED, tracker_cfg="custom_bytetrack.yaml",
                             display_conf=args.conf)
    print(f"[ENGINE] {engine.describe()}\n")

    streams = []
    for i, s in enumerate(sources):
        try:
            streams.append(StreamState(i, s, engine, args))
        except RuntimeError:
            if not streams:
                return 1

    writer = None
    if args.save:
        from traffic import VideoSink
        writer = VideoSink(args.save, 25.0,
                           (640 * args.cols, 360 * ((len(streams) + args.cols - 1) // args.cols)),
                           bitrate_m=profile.bitrate_m)

    win = "Multi-Stream Surveillance"
    gui = not args.no_gui
    if gui:
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(win, 1280, 360 * ((len(streams) + args.cols - 1) // args.cols))
        except Exception:
            gui = False

    t0 = time.perf_counter()
    frame_loop = 0
    last_show = 0.0
    try:
        while True:
            canvases = [st.step(engine, args) for st in streams]
            if args.realtime:
                # Pace the loop to the slowest live requirement: each file source
                # should advance no faster than its native fps.
                target_period = min((1.0 / st.reader.orig_fps for st in streams
                                     if not st.reader.is_live), default=0.0)
                if target_period > 0:
                    elapsed_now = time.perf_counter() - t0
                    due = frame_loop * target_period
                    if elapsed_now < due:
                        time.sleep(due - elapsed_now)
            grid = tile(canvases, args.cols)
            total_fps = sum(st.fps for st in streams)
            elapsed = time.perf_counter() - t0
            cv2.putText(grid, f"TOTAL {total_fps:5.1f} FPS [{args.mode.upper()}] | {engine.describe()} | "
                              f"q quit | s snapshot", (12, grid.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1, cv2.LINE_AA)
            if writer:
                writer.write(grid)
            if gui:
                now_s = time.perf_counter()
                if now_s - last_show >= 1 / 30.0:  # display decoupled from inference
                    last_show = now_s
                    cv2.imshow(win, grid)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        break
                    elif key == ord('s'):
                        p = f"output/multistream_{int(time.time())}.png"
                        cv2.imwrite(p, grid)
                        print(f"  [SNAP] {p}")
            frame_loop += 1
            if args.max_seconds and elapsed >= args.max_seconds:
                break
            if frame_loop % 100 == 0:
                per = "  ".join(f"CAM{st.idx}:{st.fps:4.1f}" for st in streams)
                print(f"  [{elapsed:6.1f}s] {per}  | total {total_fps:5.1f} FPS", flush=True)
    finally:
        for st in streams:
            st.reader.release()
        if writer:
            writer.close()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - t0
    print(f"\n{'=' * 78}")
    print(f" Processed {elapsed:.1f}s | {engine.describe()}")
    for st in streams:
        print(f"  CAM{st.idx} {st.name:<24s} {st.frames:5d} frames  avg {st.frames / max(elapsed, 1e-6):5.1f} FPS"
              f"  | stop-line crossings: {st.line.total_in + st.line.total_out}")
    tot = sum(st.frames for st in streams)
    print(f"  Aggregate throughput: {tot / max(elapsed, 1e-6):.1f} frames/s")
    print(f"{'=' * 78}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
