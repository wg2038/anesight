#!/usr/bin/env python3
"""
run_detector.py — Apple Silicon optimized traffic analytics pipeline.

Architecture (M3-class hardware):
  * Inference on the Neural Engine (CoreML fp16, fused NMS, rectangular canvas) with
    automatic export + caching; MPS fp16 / CUDA / CPU fallbacks selected automatically.
  * Manual per-stream ByteTrack (independent state per source, shares one model).
  * Threaded capture (latest-frame policy for live sources, queued for files).
  * VideoToolbox H.264 hardware encoding of annotated output.
  * Direction-aware tripwire counting, polygon zones, motion heatmap, speed estimation,
    CSV event log + JSON summary, event snapshots.

Interactive keys: q skip | Esc quit | space pause | . step | t trails | h heatmap |
                  z zones | l lines | s snapshot

Examples:
  python run_detector.py                          # test.mp4 with legacy scene lines
  python run_detector.py --no-gui                 # headless batch
  python run_detector.py cam.mp4 --line "100,400,900,400:gate" --heatmap --speed --mpp 0.02
  python run_detector.py video.mp4 --model yolov8m.pt --backend coreml
"""

import argparse
import sys
import time
from pathlib import Path

import cv2

from traffic import (AlertBus, DirectionalLine, InferenceEngine, MotionHeatmap,
                     PolygonZone, SessionLogger, SpeedEstimator, SuppressNested,
                     TrackBook, VideoCaptureThreaded, VideoSink, get_profile,
                     select_backend, set_qos)
from traffic.annotate import (CLASS_META, draw_alert_banner, draw_box, draw_help,
                              draw_hud, draw_line, draw_restricted, draw_slot,
                              draw_trail, draw_zone)
from traffic.engine import Detection
from traffic.factory import (ProximityMonitor, RestrictedZone, is_armed,
                             parse_armed_hours)
from traffic.parking import SlotManager, parse_slot, parse_slot_grid

VEHICLE_GROUPS = {"car": "vehicle", "bus": "vehicle", "truck": "vehicle",
                  "motorcycle": "twowheeler", "bicycle": "twowheeler", "person": "person"}
SUPPORTED = [0, 1, 2, 3, 5, 7]
VEHICLE_CLASSES = ["car", "bus", "truck"]          # motor vehicles (tripwire default)
TWO_WHEELER_CLASSES = ["motorcycle", "bicycle"]    # e-bikes surface as either COCO class
ANALYTICS_VEHICLE_CLASSES = VEHICLE_CLASSES + ["motorcycle"]

# Legacy scene lines (normalized to a 1280x674 canvas), preserved for test.mp4 parity.
LEGACY_LINES = [
    ((80, 480), (760, 480), "VEHICLE-LINE", ANALYTICS_VEHICLE_CLASSES),
    ((760, 440), (1150, 510), "CROSSWALK", ["person"]),
]


def parse_line(spec: str):
    """'x1,y1,x2,y2[:name[:cls1,cls2]]' -> ((x1,y1),(x2,y2),name,classes|None)"""
    parts = spec.split(":")
    coords = [float(v) for v in parts[0].split(",")]
    if len(coords) != 4:
        raise ValueError(f"--line needs x1,y1,x2,y2 — got '{spec}'")
    name = parts[1] if len(parts) > 1 and parts[1] else "line"
    classes = [c.strip() for c in parts[2].split(",")] if len(parts) > 2 and parts[2] else None
    return ((coords[0], coords[1]), (coords[2], coords[3]), name, classes)


def parse_zone(spec: str):
    """'x1,y1,...,xN,yN[:name]' (N>=3 points, or 4 numbers = rectangle) -> (polygon, name)"""
    parts = spec.split(":")
    coords = [float(v) for v in parts[0].split(",")]
    name = parts[1] if len(parts) > 1 and parts[1] else "zone"
    if len(coords) == 4:  # rectangle corners x1,y1,x2,y2
        x1, y1, x2, y2 = coords
        poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    else:
        if len(coords) < 6 or len(coords) % 2 != 0:
            raise ValueError(f"--zone needs 3+ point pairs (or 4 rect coords) — got '{spec}'")
        poly = list(zip(coords[0::2], coords[1::2]))
    return poly, name


def scale_lines(lines, w, h):
    sw, sh = w / 1280.0, h / 674.0
    return [((int(x1 * sw), int(y1 * sh)), (int(x2 * sw), int(y2 * sh)), name, classes)
            for (x1, y1), (x2, y2), name, classes in lines]


class GUIWindow:
    """Thin wrapper so headless mode needs no branches in the hot loop."""

    def __init__(self, title, w, h):
        self.active = True
        self.title = title
        try:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, min(w, 1280), min(h, 720))
        except Exception:
            self.active = False

    def show(self, canvas):
        if not self.active:
            return None
        try:
            cv2.imshow(self.title, canvas)
            return cv2.waitKey(1) & 0xFF
        except Exception:
            self.active = False
            return None

    def show_paused(self, canvas):
        if not self.active:
            return None
        try:
            cv2.imshow(self.title, canvas)
            return cv2.waitKey(30) & 0xFF
        except Exception:
            self.active = False
            return None

    def close(self):
        if self.active:
            try:
                cv2.destroyWindow(self.title)
            except Exception:
                pass


def parse_restricted(spec: str):
    """'x1,y1,...[:name[:trigger_cls1,cls2]]' -> (polygon, name, trigger_classes)"""
    parts = spec.split(":")
    coords = [float(v) for v in parts[0].split(",")]
    if len(coords) == 4:
        x1, y1, x2, y2 = coords
        poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    else:
        if len(coords) < 6 or len(coords) % 2:
            raise ValueError(f"--restricted needs 4 rect coords or 3+ polygon pairs — got '{spec}'")
        poly = list(zip(coords[0::2], coords[1::2]))
    name = parts[1] if len(parts) > 1 and parts[1] else "restricted"
    trigger = ([c.strip() for c in parts[2].split(",")] if len(parts) > 2 and parts[2]
               else ["person"])
    return poly, name, trigger


def resolve_scenario(args) -> str:
    """Explicit --scenario wins; otherwise infer from what the user configured."""
    if args.scenario:
        return args.scenario
    if args.slots or args.slot_grid:
        return "parking"
    if args.restricted or args.proximity_px > 0:
        return "factory"
    return "road"


def build_class_meta(engine) -> dict:
    """CLASS_META plus generic entries for custom-model classes (PPE models etc.)."""
    full = dict(CLASS_META)
    palette = [(90, 200, 250), (250, 180, 90), (180, 250, 120), (120, 160, 250),
               (250, 120, 180), (160, 250, 250), (220, 220, 120), (200, 160, 250)]
    for cid, name in sorted(engine.class_names.items()):
        if cid in full:
            continue
        full[cid] = {"name": name, "zh": name, "color": palette[cid % len(palette)]}
    return full


def process_source(source, engine, args, logger, gui=None, profile=None):
    """Run the full pipeline on one source. Returns a summary dict."""
    profile = profile or get_profile("balanced")
    scenario = resolve_scenario(args)
    try:
        reader = VideoCaptureThreaded(source, name=Path(str(source)).name)
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        return None
    w, h = reader.width, reader.height
    fps_src = reader.orig_fps
    print(f"\n{'=' * 78}")
    print(f" Source : {reader.name}  ({w}x{h} @ {fps_src:.2f} fps, "
          f"{'LIVE' if reader.is_live else f'{reader.total_frames} frames'})")
    print(f" Engine : {engine.describe()} | model {Path(engine.weights).name} "
          f"| infer avg {engine.infer_ms:.1f} ms")
    print(f"{'=' * 78}\n", flush=True)

    gui = gui or GUIWindow(f"YOLO Traffic Analytics — {reader.name}", w, h)
    show_windows = gui.active and not args.no_gui

    # Output writer
    sink = None
    if not args.no_save:
        out_path = Path(args.outdir) / f"annotated_{Path(reader.name).stem}.mp4"
        try:
            sink = VideoSink(out_path, fps_src, (w, h), encoder=args.encoder,
                             bitrate_m=profile.bitrate_m)
            print(f"[OUT] {sink.backend} encoder -> {out_path}")
        except Exception as e:
            print(f"[WARN] Video output disabled: {e}")

    # Analytics state
    book = TrackBook(vote_window=args.vote_window, confirm_hits=args.confirm_hits)
    nesting = SuppressNested(args.containment)
    heatmap = MotionHeatmap(w, h)
    speed = SpeedEstimator(mpp=args.mpp) if (args.mpp or args.speed) else None
    full_meta = build_class_meta(engine)

    lines = []
    if args.lines:
        for (p1, p2, name, classes) in args.lines:
            lines.append(DirectionalLine(p1, p2, name, classes=classes))
    elif not args.no_default_lines and scenario == "road":
        for (p1, p2, name, classes) in scale_lines(LEGACY_LINES, w, h):
            lines.append(DirectionalLine(p1, p2, name, classes=classes))
    zones = [PolygonZone(poly, name) for poly, name in (args.zones or [])]

    # ---- scenario managers
    alert_bus = AlertBus(cooldown_s=args.alert_cooldown, webhook=args.webhook, logger=logger)
    armed_window = args.armed_hours  # pre-parsed tuple|None

    slots_mgr = None
    if scenario == "parking":
        slots = list(args.slots or []) + list(args.slot_grid or [])
        if slots:
            slots_mgr = SlotManager(slots, stability=args.parking_stability)

    restricted_zones = [RestrictedZone(poly, name, trigger_classes=trigger,
                                       loiter_sec=args.loiter_sec)
                        for poly, name, trigger in (args.restricted or [])]

    proximity = ProximityMonitor(threshold_px=args.proximity_px) if args.proximity_px > 0 else None

    frame_idx = 0
    paused = False
    show_trails = True
    show_heat = args.heatmap
    show_zones = True
    show_lines = True
    skip = args.skip_frames
    cached_draws = []
    live_failures = 0
    flash = {id(ln): 0.0 for ln in lines}
    fps_avg, infer_pct_acc = 0.0, 0.0
    t_start = time.perf_counter()
    t_prev = t_start
    last_show = 0.0
    user_quit = False

    try:
        while True:
            ok, frame = reader.read()
            if not ok or frame is None:
                if reader.is_live and live_failures < 150:
                    # Live streams hiccup (wifi drop, encoder restart) — keep waiting.
                    live_failures += 1
                    time.sleep(0.1)
                    continue
                break
            live_failures = 0
            frame_idx += 1

            if args.realtime and not reader.is_live and fps_src > 0:
                # Eco pacing: schedule one inference per source frame real-time slot
                # instead of burning through the file as fast as possible.
                due = t_start + frame_idx / fps_src
                ahead = due - time.perf_counter()
                if ahead > 0:
                    time.sleep(min(ahead, 1.0))

            t_loop = time.perf_counter()
            if skip and frame_idx % (skip + 1) != 1:
                dets = None  # legacy fast mode: reuse previous detections
            else:
                dets = engine.track(frame)
                dets = nesting(dets)

            # Per-class confidence + stabilization + analytics
            in_counts, active_ids = {}, set()
            events = []
            canvas_draws = []
            if dets is not None:
                for d in dets:
                    meta = full_meta.get(d.cls_id)
                    if meta is None:
                        continue
                    # Per-class confidence: two-wheelers detect at lower confidence
                    # (small, distant) — floor their threshold via CLASS_META.min_conf.
                    thr = min(args.conf, meta["min_conf"]) if "min_conf" in meta else args.conf
                    if d.conf < thr:
                        continue
                    cls_name = meta["name"]
                    stable_cls = full_meta[book.stabilize_class(d.track_id, d.cls_id)]["name"] \
                        if d.track_id >= 0 else cls_name
                    in_counts[stable_cls] = in_counts.get(stable_cls, 0) + 1
                    active_ids.add(d.track_id)

                    newly_conf, prev_pt, pt = book.update_track(d, stable_cls)
                    if newly_conf and args.verbose:
                        print(f"  [NEW] #{d.track_id} {stable_cls} confirmed", flush=True)

                    for ln in lines:
                        ev = ln.update(d.track_id, prev_pt, pt, stable_cls)
                        if ev:
                            ev["event"] = "crossing"
                            events.append(ev)
                            flash[id(ln)] = 1.0
                            print(f"  >>> [{ln.name}] #{d.track_id} {stable_cls} "
                                  f"{ev['direction'].upper()} (in total: {ln.total_in})", flush=True)

                    kmh = None
                    if speed is not None:
                        _, kmh = speed.update(d.track_id, d.cx, d.cy)
                    canvas_draws.append((d, meta, stable_cls, kmh))
                cached_draws = canvas_draws

                zone_pts = [(cls_name, d.track_id, (d.cx, d.cy))
                            for d, meta, cls_name, kmh in canvas_draws]
                for z in zones:
                    z.update(zone_pts)

                if show_heat:
                    heatmap.update(dets)
            else:
                canvas_draws = cached_draws

            # ---- scenario analytics (road / parking / factory)
            t_mono = time.monotonic()
            if scenario == "parking" and slots_mgr is not None:
                pairs = [(d, cls_name) for d, meta, cls_name, kmh in canvas_draws]
                for sev in slots_mgr.update(pairs, t_mono):
                    key = f"{sev['event']}:{sev['slot']}:{sev['track_id']}"
                    msg = (f"slot {sev['slot']} OCCUPIED by {sev['cls']} #{sev['track_id']}"
                           if sev["event"] == "slot_occupied" else
                           f"slot {sev['slot']} FREED (was {sev['cls']} #{sev['track_id']}, "
                           f"parked {sev['dwell_s']:.0f}s)")
                    alert_bus.emit(sev["event"], key, msg, level="info",
                                   frame=frame_idx, meta=sev, t=t_mono)
            if scenario == "factory" and is_armed(armed_window):
                dets_pts = [(cls_name, d.track_id, (d.cx, d.cy), d.xyxy)
                            for d, meta, cls_name, kmh in canvas_draws]
                for rz in restricted_zones:
                    for atype, key, msg, level in rz.update(dets_pts, t_mono):
                        alert_bus.emit(atype, key, msg, level=level, frame=frame_idx, t=t_mono)
                        if args.snapshots:
                            logger.snapshot(frame, f"alert_{atype}_{frame_idx}")
                if proximity is not None:
                    for atype, key, msg, level in proximity.update(dets_pts):
                        alert_bus.emit(atype, key, msg, level=level, frame=frame_idx, t=t_mono)
                        if args.snapshots:
                            logger.snapshot(frame, f"alert_{atype}_{frame_idx}")

            # ---- render
            if show_heat:
                frame = heatmap.overlay(frame, args.heatmap_alpha)
            if show_zones:
                for z in zones:
                    draw_zone(frame, z)
            if slots_mgr is not None:
                for slot in slots_mgr.slots:
                    draw_slot(frame, slot, t_mono)
            for rz in restricted_zones:
                draw_restricted(frame, rz.zone, armed=is_armed(armed_window))
            if show_lines:
                for ln in lines:
                    draw_line(frame, ln, flash[id(ln)])
                    flash[id(ln)] = max(0.0, flash[id(ln)] - 0.12)
            for d, meta, cls_name, kmh in canvas_draws:
                if show_trails and d.track_id >= 0:
                    draw_trail(frame, book.trails[d.track_id], meta["color"])
                label = f"#{d.track_id} {cls_name} {d.conf:.2f}"
                if kmh is not None:
                    label += f" {kmh:.0f}km/h" if args.mpp else f" {kmh:.0f}px/s"
                draw_box(frame, d, cls_name, meta["color"], label)

            now = time.perf_counter()
            inst = 1.0 / max(now - t_prev, 1e-6)
            t_prev = now
            fps_avg = 0.92 * fps_avg + 0.08 * inst if frame_idx > 3 else inst
            infer_pct_acc += engine.infer_ms / max(now - t_loop, 1e-6) * 100

            extra_txt = (f"infer {engine.infer_ms:.0f}ms ({infer_pct_acc / frame_idx:.0f}% bus)"
                         if frame_idx > 8 else "")
            if slots_mgr is not None:
                occ, tot = slots_mgr.occupancy()
                extra_txt = f"PARKING {occ}/{tot} free:{tot - occ}" + (f"    {extra_txt}" if extra_txt else "")
            elif alert_bus.counts:
                extra_txt = "alerts " + " ".join(f"{k}:{v}" for k, v in sorted(alert_bus.counts.items())) \
                    + (f"    {extra_txt}" if extra_txt else "")

            frame = draw_hud(
                frame, fps=fps_avg, backend=engine.describe(),
                in_counts=in_counts, lines=lines, zones=zones,
                scene_counts=book.scene_counts, frame_idx=frame_idx,
                total_frames=reader.total_frames if not reader.is_live else frame_idx,
                source_name=reader.name, mode=profile.name,
                extra=extra_txt)
            recent_alerts = sum(1 for ts in alert_bus._last.values()
                                if t_mono - ts < 5.0)
            frame = draw_alert_banner(frame, recent_alerts)
            frame = draw_help(frame, args.help)

            if sink is not None:
                sink.write(frame)

            for ev in events:
                logger.log_event(ev, frame_idx, fps_src,
                                 speed.latest.get(ev["track_id"], (None, None))[1] if speed else None)
                if args.snapshots:
                    logger.snapshot(frame, f"line_{ev['track_id']}_{ev['cls']}")

            if show_windows:
                now_s = time.perf_counter()
                key = None
                if now_s - last_show >= 1.0 / profile.display_fps:  # display decoupled
                    last_show = now_s
                    key = gui.show(frame)
                if key is not None:
                    if key == ord('q'):
                        user_quit = True
                        break
                    elif key == 27:
                        user_quit = "abort"
                        break
                    elif key == ord('t'):
                        show_trails = not show_trails
                    elif key == ord('h'):
                        show_heat = not show_heat
                    elif key == ord('z'):
                        show_zones = not show_zones
                    elif key == ord('l'):
                        show_lines = not show_lines
                    elif key == ord('s'):
                        p = logger.snapshot(frame, f"manual_{frame_idx}")
                        print(f"  [SNAP] {p}")
                    elif key == ord('.'):
                        paused = True
                    elif key == ord(' '):
                        paused = not paused

                    while paused and show_windows:
                        key = gui.show_paused(frame)
                        if key == ord(' '):
                            paused = False
                        elif key == ord('.'):
                            ok2, f2 = reader.read()
                            if ok2:
                                frame_idx += 1
                                engine.track(f2)
                                frame = f2
                        elif key == ord('q'):
                            paused = False
                            user_quit = True
                        elif key == 27:
                            paused = False
                            user_quit = "abort"

            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        reader.release()
        if sink is not None:
            sink.close()
        gui.close()

    wall = time.perf_counter() - t_start
    avg_fps = frame_idx / wall if wall > 0 else 0
    vehicles_in = sum(ln.counts_in.get(k, 0) for ln in lines for k in ANALYTICS_VEHICLE_CLASSES)
    tw_in = sum(ln.counts_in.get(k, 0) for ln in lines for k in TWO_WHEELER_CLASSES)
    v_cum = sum(book.scene_counts.get(k, 0) for k in VEHICLE_CLASSES)
    tw_cum = sum(book.scene_counts.get(k, 0) for k in TWO_WHEELER_CLASSES)

    print(f"\n{'=' * 78}")
    print(f" FINAL REPORT — {reader.name}")
    print(f"{'=' * 78}")
    print(f"  Frames processed : {frame_idx}"
          + (f" / {reader.total_frames}" if reader.total_frames > 0 else ""))
    print(f"  Wall time        : {wall:.1f} s   |  avg {avg_fps:.1f} FPS  "
          f"|  inference {engine.infer_ms:.1f} ms/frame ({engine.describe()})")
    for ln in lines:
        print(f"  Tripwire {ln.name:<14s}: IN {ln.total_in:3d} (uniq {len(ln.unique_in_ids):3d})"
              f"  OUT {ln.total_out:3d}  | " +
              " ".join(f"{k}:{v}" for k, v in sorted(ln.counts_in.items())))
    for z in zones:
        print(f"  Zone {z.name:<18s}: unique visitors {len(z.visited_ids)}")
    if slots_mgr is not None:
        occ, tot = slots_mgr.occupancy()
        print(f"  Parking          : {occ}/{tot} occupied ({100 * occ / max(tot, 1):.0f}%)"
              f"  free {tot - occ}")
        for s in slots_mgr.slots:
            if s.occupied:
                print(f"    {s.slot_id:<6s} {s.cls_name or 'vehicle':<10s} #{s.track_id} "
                      f"parked {s.dwell_s(t_start + wall):.0f}s")
    if alert_bus.counts:
        print(f"  Alerts           : " + "  ".join(f"{k}:{v}" for k, v in sorted(alert_bus.counts.items())))
    print(f"  Scene confirmed  : person {book.scene_counts.get('person', 0)} | "
          f"vehicles {v_cum} | two-wheelers (e-bike) {tw_cum}")
    if sink is not None:
        print(f"  Output video     : {sink.path} ({sink.backend})")
    print(f"{'=' * 78}\n", flush=True)

    return {
        "source": reader.name, "frames": frame_idx, "wall_s": round(wall, 2),
        "avg_fps": round(avg_fps, 2), "infer_ms": round(engine.infer_ms, 2),
        "backend": engine.backend, "scenario": scenario,
        "lines": {ln.name: {"in": ln.total_in, "out": ln.total_out,
                            "by_class_in": dict(ln.counts_in)} for ln in lines},
        "zones": {z.name: {"visited": len(z.visited_ids)} for z in zones},
        "parking": slots_mgr.summary() if slots_mgr is not None else None,
        "alerts": dict(alert_bus.counts),
        "scene": dict(book.scene_counts),
        "user_quit": user_quit,
    }


def build_argparser():
    ap = argparse.ArgumentParser(
        description="Apple Silicon optimized YOLO traffic analytics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("sources", nargs="*", default=None,
                    help="Video files, webcam index, or rtsp/http URLs")
    ap.add_argument("--videos", nargs="+", default=None, help="(compat alias for sources)")
    ap.add_argument("--mode", default="balanced", choices=["eco", "balanced", "turbo"],
                    help="eco: yolov8n+real-time pacing (省电) | turbo: yolov8m+大画布 (全性能)")
    ap.add_argument("--model", default=None, help="YOLO weights (.pt); mode default if omitted")
    ap.add_argument("--backend", default="auto",
                    choices=["auto", "coreml", "mps", "cuda", "cpu"],
                    help="Inference backend (auto picks the fastest)")
    ap.add_argument("--imgsz", default=None,
                    help="Inference canvas '960' or 'HxW'; mode default if omitted")
    ap.add_argument("--conf", type=float, default=0.45, help="Display/analytics confidence")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU (CoreML bakes this in at export)")
    ap.add_argument("--device", default=None, help="(compat alias for --backend)")
    ap.add_argument("--no-gui", action="store_true", help="Headless mode")
    ap.add_argument("--no-save", action="store_true", help="Do not write annotated video")
    ap.add_argument("--outdir", default="output", help="Output directory")
    ap.add_argument("--encoder", default="auto", choices=["auto", "videotoolbox", "cv2"],
                    help="Output video encoder")
    ap.add_argument("--realtime", action="store_true", default=None,
                    help="Pace file sources at native fps (eco mode default)")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--skip-frames", type=int, default=None, help="(legacy) reuse detections for N frames")
    ap.add_argument("--fast", action="store_true", help="(legacy) skip 1 frame")
    # analytics
    ap.add_argument("--line", dest="lines", action="append", default=None,
                    metavar="SPEC", help="Tripwire 'x1,y1,x2,y2[:name]' (repeatable; left of p1->p2 = IN)")
    ap.add_argument("--zone", dest="zones", action="append", default=None,
                    metavar="SPEC", help="Polygon 'x1,y1,...[:name]' (repeatable)")
    ap.add_argument("--no-default-lines", action="store_true",
                    help="Disable legacy test.mp4 default tripwires")
    ap.add_argument("--heatmap", action="store_true", help="Motion heatmap overlay")
    ap.add_argument("--heatmap-alpha", type=float, default=0.45)
    ap.add_argument("--speed", action="store_true", help="Estimate speed (px/s, km/h with --mpp)")
    ap.add_argument("--mpp", type=float, default=None, help="Meters per pixel for km/h speed")
    ap.add_argument("--vote-window", type=int, default=15, help="Class majority-vote window")
    ap.add_argument("--confirm-hits", type=int, default=5, help="Frames before a track counts as scene object")
    ap.add_argument("--containment", type=float, default=0.70, help="Nested-box suppression threshold")
    # logging
    ap.add_argument("--no-log", action="store_true", help="Disable CSV/JSON session logging")
    ap.add_argument("--snapshots", action="store_true", help="Save frame on every crossing event")
    ap.add_argument("--help-keys", dest="help", action="store_true", help="Show key overlay")
    ap.add_argument("--verbose", action="store_true")
    # scenarios (auto-selected when slots/restricted given; road otherwise)
    ap.add_argument("--scenario", default=None, choices=["road", "parking", "factory"],
                    help="road: 交通 | parking: 停车场 | factory: 厂区安防 (auto if omitted)")
    ap.add_argument("--slot", dest="slots", action="append", default=None, metavar="SPEC",
                    help="车位 'x1,y1,x2,y2[:编号]'（矩形或多边形，原始像素，可重复）")
    ap.add_argument("--slot-grid", dest="slot_grid", action="append", default=None, metavar="SPEC",
                    help="车位网格 'x,y,w,h,列,行[:前缀]'（批量生成，如 A1..A8）")
    ap.add_argument("--parking-stability", type=int, default=6,
                    help="车位状态翻转所需连续帧数（防抖）")
    ap.add_argument("--restricted", dest="restricted", action="append", default=None, metavar="SPEC",
                    help="禁区 'x1,y1,...[:名称[:触发类别]]'（默认触发类别 person）")
    ap.add_argument("--loiter-sec", type=float, default=20.0, help="禁区滞留告警阈值（秒）")
    ap.add_argument("--proximity-px", type=float, default=0.0,
                    help="人车接近告警距离（px，0=关闭；厂区推荐 120）")
    ap.add_argument("--armed-hours", default=None, metavar="START-END",
                    help="布防时段 '8-18' 或 '22-6'（默认 24h 布防）")
    ap.add_argument("--webhook", default=None, help="告警 JSON POST 地址（如企业微信/钉钉/Slack 网关）")
    ap.add_argument("--alert-cooldown", type=float, default=30.0, help="同类告警去重间隔（秒）")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    if args.device:
        args.backend = args.device

    # Performance mode fills in only what the user left unspecified.
    profile = get_profile(args.mode)
    if args.model is None:
        args.model = profile.model
    if args.imgsz is None:
        args.imgsz = profile.imgsz
    if args.skip_frames is None:
        args.skip_frames = profile.skip_frames
    if args.realtime is None:
        args.realtime = profile.realtime
    if args.fast and not args.skip_frames:
        args.skip_frames = 1
    set_qos(profile.qos)  # macOS P/E-core scheduling hint (before threads spawn)

    sources = args.sources or args.videos or ["test.mp4"]
    if len(sources) == 1 and sources[0].isdigit():
        sources = [int(sources[0])]
    try:
        args.lines = [parse_line(s) for s in (args.lines or [])]
        args.zones = [parse_zone(s) for s in (args.zones or [])]
        args.slots = [parse_slot(s, i) for i, s in enumerate(args.slots or [])]
        args.slot_grid = [s for g in (args.slot_grid or []) for s in parse_slot_grid(g)]
        args.restricted = [parse_restricted(s) for s in (args.restricted or [])]
        args.armed_hours = parse_armed_hours(args.armed_hours)
        args.scenario = resolve_scenario(args)
    except ValueError as e:
        print(f"[ARG ERROR] {e}")
        return 2

    print("=" * 78)
    print(f" YOLO Traffic Analytics — Apple Silicon Edition  [{args.mode.upper()} MODE]"
          f"  scenario: {args.scenario}")
    print(f" Model {args.model} | canvas {args.imgsz} | backend {args.backend}"
          + (" | real-time pacing" if args.realtime else ""))
    print("=" * 78)

    imgsz = ([int(v) for v in args.imgsz.lower().split("x")]
             if "x" in args.imgsz.lower() else [int(args.imgsz)] * 2)
    engine = InferenceEngine(
        weights=args.model, imgsz=imgsz, backend=args.backend,
        classes=SUPPORTED, tracker_cfg="custom_bytetrack.yaml",
        display_conf=args.conf, iou=args.iou)
    print(f"[ENGINE] {engine.describe()}  (selected backend: {engine.backend})")

    logger = SessionLogger(out_dir=args.outdir, source_name=Path(str(sources[0])).name,
                           enabled=not args.no_log)

    results = []
    for src in sources:
        stat = process_source(src, engine, args, logger, profile=profile)
        results.append(stat)
        if stat and stat.get("user_quit") == "abort":
            print("[ABORT] Stopped by user.")
            break

    ok_results = [r for r in results if r]
    if ok_results:
        logger.write_summary(
            cfg={k: str(v) for k, v in vars(args).items()},
            per_source={r["source"]: r for r in ok_results})
        if not args.no_log:
            print(f"[LOG] Events CSV + JSON summary in '{args.outdir}/'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
