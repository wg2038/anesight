#!/usr/bin/env python3
"""Session logging: CSV crossing events, JSON summary, event snapshots."""

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import cv2


class SessionLogger:
    def __init__(self, out_dir="output", source_name="stream", enabled=True):
        self.enabled = enabled
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"{Path(source_name).stem}_{stamp}"
        self.csv_path = self.out_dir / f"events_{stem}.csv"
        self.alerts_path = self.out_dir / f"alerts_{stem}.csv"
        self.json_path = self.out_dir / f"summary_{stem}.json"
        self.snap_dir = self.out_dir / "snapshots"
        self.t0 = time.monotonic()
        if enabled:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["wall_time", "video_sec", "event", "line", "direction",
                     "track_id", "class", "speed_kmh", "frame"])
            with open(self.alerts_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["wall_time", "video_sec", "type", "level", "key", "message", "frame"])

    def log_event(self, event: dict, frame_idx: int, fps: float, speed_kmh=None):
        if not self.enabled:
            return
        video_sec = frame_idx / fps if fps else 0.0
        with open(self.csv_path, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%H:%M:%S.%f")[:-3],
                f"{video_sec:.2f}",
                event.get("event", "crossing"),
                event.get("line", ""),
                event.get("direction", ""),
                event.get("track_id", ""),
                event.get("cls", ""),
                "" if speed_kmh is None else f"{speed_kmh:.1f}",
                frame_idx,
            ])

    def log_alert(self, alert, frame_idx: int | None = None, fps: float | None = None):
        if not self.enabled:
            return
        video_sec = (frame_idx / fps) if (frame_idx and fps) else None
        with open(self.alerts_path, "a", newline="") as f:
            csv.writer(f).writerow([
                datetime.now().strftime("%H:%M:%S.%f")[:-3],
                f"{video_sec:.2f}" if video_sec is not None else "",
                alert.type, alert.level, alert.key, alert.message,
                frame_idx if frame_idx is not None else alert.frame,
            ])

    def snapshot(self, frame, tag: str):
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        p = self.snap_dir / f"{datetime.now().strftime('%H%M%S')}_{tag}.png"
        cv2.imwrite(str(p), frame)
        return p

    def write_summary(self, cfg: dict, per_source: dict):
        if not self.enabled:
            return None
        payload = {
            "generated": datetime.now().isoformat(),
            "duration_s": round(time.monotonic() - self.t0, 2),
            "config": cfg,
            "sources": per_source,
        }
        self.json_path.write_text(json.dumps(payload, indent=2, default=str))
        return self.json_path
