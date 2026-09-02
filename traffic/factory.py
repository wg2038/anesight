#!/usr/bin/env python3
"""Factory / site security: restricted zones (intrusion + loitering), armed schedules,
and vehicle-pedestrian proximity warnings (forklift & pedestrian collision avoidance).

All classes are pure logic over detections + timestamps — no I/O — so they are unit
testable. Alerts are returned as (type, key, message, level) tuples and pushed through
traffic.alerts.AlertBus by the caller.
"""

from datetime import datetime

import numpy as np


def parse_armed_hours(spec: str | None) -> tuple[int, int] | None:
    """'8-18' -> (8, 18); '22-6' -> (22, 6) overnight wrap; None/''/'always' -> None."""
    if not spec or spec.strip().lower() in ("", "always", "24-7", "24/7"):
        return None
    s = spec.strip().replace("：", ":").split("-")
    if len(s) != 2:
        raise ValueError(f"--armed-hours needs 'START-END' like '8-18' — got '{spec}'")
    h1, h2 = int(s[0]), int(s[1])
    if not (0 <= h1 <= 24 and 0 <= h2 <= 24):
        raise ValueError(f"--armed-hours hours must be 0-24 — got '{spec}'")
    return (h1, h2)


def is_armed(window: tuple[int, int] | None, now: datetime | None = None) -> bool:
    if window is None:
        return True
    now = now or datetime.now()
    h1, h2 = window
    if h1 <= h2:
        return h1 <= now.hour < h2 or (h1 == h2 and now.hour == h1)
    return now.hour >= h1 or now.hour < h2  # overnight window, e.g. 22-6


class RestrictedZone:
    """Polygon where `trigger_classes` raise alerts, with loiter-time escalation.

    Alerts (deduped downstream by AlertBus):
      intrusion — a trigger-class track just entered the armed zone
      loiter    — the same track stayed >= loiter_sec inside
    """

    def __init__(self, polygon, name="restricted",
                 trigger_classes=("person",), loiter_sec: float = 20.0):
        from traffic.analytics import PolygonZone
        self.zone = PolygonZone(polygon, name)
        self.name = name
        self.trigger_classes = tuple(trigger_classes)
        self.loiter_sec = float(loiter_sec)
        self._dwell: dict[int, float] = {}   # track_id -> first-seen monotonic t
        self._loitered: set[int] = set()

    def update(self, dets_with_pts, t: float) -> list[tuple[str, str, str, str]]:
        """dets_with_pts: iterable of (cls_name, track_id, (cx, cy), box).

        Returns alert tuples (type, key, message, level).
        """
        alerts = []
        inside_now: set[int] = set()
        for cls_name, tid, pt, box in dets_with_pts:
            if cls_name not in self.trigger_classes or tid is None or tid < 0:
                continue
            if not self.zone.contains(pt):
                continue
            inside_now.add(tid)
            first = self._dwell.get(tid)
            if first is None:
                self._dwell[tid] = t
                alerts.append(("intrusion", f"intrusion:{self.name}:{tid}",
                               f"{cls_name} #{tid} entered restricted zone '{self.name}'",
                               "critical"))
            elif (t - first >= self.loiter_sec and tid not in self._loitered):
                self._loitered.add(tid)
                alerts.append(("loiter", f"loiter:{self.name}:{tid}",
                               f"{cls_name} #{tid} loitering in '{self.name}' "
                               f"for {t - first:.0f}s", "warning"))
        # Tracks that left: keep dwell reset so re-entry re-alerts (AlertBus dedups spam).
        left = set(self._dwell) - inside_now
        for tid in left:
            self._dwell.pop(tid, None)
            self._loitered.discard(tid)
        return alerts

    def summary(self) -> dict:
        return {"name": self.name, "inside_now": len(self._dwell)}


class ProximityMonitor:
    """Alert when a person is within `threshold_px` of a vehicle (collision risk)."""

    def __init__(self, threshold_px: float = 120.0,
                 person_class="person",
                 vehicle_classes=("car", "bus", "truck", "motorcycle", "bicycle")):
        self.threshold_px = float(threshold_px)
        self.person_class = person_class
        self.vehicle_classes = tuple(vehicle_classes)

    @staticmethod
    def _point_rect_dist(px, py, xyxy) -> float:
        x1, y1, x2, y2 = xyxy
        dx = max(x1 - px, 0.0, px - x2)
        dy = max(y1 - py, 0.0, py - y2)
        return float(np.hypot(dx, dy))

    def update(self, dets_with_pts) -> list[tuple[str, str, str, str]]:
        """dets_with_pts: iterable of (cls_name, track_id, (cx, cy), box)."""
        people = [(tid, pt) for cls_name, tid, pt, _ in dets_with_pts
                  if cls_name == self.person_class and tid is not None and tid >= 0]
        vehicles = [(tid, cls_name, box) for cls_name, tid, _, box in dets_with_pts
                    if cls_name in self.vehicle_classes and tid is not None and tid >= 0]
        alerts = []
        if not people or not vehicles:
            return alerts
        for pid, ppt in people:
            best = None  # (dist, vid, cls)
            for vid, cls_name, box in vehicles:
                d = self._point_rect_dist(ppt[0], ppt[1], box)
                if best is None or d < best[0]:
                    best = (d, vid, cls_name)
            if best is not None and best[0] < self.threshold_px:
                d, vid, cls_name = best
                pair = tuple(sorted((pid, vid)))
                alerts.append(("proximity", f"proximity:{pair[0]}:{pair[1]}",
                               f"person #{pid} within {d:.0f}px of {cls_name} #{vid}",
                               "warning"))
        return alerts
