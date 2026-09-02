#!/usr/bin/env python3
"""Traffic analytics: directional tripwires, polygon zones, heatmaps, speed.

All geometry works in original frame pixel coordinates. Designed to be allocation-free
in the hot path (no numpy object churn per frame beyond small fixed arrays).
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from traffic.engine import Detection


# --------------------------------------------------------------------------- lines

class DirectionalLine:
    """Tripwire with direction-aware in/out counting and per-class totals.

    Crossing detection uses signed perpendicular distance with a hysteresis band:
    a track's side only "commits" once it is farther than `hysteresis_px` from the
    line, so stationary objects jittering on the line never generate events. A
    crossing fires when the committed side flips. Each track is counted at most
    once per direction (`count_once=True`), which matches real traffic: one
    vehicle crosses a stop line once.

    The "positive" side is the left side of the directed segment p1->p2 (sign of
    the cross product), so flipping p1/p2 flips what counts as 'in'. `classes`
    optionally restricts which class names are counted (None = all).
    """

    def __init__(self, p1, p2, name="line", classes=None,
                 hysteresis_px: float = 6.0, count_once: bool = True):
        self.p1 = (float(p1[0]), float(p1[1]))
        self.p2 = (float(p2[0]), float(p2[1]))
        self.name = name
        self.classes = set(classes) if classes else None
        self.hysteresis_px = float(hysteresis_px)
        self.count_once = count_once
        self.counts_in = defaultdict(int)    # cls_name -> int
        self.counts_out = defaultdict(int)
        self.in_ids = set()
        self.out_ids = set()
        self._side = {}                       # track_id -> committed side (-1/0/+1)
        self._nx, self._ny = self._unit_normal()

    def _unit_normal(self):
        dx, dy = self.p2[0] - self.p1[0], self.p2[1] - self.p1[1]
        n = np.hypot(dx, dy) or 1.0
        # Left normal of p1->p2.
        return -dy / n, dx / n

    def signed_dist(self, pt) -> float:
        """Signed perpendicular distance: >0 on the left of p1->p2."""
        return (pt[0] - self.p1[0]) * self._nx + (pt[1] - self.p1[1]) * self._ny

    def signed_side(self, pt) -> int:
        d = self.signed_dist(pt)
        if abs(d) < 1e-6:
            return 0
        return 1 if d > 0 else -1

    def update(self, track_id: int, prev_pt, curr_pt, cls_name: str):
        """Feed one track movement step. Returns event dict on a fresh crossing."""
        if self.classes is not None and cls_name not in self.classes:
            return None
        state = self._side.get(track_id, 0)

        if state == 0:
            # First sighting of this track: commit its side from where it came
            # (prev_pt preferred; curr_pt when the track spawns on the line).
            d_prev = self.signed_dist(prev_pt)
            if abs(d_prev) > self.hysteresis_px:
                self._side[track_id] = 1 if d_prev > 0 else -1
                state = self._side[track_id]
            else:
                d_curr = self.signed_dist(curr_pt)
                if abs(d_curr) > self.hysteresis_px:
                    self._side[track_id] = 1 if d_curr > 0 else -1
                return None

        d = self.signed_dist(curr_pt)
        if state < 0 and d > self.hysteresis_px:
            direction = "in"
            new_side = 1
        elif state > 0 and d < -self.hysteresis_px:
            direction = "out"
            new_side = -1
        else:
            return None

        already = track_id in self.in_ids if direction == "in" else track_id in self.out_ids
        if direction == "in":
            self.in_ids.add(track_id)
        else:
            self.out_ids.add(track_id)
        if not (self.count_once and already):
            bucket = self.counts_in if direction == "in" else self.counts_out
            bucket[cls_name] += 1
            event = {"line": self.name, "direction": direction,
                     "track_id": track_id, "cls": cls_name}
        else:
            event = None
        self._side[track_id] = new_side
        return event

    @property
    def total_in(self) -> int:
        return sum(self.counts_in.values())

    @property
    def total_out(self) -> int:
        return sum(self.counts_out.values())

    @property
    def unique_in_ids(self) -> set:
        return self.in_ids

    def summary(self):
        return {
            "name": self.name,
            "p1": self.p1, "p2": self.p2,
            "in": dict(self.counts_in), "out": dict(self.counts_out),
            "unique_in": len(self.in_ids),
        }


# --------------------------------------------------------------------------- zones

class PolygonZone:
    """Named polygon region with per-class occupancy counting."""

    def __init__(self, polygon, name="zone"):
        self.polygon = np.asarray(polygon, dtype=np.float32)
        self.name = name
        self.occupancy = defaultdict(int)   # per frame, per class
        self.visited_ids = set()            # unique tracks ever inside

    def contains(self, pt) -> bool:
        return cv2.pointPolygonTest(self.polygon.astype(np.float32),
                                    (float(pt[0]), float(pt[1])), False) >= 0

    def update(self, dets_with_pts):
        """dets_with_pts: iterable of (cls_name, track_id, point). Resets occupancy."""
        self.occupancy.clear()
        for cls_name, tid, pt in dets_with_pts:
            if self.contains(pt):
                self.occupancy[cls_name] += 1
                if tid is not None and tid >= 0:
                    self.visited_ids.add(tid)
        return self.occupancy

    @property
    def total_occupancy(self) -> int:
        return sum(self.occupancy.values())

    def summary(self):
        return {"name": self.name, "polygon": self.polygon.tolist(),
                "visited_unique": len(self.visited_ids)}


# --------------------------------------------------------------------------- heatmap

class MotionHeatmap:
    """Decay-based motion heatmap accumulated from detection footprints."""

    def __init__(self, width: int, height: int, decay: float = 0.94,
                 colorize_every: int = 2):
        self.buf = np.zeros((int(height), int(width)), dtype=np.float32)
        self._scratch = np.zeros_like(self.buf)
        self.decay = decay
        self.colorize_every = colorize_every
        self._cached = None
        self._frame = 0

    def update(self, dets, footprint=0.35):
        """Add a blob at each detection's bottom-center footprint."""
        self.buf *= self.decay
        for d in dets:
            x1, y1, x2, y2 = d.xyxy
            r = max(4, int((y2 - y1) * footprint))
            cx, cy = int(d.cx), int(min(self.buf.shape[0] - 1, y2))
            self._scratch[:] = 0
            cv2.circle(self._scratch, (cx, cy), r, 1.0, -1)
            self.buf += self._scratch
        self._frame += 1
        if self._cached is None or self._frame % self.colorize_every == 0:
            # Colorize at half resolution (the heatmap is blurry by nature) and
            # upscale — 4x cheaper than full-res colormap+blur.
            small = cv2.resize(self.buf, (self.buf.shape[1] // 2, self.buf.shape[0] // 2),
                               interpolation=cv2.INTER_AREA)
            norm = np.clip(small / max(4.0, small.max()), 0, 1)
            heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heat = cv2.GaussianBlur(heat, (15, 15), 0)
            self._cached = cv2.resize(heat, (self.buf.shape[1], self.buf.shape[0]),
                                      interpolation=cv2.INTER_LINEAR)
        return self._cached

    def overlay(self, canvas, alpha: float = 0.45):
        if self._cached is None:
            return canvas
        mask = self.buf > 0.15
        m3 = cv2.merge([mask, mask, mask])
        blended = cv2.addWeighted(self._cached, alpha, canvas, 1.0, 0)
        return np.where(m3, blended, canvas)


# --------------------------------------------------------------------------- speed

class SpeedEstimator:
    """Per-track speed from centroid displacement (px/s, and km/h if mpp given)."""

    def __init__(self, window_s: float = 0.8, mpp: float | None = None):
        self.window_s = window_s
        self.mpp = mpp  # meters per pixel (calibration)
        self._hist = defaultdict(deque)  # track_id -> deque[(t, cx, cy)]
        self.latest = {}                 # track_id -> (px_s, kmh or None)

    def update(self, track_id: int, cx: float, cy: float, t: float | None = None):
        t = t if t is not None else time.monotonic()
        h = self._hist[track_id]
        h.append((t, cx, cy))
        while len(h) > 2 and t - h[0][0] > self.window_s:
            h.popleft()
        px_s = kmh = None
        if len(h) >= 2:
            (t0, x0, y0), (t1, x1, y1) = h[0], h[-1]
            dt = t1 - t0
            if dt > 1e-3:
                dist = float(np.hypot(x1 - x0, y1 - y0))
                px_s = dist / dt
                if self.mpp:
                    kmh = px_s * self.mpp * 3.6
        self.latest[track_id] = (px_s, kmh)
        return self.latest[track_id]

    def forget_old(self, active_ids):
        stale = set(self._hist) - set(active_ids)
        for s in stale:
            self._hist.pop(s, None)
            self.latest.pop(s, None)


# --------------------------------------------------------------------------- tracks

class TrackBook:
    """Per-stream track state: class voting, confirmation, trails, nesting suppression."""

    def __init__(self, vote_window: int = 15, confirm_hits: int = 5,
                 trail_len: int = 14, containment_thresh: float = 0.70):
        self.vote_window = vote_window
        self.confirm_hits = confirm_hits
        self.trail_len = trail_len
        self.containment_thresh = containment_thresh

        self.class_history = defaultdict(deque)  # tid -> deque[cls]
        self.hits = defaultdict(int)
        self.confirmed = set()
        self.trails = defaultdict(deque)         # tid -> deque[(cx, cy)]
        self.scene_counts = defaultdict(int)     # confirmed unique per class
        self._last_pt = {}                       # tid -> bottom-center

    def stabilize_class(self, tid: int, cls_id: int) -> int:
        """Majority vote over the last N frames to stop label flicker."""
        if tid is None or tid < 0:
            return cls_id
        h = self.class_history[tid]
        h.append(cls_id)
        if len(h) > self.vote_window:
            h.popleft()
        return int(np.bincount(list(h)).argmax())

    def update_track(self, d: Detection, cls_id: int):
        """Register one observation; returns (confirmed_new, prev_pt, curr_pt)."""
        tid = d.track_id
        self.hits[tid] += 1
        pt = d.bottom_center
        prev_pt = self._last_pt.get(tid, pt)
        self._last_pt[tid] = pt
        self.trails[tid].append((int(pt[0]), int(pt[1])))
        if len(self.trails[tid]) > self.trail_len:
            self.trails[tid].popleft()
        newly_confirmed = False
        if self.hits[tid] >= self.confirm_hits and tid not in self.confirmed:
            self.confirmed.add(tid)
            newly_confirmed = True
            self.scene_counts[cls_id] += 1
        return newly_confirmed, prev_pt, pt


class SuppressNested:
    """Drop boxes that are >= threshold contained inside a larger kept box."""

    def __init__(self, containment: float = 0.70):
        self.containment = containment

    def __call__(self, dets: list[Detection]) -> list[Detection]:
        if len(dets) <= 1:
            return dets
        boxes = np.array([d.xyxy for d in dets], dtype=np.float32)
        areas = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)
        order = np.argsort(-areas)
        suppressed = np.zeros(len(dets), dtype=bool)
        kept: list[Detection] = []
        for i_pos, i in enumerate(order):
            if suppressed[i]:
                continue
            kept.append(dets[i])
            bx = boxes[i]
            rest = boxes[order[i_pos + 1:]]
            ix1 = np.maximum(bx[0], rest[:, 0]); iy1 = np.maximum(bx[1], rest[:, 1])
            ix2 = np.minimum(bx[2], rest[:, 2]); iy2 = np.minimum(bx[3], rest[:, 3])
            inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
            rest_areas = areas[order[i_pos + 1:]]
            containment = np.divide(inter, rest_areas, out=np.zeros_like(inter),
                                    where=rest_areas > 0)
            drop = order[i_pos + 1:][containment >= self.containment]
            suppressed[drop] = True
        return kept
