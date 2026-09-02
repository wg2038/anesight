#!/usr/bin/env python3
"""Parking management: per-slot occupancy state machine + turnover analytics.

Modeled on ultralytics' parking-management solution but detection-based: a slot is
"occupied" when a vehicle box overlaps it by >= `match_ratio` of the vehicle area for
`stability` consecutive frames (hysteresis prevents flip-flopping on partial overlaps).
Slots can be individual polygons or auto-generated grids — parking rows are grids.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

PARKING_VEHICLE_CLASSES = ("car", "bus", "truck", "motorcycle", "bicycle")


@dataclass
class ParkingSlot:
    slot_id: str
    polygon: np.ndarray            # (N,2) float32
    occupied: bool = False
    track_id: int | None = None
    cls_name: str | None = None
    since_t: float = 0.0           # monotonic time of last state flip
    _pending: bool = False
    _streak: int = 0

    def match_ratio(self, xyxy) -> float:
        """Intersection over vehicle-box area (how much of the vehicle is in the slot)."""
        px = self.polygon.astype(np.float32)
        x1, y1, x2, y2 = xyxy
        box = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        # Rasterize-free: use polygon/box intersection via cv2 on integer grid coords.
        inter = _poly_box_intersection_area(px, box)
        area = max((x2 - x1) * (y2 - y1), 1e-6)
        return inter / area

    def dwell_s(self, t: float) -> float:
        return max(0.0, t - self.since_t) if self.occupied else 0.0


def _poly_box_intersection_area(poly, box) -> float:
    """Area of polygon ∩ axis-aligned box, via cv2 intersectConvexConvex."""
    bx = np.array([[box[0][0], box[0][1]], [box[1][0], box[1][1]],
                   [box[2][0], box[2][1]], [box[3][0], box[3][1]]], dtype=np.float32)
    try:
        inter, _ = cv2.intersectConvexConvex(poly.reshape(-1, 1, 2), bx.reshape(-1, 1, 2))
        return float(inter)
    except Exception:
        return 0.0


class SlotManager:
    def __init__(self, slots: list[ParkingSlot], stability: int = 6,
                 match_ratio: float = 0.5,
                 vehicle_classes=PARKING_VEHICLE_CLASSES):
        if not slots:
            raise ValueError("SlotManager needs at least one slot")
        self.slots = slots
        self.stability = max(1, int(stability))
        self.match_ratio = match_ratio
        self.vehicle_classes = tuple(vehicle_classes)
        self.by_id = {s.slot_id: s for s in slots}

    def update(self, vehicle_dets, t: float) -> list[dict]:
        """Feed displayed vehicle detections; returns slot_occupied/slot_freed events.

        vehicle_dets: iterable of (Detection, cls_name).
        """
        matched: dict[str, tuple[object, str, float]] = {}  # slot_id -> (det, cls, ratio)
        for det, cls_name in vehicle_dets:
            if cls_name not in self.vehicle_classes:
                continue
            best_id, best_r = None, self.match_ratio
            for slot in self.slots:
                r = slot.match_ratio(det.xyxy)
                if r > best_r:
                    best_id, best_r = slot.slot_id, r
            if best_id is not None:
                prev = matched.get(best_id)
                if prev is None or best_r > prev[2]:
                    matched[best_id] = (det, cls_name, best_r)

        events = []
        for slot in self.slots:
            m = matched.get(slot.slot_id)
            pending = m is not None
            if pending:
                det, cls_name, _ = m
                if slot.occupied and slot.track_id != det.track_id:
                    # Different vehicle claims the slot — treat as a swap.
                    events.append({"event": "slot_freed", "slot": slot.slot_id,
                                   "track_id": slot.track_id, "cls": slot.cls_name,
                                   "dwell_s": round(slot.dwell_s(t), 1)})
                    slot.occupied = False
            if pending != slot.occupied:
                slot._pending = pending
                slot._streak += 1
                if slot._streak >= self.stability:
                    prev_dwell = slot.dwell_s(t)   # capture BEFORE resetting since_t
                    slot.occupied = pending
                    slot._streak = 0
                    slot.since_t = t
                    if pending:
                        det, cls_name, _ = m
                        slot.track_id = det.track_id
                        slot.cls_name = cls_name
                        events.append({"event": "slot_occupied", "slot": slot.slot_id,
                                       "track_id": det.track_id, "cls": cls_name, "dwell_s": 0.0})
                    else:
                        events.append({"event": "slot_freed", "slot": slot.slot_id,
                                       "track_id": slot.track_id, "cls": slot.cls_name,
                                       "dwell_s": round(prev_dwell, 1)})
                        slot.track_id = None
            else:
                slot._streak = 0
                if pending:
                    det, cls_name, _ = m
                    slot.track_id = det.track_id
                    slot.cls_name = cls_name
        return events

    def occupancy(self) -> tuple[int, int]:
        occupied = sum(1 for s in self.slots if s.occupied)
        return occupied, len(self.slots)

    def summary(self) -> dict:
        occupied, total = self.occupancy()
        return {
            "total": total, "occupied": occupied, "free": total - occupied,
            "occupancy_ratio": round(occupied / total, 3) if total else 0.0,
            "slots": {s.slot_id: {"occupied": s.occupied, "cls": s.cls_name,
                                  "track_id": s.track_id}
                      for s in self.slots},
        }


def parse_slot(spec: str, index: int = 0) -> ParkingSlot:
    """'x1,y1,...,xN,yN[:slot_id]' (4 numbers = rect). Default id S<index+1>."""
    parts = spec.split(":")
    coords = [float(v) for v in parts[0].split(",")]
    sid = parts[1] if len(parts) > 1 and parts[1] else f"S{index + 1}"
    if len(coords) == 4:
        x1, y1, x2, y2 = coords
        poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    else:
        if len(coords) < 6 or len(coords) % 2:
            raise ValueError(f"--slot needs 4 rect coords or 3+ polygon pairs — got '{spec}'")
        poly = list(zip(coords[0::2], coords[1::2]))
    return ParkingSlot(slot_id=sid, polygon=np.asarray(poly, dtype=np.float32))


def parse_slot_grid(spec: str) -> list[ParkingSlot]:
    """'x,y,w,h,cols,rows[:prefix]' -> cols*rows slots.

    (x, y) is the grid's top-left corner; slots run left→right, top→bottom.
    Without a prefix, ids use row letters: A1..A8, B1..B8 (garage convention).
    With a prefix (e.g. ':P'), ids are continuous: P1..P16.
    """
    parts = spec.split(":")
    vals = [float(v) for v in parts[0].split(",")]
    if len(vals) != 6:
        raise ValueError(f"--slot-grid needs x,y,w,h,cols,rows — got '{spec}'")
    x, y, w, h, cols, rows = vals
    cols, rows = int(cols), int(rows)
    if cols < 1 or rows < 1 or w <= 0 or h <= 0:
        raise ValueError(f"--slot-grid dimensions invalid — got '{spec}'")
    prefix = parts[1] if len(parts) > 1 and parts[1] else None
    letters = "ABCDEFGHIJKLMNOP"
    slots = []
    for r in range(rows):
        for c in range(cols):
            sid = (f"{prefix}{r * cols + c + 1}" if prefix
                   else f"{letters[r % len(letters)]}{c + 1}")
            poly = [(x + c * w, y + r * h), (x + (c + 1) * w, y + r * h),
                    (x + (c + 1) * w, y + (r + 1) * h), (x + c * w, y + (r + 1) * h)]
            slots.append(ParkingSlot(slot_id=sid, polygon=np.asarray(poly, dtype=np.float32)))
    return slots
