#!/usr/bin/env python3
"""Rendering: HUD dashboard, boxes, trails, tripwires, zones, heatmap."""

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX


class Palette:
    PERSON = (255, 150, 40)
    CAR = (0, 220, 90)
    MOTOR = (60, 255, 255)
    BICYCLE = (140, 255, 200)
    BUS = (0, 150, 255)
    TRUCK = (220, 60, 255)
    LINE_IN = (0, 230, 118)
    LINE_OUT = (90, 160, 255)
    ZONE = (180, 120, 255)
    TEXT = (235, 235, 235)


# Two-wheeler classes (COCO has no 'e-bike'; e-bikes surface as motorcycle AND bicycle,
# both usually at lower confidence than cars). They form the TW category in analytics.
TWO_WHEELER_NAMES = {"motorcycle", "bicycle"}

CLASS_META = {
    0: {"name": "person", "zh": "行人", "color": Palette.PERSON},
    1: {"name": "bicycle", "zh": "两轮车(自行车/电动)", "color": Palette.BICYCLE,
        "min_conf": 0.30},
    2: {"name": "car", "zh": "轿车", "color": Palette.CAR},
    3: {"name": "motorcycle", "zh": "两轮车(摩托/电动)", "color": Palette.MOTOR,
        "min_conf": 0.30},
    5: {"name": "bus", "zh": "公交", "color": Palette.BUS},
    7: {"name": "truck", "zh": "卡车", "color": Palette.TRUCK},
}


def draw_box(canvas, det, cls_name, color, label, thickness=2):
    x1, y1, x2, y2 = (int(v) for v in det.xyxy)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    (tw, th), base = cv2.getTextSize(label, FONT, 0.42, 1)
    y_top = max(0, y1 - th - base - 5)
    cv2.rectangle(canvas, (x1, y_top), (x1 + tw + 6, y_top + th + base + 4), color, -1)
    cv2.putText(canvas, label, (x1 + 3, y_top + th + 2), FONT, 0.42,
                (15, 15, 15), 1, cv2.LINE_AA)


def draw_trail(canvas, pts, color):
    if len(pts) < 3:
        return
    arr = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(canvas, [arr], False, color, 2, cv2.LINE_AA)


def draw_line(canvas, line: "DirectionalLine-like", flash: float = 0.0):
    p1, p2 = (int(line.p1[0]), int(line.p1[1])), (int(line.p2[0]), int(line.p2[1]))
    color = Palette.LINE_IN if flash > 0 else (0, 180, 90)
    thick = 4 if flash > 0 else 2
    cv2.line(canvas, p1, p2, color, thick, cv2.LINE_AA)
    # Direction arrow along the segment (left side = 'in')
    dx, dy = line.p2[0] - line.p1[0], line.p2[1] - line.p1[1]
    n = (dx * dx + dy * dy) ** 0.5 or 1.0
    ux, uy = dx / n, dy / n
    nx, ny = -uy, ux  # left normal
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    ax, ay = int(mx + nx * 18), int(my + ny * 18)
    cv2.arrowedLine(canvas, (int(mx), int(my)), (ax, ay), color, 2, cv2.LINE_AA, tipLength=0.35)
    label = f"{line.name} IN:{line.total_in} OUT:{line.total_out}"
    cv2.putText(canvas, label, (p1[0], p1[1] - 10), FONT, 0.48, color, 1, cv2.LINE_AA)


def draw_zone(canvas, zone: "PolygonZone-like"):
    pts = zone.polygon.astype(np.int32).reshape(-1, 1, 2)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts], (*Palette.ZONE, ))
    cv2.addWeighted(overlay, 0.12, canvas, 0.88, 0, canvas)
    cv2.polylines(canvas, [pts], True, Palette.ZONE, 2, cv2.LINE_AA)
    label = f"{zone.name} occ:{zone.total_occupancy}"
    x0, y0 = pts[0][0]
    cv2.putText(canvas, label, (int(x0), int(y0) - 8), FONT, 0.48, Palette.ZONE, 1, cv2.LINE_AA)


def draw_slot(canvas, slot, t: float | None = None):
    """Parking slot: green = free, red = occupied, with dwell time."""
    pts = slot.polygon.astype(np.int32).reshape(-1, 1, 2)
    color = (0, 100, 255) if slot.occupied else (90, 200, 90)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)
    cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)
    x0, y0 = pts[0][0]
    if slot.occupied:
        label = f"{slot.slot_id} {slot.cls_name or ''}"
        if t is not None and slot.since_t > 0:
            d = max(0, int(t - slot.since_t))
            label += f" {d // 60:02d}:{d % 60:02d}"
    else:
        label = f"{slot.slot_id}"
    cv2.putText(canvas, label, (int(x0) + 2, int(y0) + 18), FONT, 0.42, color, 1, cv2.LINE_AA)


def draw_restricted(canvas, zone: "PolygonZone-like", armed: bool = True):
    """Restricted zone: red outline; gray/dashed look when disarmed."""
    pts = zone.polygon.astype(np.int32).reshape(-1, 1, 2)
    color = (60, 60, 255) if armed else (140, 140, 140)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.10 if armed else 0.05, canvas, 0.9, 0, canvas)
    cv2.polylines(canvas, [pts], True, color, 2, cv2.LINE_AA)
    label = f"{zone.name}" + ("" if armed else " (disarmed)")
    x0, y0 = pts[0][0]
    cv2.putText(canvas, label, (int(x0), int(y0) - 8), FONT, 0.48, color, 1, cv2.LINE_AA)


def draw_alert_banner(canvas, alerts_active: int):
    """Flash a top-right banner when security alerts fired recently."""
    if alerts_active <= 0:
        return canvas
    h, w = canvas.shape[:2]
    text = f"!! {alerts_active} ALERT{'S' if alerts_active > 1 else ''} !!"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.6, 2)
    cv2.rectangle(canvas, (w - tw - 26, 8), (w - 8, th + 22), (40, 40, 220), -1)
    cv2.putText(canvas, text, (w - tw - 18, th + 16), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def _progress_bar(canvas, frac, x, y, w, h, color=(0, 200, 120)):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (70, 70, 70), -1)
    cv2.rectangle(canvas, (x, y), (x + int(w * min(1.0, max(0.0, frac))), y + h), color, -1)


def draw_hud(canvas, *, fps, backend, in_counts, lines, zones,
             scene_counts, frame_idx, total_frames, source_name, extra="", mode=None):
    """Top dashboard: consistent with printed terminal stats."""
    h, w = canvas.shape[:2]
    rows = 6
    banner_h = 16 + 20 * rows
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (18, 20, 22), -1)
    cv2.addWeighted(overlay, 0.86, canvas, 0.14, 0, canvas)
    cv2.line(canvas, (0, banner_h), (w, banner_h), (0, 200, 120), 2)

    p_in = in_counts.get("person", 0)
    v_names = ("car", "bus", "truck")
    tw_names = ("motorcycle", "bicycle")
    v_in = sum(in_counts.get(k, 0) for k in v_names)
    tw_in = sum(in_counts.get(k, 0) for k in tw_names)
    v_by = {k: in_counts.get(k, 0) for k in ("car", "bus", "truck", "motorcycle", "bicycle")}

    line_part = " | ".join(
        f"{ln.name}: in {ln.total_in} out {ln.total_out}" for ln in lines
    ) or "no tripwires"
    zone_part = " | ".join(f"{z.name}: {z.total_occupancy}" for z in zones) or "no zones"

    l1 = f"FPS {fps:5.1f} | {backend}" + (f" [{mode.upper()}]" if mode else "") + f" | {source_name}"
    l2 = (f"IN-FRAME  P:{p_in:2d}  V:{v_in:2d} (car {v_by['car']} bus {v_by['bus']} truck {v_by['truck']})"
          f"  TW:{tw_in:2d} (moto {v_by['motorcycle']} bike {v_by['bicycle']})")
    l3 = f"TRIPWIRE  {line_part}"
    l4 = f"ZONES     {zone_part}    {extra}"
    p_cum = scene_counts.get("person", 0)
    v_cum = sum(scene_counts.get(k, 0) for k in v_names)
    tw_cum = sum(scene_counts.get(k, 0) for k in tw_names)
    idx = 0 if total_frames <= 0 else frame_idx
    pct = 100.0 * idx / total_frames if total_frames > 0 else 0.0
    l5 = f"SCENE     P:{p_cum:3d}  V:{v_cum:3d}  TW:{tw_cum:3d}   [{idx}/{total_frames} {pct:4.1f}%]"
    l6 = "TW=two-wheelers (e-bike/motorcycle/bicycle @ low conf)"

    colors = [(0, 230, 230), (255, 200, 80), (120, 255, 120), (200, 160, 255),
              (150, 200, 255), (150, 150, 150)]
    for i, (txt, col) in enumerate(zip((l1, l2, l3, l4, l5, l6), colors)):
        cv2.putText(canvas, txt, (12, 22 + 20 * i), FONT, 0.44, col, 1, cv2.LINE_AA)
    _progress_bar(canvas, pct / 100.0, w - 240, banner_h - 10, 228, 5)
    return canvas


def draw_help(canvas, show):
    if not show:
        return canvas
    h, w = canvas.shape[:2]
    tips = ["q skip | Esc quit | space pause | . step | t trails | h heatmap | z zones | l lines | s snapshot"]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, h - 34), (w, h - 6), (18, 20, 22), -1)
    cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)
    cv2.putText(canvas, tips[0], (12, h - 14), FONT, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas
