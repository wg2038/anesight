#!/usr/bin/env python3
"""Unit tests for traffic.analytics + engine helpers. Run: python tests/test_analytics.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from traffic.analytics import (DirectionalLine, MotionHeatmap, PolygonZone,
                               SpeedEstimator, SuppressNested, TrackBook)
from traffic.engine import Detection


def det(x1, y1, x2, y2, conf=0.8, cls=2, tid=1):
    return Detection(np.array([x1, y1, x2, y2], dtype=np.float32), conf, cls, tid)


PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


# ---------------------------------------------------------------- DirectionalLine
print("\n[DirectionalLine]")

def walk_line(line, path, cls_name="car", tid=7):
    events = []
    for a, b in zip(path, path[1:]):
        ev = line.update(tid, a, b, cls_name)
        if ev:
            events.append(ev)
    return events

# Horizontal line y=100, p1->p2 left-to-right: left normal points up (negative y).
ln = DirectionalLine((0, 100), (100, 100), "t")
side_above = ln.signed_side((50, 50))
side_below = ln.signed_side((50, 150))
check("side above is negative(left)", side_above == -1, f"got {side_above}")
check("side below is positive(right)", side_below == 1, f"got {side_below}")

ln = DirectionalLine((0, 100), (100, 100), "t")
evs = walk_line(ln, [(50, 150), (50, 90), (50, 50)])  # bottom -> top = right->left = 'out'?? left is -1
# crossing from below(+1) to above(-1): s_curr < s_prev => direction 'out'
check("crossing up counts", len(evs) == 1 and evs[0]["direction"] == "out", str(evs))
check("total_out incremented", ln.total_out == 1)

ln = DirectionalLine((0, 100), (100, 100), "t")
evs = walk_line(ln, [(50, 50), (50, 120), (50, 150)])  # above -> below = in
check("crossing down = in", len(evs) == 1 and evs[0]["direction"] == "in", str(evs))
check("per-class counted", ln.counts_in["car"] == 1)

ln = DirectionalLine((0, 100), (100, 100), "t")
evs = walk_line(ln, [(50, 150), (50, 145), (50, 148), (50, 151)])  # same side, jitter
check("no false crossing on jitter", len(evs) == 0, str(evs))

ln = DirectionalLine((0, 100), (100, 100), "t")
evs = walk_line(ln, [(50, 150), (50, 50), (50, 150), (50, 50)])  # ping-pong
check("ping-pong counted once per direction",
      len(evs) == 2 and [e["direction"] for e in evs] == ["out", "in"], str(evs))
check("in/out ids unique", len(ln.in_ids) == 1 and len(ln.out_ids) == 1)

ln = DirectionalLine((0, 100), (100, 100), "t", count_once=False)
evs = walk_line(ln, [(50, 150), (50, 50), (50, 150), (50, 50)])
check("ping-pong all events when count_once=False",
      len(evs) == 3 and [e["direction"] for e in evs] == ["out", "in", "out"], str(evs))

ln = DirectionalLine((0, 100), (100, 100), "t")
evs = walk_line(ln, [(50, 150), (50, 98), (50, 102), (50, 96)])  # jitter across line
check("hysteresis kills jitter on line", len(evs) == 0, str(evs))

ln = DirectionalLine((0, 100), (100, 100), "t", classes=["person"])
evs = walk_line(ln, [(50, 50), (50, 150)], cls_name="car")
check("class filter blocks counting", len(evs) == 0, str(evs))
evs = walk_line(ln, [(60, 50), (60, 150)], cls_name="person")
check("class filter allows matching class", len(evs) == 1 and evs[0]["direction"] == "in", str(evs))

# Vertical line: left of p1->p2 (pointing down) is the -x side
ln = DirectionalLine((100, 0), (100, 100), "v")
check("vertical left side +", ln.signed_side((50, 50)) == 1, f"got {ln.signed_side((50,50))}")
evs = walk_line(ln, [(150, 50), (80, 50)])  # right(+x,-1) -> left = in
check("vertical crossing in", evs and evs[0]["direction"] == "in", str(evs))

# track starting exactly on line shouldn't count immediately
ln = DirectionalLine((0, 100), (100, 100), "t")
evs = walk_line(ln, [(50, 100), (50, 150)])
check("spawn-on-line not counted", len(evs) == 0, str(evs))

# ---------------------------------------------------------------- PolygonZone
print("\n[PolygonZone]")
zone = PolygonZone([(0, 0), (100, 0), (100, 100), (0, 100)], "z")
check("inside", zone.contains((50, 50)))
check("outside", not zone.contains((150, 50)))
d1, d2 = det(40, 40, 60, 60, tid=1), det(150, 150, 160, 160, tid=2)
zone.update([("car", 1, (d1.cx, d1.cy)), ("car", 2, (d2.cx, d2.cy))])
check("occupancy=1", zone.total_occupancy == 1)
check("visited unique=1", len(zone.visited_ids) == 1)
zone.update([])
check("occupancy resets", zone.total_occupancy == 0 and len(zone.visited_ids) == 1)

# ---------------------------------------------------------------- SuppressNested
print("\n[SuppressNested]")
sup = SuppressNested(0.7)
dets = [det(0, 0, 100, 100, cls=7, tid=1), det(10, 60, 90, 95, cls=2, tid=2),  # 80% contained
        det(200, 0, 300, 100, cls=2, tid=3)]
kept = sup(dets)
check("nested suppressed", len(kept) == 2 and {d.track_id for d in kept} == {1, 3},
      str([d.track_id for d in kept]))

sup = SuppressNested(0.7)
dets = [det(0, 0, 100, 100, cls=7, tid=1), det(50, 50, 180, 180, cls=2, tid=2)]  # 25% contained
kept = sup(dets)
check("overlapping-but-not-nested kept", len(kept) == 2, str(len(kept)))

# ---------------------------------------------------------------- TrackBook
print("\n[TrackBook]")
book = TrackBook()
c = book.stabilize_class(5, 2)
for _ in range(9):
    c = book.stabilize_class(5, 7)  # flicker
check("majority vote wins", c == 7, f"got {c}")
d = det(0, 0, 10, 10, tid=9)
for i in range(5):
    new, prev, cur = book.update_track(d, "car")
check("confirm after 5 hits", new and book.scene_counts["car"] == 1)
check("trail populated", len(book.trails[9]) == 5)

# ---------------------------------------------------------------- MotionHeatmap
print("\n[MotionHeatmap]")
hm = MotionHeatmap(200, 200)
hm.update([det(90, 90, 110, 110)])
check("heatmap active", hm.buf.max() > 0 and hm.buf[110, 100] > 0)
hm.update([det(90, 90, 110, 110)])
hm.update([det(90, 90, 110, 110)])
peak = hm.buf[110, 100]
hm2 = MotionHeatmap(200, 200)
hm2.update([det(90, 90, 110, 110)])
check("accumulation grows", peak > hm2.buf[110, 100], f"{peak} vs {hm2.buf[110,100]}")
ov = hm.overlay(np.zeros((200, 200, 3), dtype=np.uint8))
check("overlay has color", ov.sum() > 0)
hm.update([])  # decay
check("decay shrinks", hm.buf[110, 100] < peak)

# ---------------------------------------------------------------- SpeedEstimator
print("\n[SpeedEstimator]")
se = SpeedEstimator(window_s=1.0, mpp=0.05)
se.update(1, 0, 0, t=0.0)
se.update(1, 100, 0, t=1.0)  # 100 px/s * 0.05 m/px = 5 m/s = 18 km/h
px, kmh = se.latest[1]
check("px/s correct", abs(px - 100) < 1e-6, str(px))
check("km/h correct", abs(kmh - 18.0) < 0.01, str(kmh))
se.update(2, 0, 0, t=0.0)
check("no speed w/ single obs", se.latest[2] == (None, None))

# ---------------------------------------------------------------- Detection props
print("\n[Detection]")
d = det(0, 0, 20, 10)
check("cx/cy", d.cx == 10 and d.cy == 5, f"got {d.cx},{d.cy}")
check("bottom_center", d.bottom_center == (10.0, 10.0))

print(f"\n{'=' * 40}\nRESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
