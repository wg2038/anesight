#!/usr/bin/env python3
"""Scenario logic tests: parking slots, factory security, alert bus. Run: python tests/test_scenarios.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from traffic.alerts import AlertBus
from traffic.engine import Detection
from traffic.factory import (ProximityMonitor, RestrictedZone, is_armed,
                             parse_armed_hours)
from traffic.parking import SlotManager, parse_slot, parse_slot_grid

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def det(x1, y1, x2, y2, conf=0.8, cls=2, tid=1):
    return Detection(np.array([x1, y1, x2, y2], dtype=np.float32), conf, cls, tid)


print("\n[Parking slot parsing]")
s = parse_slot("100,200,160,310:A3")
check("rect slot id", s.slot_id == "A3")
check("rect slot polygon 4 pts", s.polygon.shape == (4, 2))
s2 = parse_slot("100,200,160,310")
check("default id S1", s2.slot_id == "S1")
grid = parse_slot_grid("100,400,60,110,8,2")
check("grid generates 16 slots", len(grid) == 16)
check("grid ids A1..B8 (row letters default)", grid[0].slot_id == "A1" and grid[7].slot_id == "A8"
      and grid[8].slot_id == "B1" and grid[15].slot_id == "B8")
grid2 = parse_slot_grid("100,400,60,110,4,2:P")
check("grid prefix = continuous ids", grid2[0].slot_id == "P1" and grid2[7].slot_id == "P8")
check("grid layout spans width", grid[7].polygon[1][0] - grid[0].polygon[0][0] == 480)
try:
    parse_slot_grid("1,2,3")
    check("bad grid rejected", False)
except ValueError:
    check("bad grid rejected", True)

print("\n[Parking occupancy]")
mgr = SlotManager([parse_slot(f"{100 + i * 70},200,{160 + i * 70},310:{chr(65 + i)}1")
                   for i in range(3)], stability=3)
car_in_slot = det(105, 210, 155, 300)          # fully inside slot 0
car_between = det(500, 210, 560, 300)          # outside all slots
empty = []
mgr.update([(car_in_slot, "car"), (car_between, "car")], t=0.0)
mgr.update([(car_in_slot, "car")], t=1.0)
evs = mgr.update([(car_in_slot, "car")], t=2.0)   # 3rd consecutive frame → flip
check("slot flips after stability frames",
      len(evs) == 1 and evs[0]["event"] == "slot_occupied" and evs[0]["slot"] == "A1", str(evs))
occ, tot = mgr.occupancy()
check("occupancy 1/3", occ == 1 and tot == 3)
check("dwell accumulates", mgr.by_id["A1"].dwell_s(62.0) == 60.0)
evs = mgr.update(empty, t=63.0)
evs += mgr.update(empty, t=64.0)
evs += mgr.update(empty, t=65.0)
check("slot frees after car leaves",
      any(e["event"] == "slot_freed" and e["dwell_s"] == 63.0 for e in evs), str(evs))
check("dwell resets on free", mgr.by_id["A1"].dwell_s(70.0) == 0.0)

# swap: another car replaces the occupant
mgr2 = SlotManager([parse_slot("0,0,100,100:P1")], stability=1)
mgr2.update([(det(5, 5, 95, 95, tid=7), "car")], t=0)
evs = mgr2.update([(det(5, 5, 95, 95, tid=9), "car")], t=1)
check("swap frees old + occupies new",
      any(e["event"] == "slot_freed" and e["track_id"] == 7 for e in evs)
      and any(e["event"] == "slot_occupied" and e["track_id"] == 9 for e in evs), str(evs))

# partial overlap matches by ratio (60% of box in slot)
mgr3 = SlotManager([parse_slot("100,0,200,100:R")], stability=1, match_ratio=0.5)
half = det(60, 0, 160, 100)  # 100 of 160 px width inside → 62.5%
evs = mgr3.update([(half, "car")], t=0)
check("partial overlap occupies", any(e["event"] == "slot_occupied" for e in evs), str(evs))

# motorcycles count as parking vehicles too (e-bike parking)
mgr4 = SlotManager([parse_slot("0,0,100,100:M")], stability=1)
moto = det(10, 10, 60, 90, cls=3)
evs = mgr4.update([(moto, "motorcycle")], t=0)
check("motorcycle occupies slot", any(e["event"] == "slot_occupied" for e in evs))
person = det(10, 10, 60, 90, cls=0)
mgr5 = SlotManager([parse_slot("0,0,100,100:P")], stability=1)
evs = mgr5.update([(person, "person")], t=0)
check("person never occupies slot", not evs)

print("\n[Restricted zones / intrusion / loitering]")
rz = RestrictedZone([(0, 0), (200, 0), (200, 200), (0, 200)], "vault", loiter_sec=10)
d = det(50, 50, 150, 150, cls=0, tid=42)
a1 = rz.update([("person", 42, (100, 100), d.xyxy)], t=0.0)
check("entry raises intrusion", len(a1) == 1 and a1[0][0] == "intrusion"
      and "vault" in a1[0][2] and a1[0][3] == "critical", str(a1))
a2 = rz.update([("person", 42, (100, 100), d.xyxy)], t=5.0)
check("no repeat before loiter threshold", len(a2) == 0, str(a2))
a3 = rz.update([("person", 42, (100, 100), d.xyxy)], t=11.0)
check("loiter fires after 10s", len(a3) == 1 and a3[0][0] == "loiter", str(a3))
a4 = rz.update([("person", 42, (100, 100), d.xyxy)], t=20.0)
check("loiter fires once", len(a4) == 0, str(a4))
a5 = rz.update([], t=30.0)
a6 = rz.update([("person", 42, (100, 100), d.xyxy)], t=31.0)
check("re-entry re-alerts after leaving", a6 and a6[0][0] == "intrusion", str(a6))
car = det(50, 50, 150, 150, cls=2, tid=7)
a7 = rz.update([("car", 7, (100, 100), car.xyxy)], t=40.0)
check("car not in trigger classes", len(a7) == 0, str(a7))
# outside zone
far = det(500, 500, 600, 600, cls=0, tid=8)
a8 = rz.update([("person", 8, (550, 550), far.xyxy)], t=50.0)
check("person outside zone ignored", len(a8) == 0, str(a8))

print("\n[Proximity monitor]")
pm = ProximityMonitor(threshold_px=100)
person_d = det(300, 200, 360, 300, cls=0, tid=1)
truck_d = det(380, 150, 600, 320, cls=7, tid=2)
evs = pm.update([("person", 1, (330, 300), person_d.xyxy),
                 ("truck", 2, (490, 235), truck_d.xyxy)])
check("close person+truck alerts", len(evs) == 1 and evs[0][0] == "proximity", str(evs))
far_person = det(900, 200, 960, 300, cls=0, tid=1)
evs = pm.update([("person", 1, (930, 300), far_person.xyxy),
                 ("truck", 2, (490, 235), truck_d.xyxy)])
check("far person no alert", len(evs) == 0, str(evs))
evs = pm.update([("car", 5, (330, 250), det(300, 200, 360, 300, cls=2).xyxy)])
check("vehicle-vehicle ignored", len(evs) == 0, str(evs))

print("\n[Armed hours]")
check("always on by default", is_armed(None))
check("8-18 armed at 12:00", is_armed((8, 18), now=__import__("datetime").datetime(2026, 1, 1, 12, 0)))
check("8-18 disarmed at 20:00", not is_armed((8, 18), now=__import__("datetime").datetime(2026, 1, 1, 20, 0)))
check("22-6 overnight armed at 3:00", is_armed((22, 6), now=__import__("datetime").datetime(2026, 1, 1, 3, 0)))
check("22-6 armed at 23:00", is_armed((22, 6), now=__import__("datetime").datetime(2026, 1, 1, 23, 0)))
check("22-6 disarmed at 12:00", not is_armed((22, 6), now=__import__("datetime").datetime(2026, 1, 1, 12, 0)))
check("parse '9-17'", parse_armed_hours("9-17") == (9, 17))
try:
    parse_armed_hours("25-30")
    check("invalid hours rejected", False)
except ValueError:
    check("invalid hours rejected", True)

print("\n[AlertBus cooldown + webhook suppression]")
bus = AlertBus(cooldown_s=30.0)
a1 = bus.emit("intrusion", "z1:t1", "msg1", t=0.0)
a2 = bus.emit("intrusion", "z1:t1", "msg1 again", t=10.0)   # within cooldown
a3 = bus.emit("intrusion", "z1:t2", "different key", t=10.0)
a4 = bus.emit("intrusion", "z1:t1", "after cooldown", t=31.0)
check("first alert passes", a1 is not None)
check("same key suppressed in cooldown", a2 is None)
check("different key passes", a3 is not None)
check("same key passes after cooldown", a4 is not None)
check("counts tracked", bus.counts["intrusion"] == 3, str(bus.counts))

print(f"\n{'=' * 40}\nRESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
