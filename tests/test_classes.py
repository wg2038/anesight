#!/usr/bin/env python3
"""Class taxonomy integrity tests (two-wheeler / e-bike handling). Run: python tests/test_classes.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from traffic.annotate import CLASS_META, TWO_WHEELER_NAMES

# Keep these in sync with run_detector.SUPPORTED (import would pull cv2/torch heavy deps)
EXPECTED_SUPPORTED = [0, 1, 2, 3, 5, 7]

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


print("\n[Class taxonomy]")
check("COCO person=0 car=2 motorcycle=3 bus=5 truck=7 bicycle=1 covered",
      all(c in CLASS_META for c in EXPECTED_SUPPORTED),
      str(sorted(CLASS_META)))

tw = {c for c, m in CLASS_META.items() if m["name"] in TWO_WHEELER_NAMES}
check("two-wheelers = motorcycle + bicycle", tw == {1, 3}, str(tw))

for c in (1, 3):
    check(f"class {c} has low-conf floor", CLASS_META[c].get("min_conf", 1.0) <= 0.35,
          str(CLASS_META[c].get("min_conf")))

for c, m in CLASS_META.items():
    check(f"class {c} ({m['name']}) has color + zh",
          "color" in m and "zh" in m and "两轮车" in m["zh"] if c in (1, 3) else "color" in m and "zh" in m)

names = [m["name"] for m in CLASS_META.values()]
check("display names unique", len(names) == len(set(names)))

print(f"\n{'=' * 40}\nRESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
