#!/usr/bin/env python3
"""Performance mode resolution tests. Run: python tests/test_perf.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from traffic.perf import PROFILES, get_profile, set_qos

sys.path.insert(0, str(Path(__file__).parent.parent))
from run_detector import build_argparser

PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def resolve(argv):
    """Replicate run_detector.main's mode resolution (without side effects)."""
    args = build_argparser().parse_args(argv)
    p = get_profile(args.mode)
    model = args.model if args.model is not None else p.model
    imgsz = args.imgsz if args.imgsz is not None else p.imgsz
    realtime = args.realtime if args.realtime is not None else p.realtime
    return model, imgsz, realtime, p


print("\n[Perf profiles]")
check("three modes defined", set(PROFILES) == {"eco", "balanced", "turbo"})

print("\n[eco defaults]")
m, i, rt, p = resolve(["test.mp4", "--mode", "eco"])
check("eco uses yolov8n", m == "yolov8n.pt", m)
check("eco paces real-time", rt is True)
check("eco utility QoS", p.qos == "utility")
check("eco display 15fps", p.display_fps == 15.0)

print("\n[turbo defaults]")
m, i, rt, p = resolve(["test.mp4", "--mode", "turbo"])
check("turbo uses yolov8m", m == "yolov8m.pt", m)
check("turbo big canvas", i == "704x1280", i)
check("turbo not paced", rt is False)
check("turbo interactive QoS", p.qos == "user_interactive")
check("turbo bitrate 12M", p.bitrate_m == 12)

print("\n[explicit flags beat mode]")
m, i, rt, p = resolve(["test.mp4", "--mode", "eco", "--model", "yolov8s.pt"])
check("explicit model wins over eco", m == "yolov8s.pt", m)
m, i, rt, p = resolve(["test.mp4", "--mode", "turbo", "--imgsz", "512x960"])
check("explicit imgsz wins over turbo", i == "512x960", i)
m, i, rt, p = resolve(["test.mp4", "--mode", "eco", "--realtime"])
check("--realtime under eco stays True", rt is True)
args = build_argparser().parse_args(["test.mp4", "--mode", "turbo", "--realtime"])
check("turbo + explicit --realtime paces", args.realtime is True)

print("\n[default mode]")
m, i, rt, p = resolve(["test.mp4"])
check("default = balanced s@512x960", m == "yolov8s.pt" and i == "512x960" and rt is False)

print("\n[QoS]")
rc = set_qos("utility")
check("set utility QoS on darwin", rc is True or sys.platform != "darwin")
rc = set_qos("bogus")
check("bogus QoS refused", rc is False)
rc = set_qos(None)
check("None QoS no-op", rc is False)

print(f"\n{'=' * 40}\nRESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
