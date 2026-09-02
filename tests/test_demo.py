#!/usr/bin/env python3
"""Demo dashboard smoke tests: every section must run non-interactively.
Run: python tests/test_demo.py"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
PASS = FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name} {detail}")


def run(section, extra=None):
    return subprocess.run([sys.executable, str(ROOT / "demo.py"), "--section", section]
                          + (extra or []), capture_output=True, text=True, cwd=ROOT,
                          timeout=300)


print("\n[Demo sections — non-interactive smoke]")
for sec, expect in [("system", "系统体检"), ("arch", "ANE 张量直连"),
                    ("modes", "eco"), ("tests", "质量门禁")]:
    p = run(sec)
    check(f"section {sec} exits 0", p.returncode == 0, p.stderr[-200:])
    check(f"section {sec} shows '{expect}'", expect in p.stdout)

p = run("tests")
check("tests section gates 114/0", "114 passed / 0 failed" in p.stdout and
      "质量门禁通过" in p.stdout, p.stdout[-200:])

p = run("bench", ["--max-frames", "30"])
check("bench section runs backends", "CoreML/ANE" in p.stdout and "FPS" in p.stdout,
      p.stderr[-200:])

print(f"\n{'=' * 40}\nRESULT: {PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
