#!/usr/bin/env python3
"""Alert bus: deduplicated security/ops alerts with CSV logging and webhook delivery.

Every alert carries a dedup key (e.g. "intrusion:zoneA:track17"); the same key is
suppressed for `cooldown_s` so a camera watching one intruder emits one alert, not 30.
Webhook delivery is fire-and-forget JSON POST (never blocks the pipeline, never raises).
"""

import json
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import csv


@dataclass
class Alert:
    type: str            # intrusion | loiter | proximity | slot_occupied | slot_freed | ...
    level: str           # info | warning | critical
    key: str             # dedup key
    message: str
    frame: int = 0
    meta: dict = field(default_factory=dict)


class AlertBus:
    def __init__(self, cooldown_s: float = 30.0, webhook: str | None = None, logger=None):
        self.cooldown_s = float(cooldown_s)
        self.webhook = webhook
        self.logger = logger
        self._last: dict[str, float] = {}
        self.counts: dict[str, int] = {}   # type -> emitted count

    def emit(self, atype: str, key: str, message: str, level: str = "warning",
             frame: int = 0, meta: dict | None = None, t: float | None = None) -> Alert | None:
        now = t if t is not None else time.monotonic()
        last = self._last.get(key)
        if last is not None and now - last < self.cooldown_s:
            return None
        self._last[key] = now
        alert = Alert(atype, level, key, message, frame, meta or {})
        self.counts[atype] = self.counts.get(atype, 0) + 1

        tag = "ALERT" if level != "critical" else "CRITICAL"
        print(f"  >>> [{tag}] {message}", flush=True)
        if self.logger is not None:
            self.logger.log_alert(alert)
        if self.webhook:
            self._post(alert)
        return alert

    def _post(self, alert: Alert):
        payload = json.dumps({
            "type": alert.type, "level": alert.level, "message": alert.message,
            "frame": alert.frame, "ts": datetime.now().isoformat(), **alert.meta,
        }).encode()
        try:
            req = urllib.request.Request(self.webhook, data=payload,
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=1.5).close()
        except Exception:
            pass  # monitoring must never die because a webhook endpoint is down
