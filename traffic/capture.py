#!/usr/bin/env python3
"""Threaded video capture with live-safe frame policy.

Files (mp4/avi/mov/...): reader thread keeps a bounded queue so decode never blocks
the inference loop; supports seeking (`seek`) for interactive scrubbing.
Live sources (webcam index, rtsp://, http://): "latest frame wins" — stale frames are
dropped so processing runs in real time even when inference is slower than the sensor.

Concurrency: all decoder access (read/seek/release) is serialized through one lock and
frames carry a generation counter so pre-seek frames never leak into the stream after
a scrub. Releasing joins the reader first — concurrent read+release corrupts FFmpeg.
"""

import queue
import threading
import time

import cv2

FILE_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".ts", ".mpg", ".mpeg"}


class VideoCaptureThreaded:
    def __init__(self, src, queue_size: int = 8, name: str = ""):
        self.src = src
        self.name = name or str(src)
        self.is_live = self._is_live(src)
        self.stopped = False

        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {src}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.orig_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not self.is_live else -1

        self._io_lock = threading.Lock()
        self._gen = 0

        if self.is_live:
            # Small queue: if inference lags, old frames are replaced by the newest one.
            self.q: queue.Queue = queue.Queue(maxsize=2)
        else:
            self.q = queue.Queue(maxsize=max(2, queue_size))
        self.thread = threading.Thread(target=self._reader, daemon=True, name=f"capture-{self.name}")
        self.thread.start()

    @staticmethod
    def _is_live(src) -> bool:
        if isinstance(src, int) or (isinstance(src, str) and src.isdigit()):
            return True
        s = str(src).lower()
        return s.startswith(("rtsp://", "http://", "https://", "udp://", "rtp://"))

    def _reader(self):
        end_signalled = False
        while not self.stopped:
            with self._io_lock:
                gen = self._gen
                ok, frame = self.cap.read()

            if not ok:
                if self.is_live:
                    time.sleep(0.05)
                    continue
                if not end_signalled and gen == self._gen:
                    while not self.stopped:
                        try:
                            self.q.put_nowait((False, None))  # end-of-stream marker
                            end_signalled = True
                            break
                        except queue.Full:
                            time.sleep(0.01)
                if not self.is_live and not end_signalled:
                    continue  # our marker was dropped by a seek; re-emit
                time.sleep(0.02)
                continue

            if gen != self._gen or self.stopped:
                continue  # stale frame from before a seek

            if self.is_live:
                # Latest-frame policy: drop stale queued frames.
                while not self.stopped:
                    try:
                        self.q.put_nowait((True, frame))
                        break
                    except queue.Full:
                        try:
                            self.q.get_nowait()
                        except queue.Empty:
                            pass
            else:
                while not self.stopped and gen == self._gen:
                    try:
                        self.q.put((True, frame), timeout=0.05)
                        break
                    except queue.Full:
                        continue

    def read(self, timeout: float = 5.0):
        """Return (ok, frame). Blocks up to timeout; (False, None) when stream ends."""
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return (False, None)

    def seek(self, frame_idx: int):
        """Seek to an absolute frame index (file sources only)."""
        if self.is_live:
            return False
        with self._io_lock:
            self._gen += 1  # invalidate in-flight frames
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
            with self.q.mutex:
                self.q.queue.clear()
        return True

    def release(self):
        self.stopped = True
        # Join the reader BEFORE releasing the capture: concurrent read+release
        # corrupts FFmpeg's decoder state (fctx->async_lock assertion abort).
        self.thread.join(timeout=3.0)
        with self.q.mutex:
            self.q.queue.clear()
        try:
            self.cap.release()
        except Exception:
            pass
