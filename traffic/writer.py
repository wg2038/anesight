#!/usr/bin/env python3
"""Video output with Apple VideoToolbox H.264 hardware encoding.

Prefers an `ffmpeg -c:v h264_videotoolbox` rawvideo pipe (hardware encoder, H.264
yuv420p plays everywhere); falls back to OpenCV's mp4v writer when ffmpeg is missing.
"""

import subprocess
from pathlib import Path

import cv2
import numpy as np


def _probe_videotoolbox() -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=15)
        return "h264_videotoolbox" in out.stdout
    except Exception:
        return False


_VT_AVAILABLE = None  # lazy probe result cache


class VideoSink:
    def __init__(self, path, fps: float, size: tuple, encoder: str = "auto",
                 bitrate_m: int = 8):
        """encoder: 'auto' | 'videotoolbox' | 'cv2'."""
        global _VT_AVAILABLE
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.size = (int(size[0]), int(size[1]))  # (w, h)
        self.backend = None
        self._frames = 0

        if encoder in ("auto", "videotoolbox"):
            if _VT_AVAILABLE is None:
                _VT_AVAILABLE = _probe_videotoolbox()
            if _VT_AVAILABLE:
                cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                       "-f", "rawvideo", "-pix_fmt", "bgr24",
                       "-s", f"{self.size[0]}x{self.size[1]}", "-r", f"{fps:.4f}",
                       "-i", "pipe:0",
                       "-c:v", "h264_videotoolbox", "-b:v", f"{bitrate_m}M",
                       "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                       str(self.path)]
                self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                             stdout=subprocess.DEVNULL,
                                             stderr=subprocess.DEVNULL)
                self.backend = "videotoolbox"
                return
            if encoder == "videotoolbox":
                raise RuntimeError("h264_videotoolbox requested but ffmpeg lacks it")

        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"),
                                      fps, self.size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open cv2 VideoWriter for {self.path}")
        self.backend = "cv2-mp4v"

    def write(self, frame: np.ndarray):
        if self.backend == "videotoolbox":
            try:
                self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            except BrokenPipeError:
                raise RuntimeError("ffmpeg encoder died — disk full or bad args?")
        else:
            self.writer.write(frame)
        self._frames += 1

    @property
    def frames(self) -> int:
        return self._frames

    def close(self):
        if self.backend == "videotoolbox":
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()
        else:
            self.writer.release()
