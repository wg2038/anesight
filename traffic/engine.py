#!/usr/bin/env python3
"""Inference engine: optimal backend selection and tracking for Apple Silicon.

Backend priority (auto):
  1. CUDA       — if an NVIDIA GPU is visible (non-Apple hosts)
  2. CoreML/ANE — Apple Silicon: fp16 mlprogram with fused NMS, exported once and cached.
                  The exported package is then spec-patched to take a raw NCHW float
                  tensor instead of a PIL image: this bypasses coremltools' PIL
                  image pipeline (several ms of copies/encodes per frame) and feeds
                  the Neural Engine directly (~4 ms/frame for YOLOv8s at 512x960).
                  Rectangular canvas (e.g. 512x960) matches landscape video and is
                  ~2.6x faster than a square canvas on the ANE.
  3. MPS        — PyTorch Metal fallback (fp16)
  4. CPU        — last resort

Tracking uses a manual per-stream BYTETracker so several streams each keep independent
track state while sharing one compiled model — `model.track(persist=True)` cannot do
this. The tracker is fed the full low-threshold confidence pool (ByteTrack's two-stage
association needs it); display/analytics confidence filtering happens afterwards.
"""

import ast
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
import yaml

# coremltools logs a version-mismatch warning on every import when torch is newer
# than the version it was validated against. We validated this pairing empirically
# (ANE fp16 vs MPS fp32: mean IoU 0.983, identical counts), so keep output clean.
logging.getLogger("coremltools").setLevel(logging.ERROR)


@dataclass
class Detection:
    """A single tracked detection in original frame pixel coordinates."""
    xyxy: np.ndarray   # float32 (4,)
    conf: float
    cls_id: int
    track_id: int

    @property
    def cx(self) -> float:
        return float((self.xyxy[0] + self.xyxy[2]) / 2)

    @property
    def cy(self) -> float:
        return float((self.xyxy[1] + self.xyxy[3]) / 2)

    @property
    def bottom_center(self):
        return (float((self.xyxy[0] + self.xyxy[2]) / 2), float(self.xyxy[3]))


class Backend:
    CUDA = "cuda"
    COREML = "coreml"
    MPS = "mps"
    CPU = "cpu"


def select_backend(pref: str = "auto") -> str:
    """Pick the fastest available backend, honouring an explicit preference."""
    if pref not in (None, "auto"):
        return pref
    if torch.cuda.is_available():
        return Backend.CUDA
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            import coremltools  # noqa: F401
            import platform
            if platform.system() == "Darwin":
                return Backend.COREML
        except ImportError:
            pass
        return Backend.MPS
    return Backend.CPU


# --------------------------------------------------------------------- CoreML prep

def _patch_spec_to_tensor(src_pkg: Path, dst_pkg: Path, imgsz) -> Path:
    """Copy an exported mlpackage and rewrite its input from ImageType to a float
    tensor (NCHW 0-255 RGB). The /255 scale is baked into the mlprogram graph, and
    the letterbox ratio/pad are inverted outside, so the tensor goes straight to ANE.
    """
    from coremltools.proto import Model_pb2

    shutil.copytree(src_pkg, dst_pkg, dirs_exist_ok=True)
    import coremltools as ct
    spec = ct.models.MLModel(str(src_pkg)).get_spec()
    h, w = imgsz

    def tensor_desc(name="image"):
        d = Model_pb2.FeatureDescription()
        d.name = name
        arr = d.type.multiArrayType
        arr.shape.extend([1, 3, h, w])
        arr.dataType = Model_pb2.ArrayFeatureType.FLOAT32
        return d

    inputs = list(spec.description.input)
    if inputs[0].type.WhichOneof("Type") != "imageType":
        raise ValueError("expected an image input to patch")
    inputs[0] = tensor_desc(inputs[0].name)
    del spec.description.input[:]
    spec.description.input.extend(inputs)

    subs = list(spec.pipeline.models) if spec.HasField("pipeline") else []
    if subs:
        sins = list(subs[0].description.input)
        sins[0] = tensor_desc(sins[0].name)
        del subs[0].description.input[:]
        subs[0].description.input.extend(sins)

    (dst_pkg / "Data/com.apple.CoreML/model.mlmodel").write_bytes(spec.SerializeToString())
    return dst_pkg


def _ensure_coreml_tensor_model(weights: str, imgsz):
    """Export once via Ultralytics (fp16, fused NMS), then spec-patch to tensor input.

    Returns (package_path, is_tensor). Falls back to the standard image-input package
    (PIL path) when the spec patch is not applicable.
    """
    h, w = int(imgsz[0]), int(imgsz[1])
    weights_path = Path(weights)
    cached_tensor = weights_path.parent / f"{weights_path.stem}_{h}x{w}_ane_t.mlpackage"
    cached_plain = weights_path.parent / f"{weights_path.stem}_{h}x{w}_ane.mlpackage"

    if cached_tensor.exists():
        return cached_tensor, True
    if cached_plain.exists():
        return cached_plain, False

    print(f"[ENGINE] Exporting CoreML fp16 ANE model ({h}x{w}) — one-time cost...", flush=True)
    from ultralytics import YOLO
    tmp = Path(YOLO(weights).export(format="coreml", quantize="fp16", imgsz=[h, w],
                                    nms=True, device="cpu"))
    try:
        _patch_spec_to_tensor(tmp, cached_tensor, (h, w))
        print(f"[ENGINE] Cached tensor-input CoreML model -> {cached_tensor.name}", flush=True)
        return cached_tensor, True
    except Exception as e:
        print(f"[ENGINE] Tensor-input patch failed ({e}); using standard image-input model.")
        if cached_plain.exists():
            shutil.rmtree(cached_plain, ignore_errors=True)
        tmp.rename(cached_plain)
        return cached_plain, False
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


class _CoreMLTensorRunner:
    """Direct coremltools runner: letterbox -> NCHW float tensor -> fused-NMS model."""

    def __init__(self, pkg_path: Path, imgsz, classes=None, low_conf=0.20):
        import coremltools as ct
        self.model = ct.models.MLModel(str(pkg_path), compute_units=ct.ComputeUnit.CPU_AND_NE)
        spec = self.model.get_spec()
        self.input_name = spec.description.input[0].name
        self.h, self.w = int(imgsz[0]), int(imgsz[1])
        self.classes = classes
        self.low_conf = low_conf
        self.pad_top = 0
        self.pad_left = 0
        self.ratio = 1.0
        self._canvas = np.full((self.h, self.w, 3), 114, dtype=np.float32)
        self.names = self._read_names()

    def _read_names(self) -> dict:
        """Class-id -> name from model metadata (enables custom non-COCO models)."""
        try:
            raw = self.model.user_defined_metadata.get("names")
            if raw:
                return {int(k): v for k, v in ast.literal_eval(str(raw)).items()}
        except Exception:
            pass
        return {}

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        oh, ow = frame.shape[:2]
        r = min(self.h / oh, self.w / ow)
        nh, nw = int(round(oh * r)), int(round(ow * r))
        self.ratio, self.pad_top, self.pad_left = r, (self.h - nh) // 2, (self.w - nw) // 2
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = self._canvas
        canvas[:] = 114
        canvas[self.pad_top:self.pad_top + nh, self.pad_left:self.pad_left + nw] = resized
        rgb = canvas[:, :, ::-1]
        return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])

    def predict_dets(self, frame: np.ndarray) -> np.ndarray:
        """Returns (N,6) float32 [x1,y1,x2,y2,conf,cls] in ORIGINAL frame coords."""
        tensor = self._preprocess(frame)
        y = self.model.predict({self.input_name: tensor})
        conf = np.asarray(y["confidence"], dtype=np.float32)
        if conf.size == 0:
            return np.zeros((0, 6), dtype=np.float32)
        coords = np.asarray(y["coordinates"], dtype=np.float32).reshape(-1, 4)  # xywh normalized
        conf = conf.reshape(-1, conf.shape[-1])
        cls_ids = conf.argmax(1)
        scores = conf[np.arange(len(cls_ids)), cls_ids]

        lb = coords * [self.w, self.h, self.w, self.h]
        # xywh(center) -> xyxy
        xyxy = np.empty_like(lb)
        xyxy[:, 0] = lb[:, 0] - lb[:, 2] / 2
        xyxy[:, 1] = lb[:, 1] - lb[:, 3] / 2
        xyxy[:, 2] = lb[:, 0] + lb[:, 2] / 2
        xyxy[:, 3] = lb[:, 1] + lb[:, 3] / 2
        # invert letterbox back to original frame pixels
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - self.pad_left) / self.ratio
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - self.pad_top) / self.ratio
        oh, ow = frame.shape[:2]
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, ow - 1)
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, oh - 1)

        mask = scores >= self.low_conf
        if self.classes is not None:
            mask &= np.isin(cls_ids, self.classes)
        if not mask.any():
            return np.zeros((0, 6), dtype=np.float32)
        dets = np.empty((int(mask.sum()), 6), dtype=np.float32)
        sel_xyxy = xyxy[mask]
        dets[:, :4] = sel_xyxy
        dets[:, 4] = scores[mask]
        dets[:, 5] = cls_ids[mask]
        return dets


# --------------------------------------------------------------------- engine

class InferenceEngine:
    """Unified detection+tracking engine across CoreML/ANE, MPS, CUDA and CPU."""

    def __init__(self, weights: str = "yolov8s.pt",
                 imgsz=None,
                 backend: str = "auto",
                 classes: list | None = None,
                 tracker_cfg: str = "custom_bytetrack.yaml",
                 display_conf: float = 0.25,
                 iou: float = 0.7):
        self.backend_pref = backend
        self.backend = select_backend(backend)
        self.imgsz = list(imgsz) if imgsz else [960, 960]
        self.classes = classes
        self.display_conf = display_conf
        self.iou = iou
        self.weights = weights
        self.infer_ms = 0.0
        self.pre_ms = 0.0
        self._track_low = 0.20
        self._runner = None          # tensor-input CoreML path
        self._ultra_model = None     # fallback path
        self._trackers = {}

        if self.backend == Backend.COREML:
            pkg, is_tensor = _ensure_coreml_tensor_model(weights, self.imgsz)
            if is_tensor:
                try:
                    self._runner = _CoreMLTensorRunner(pkg, self.imgsz, classes=classes,
                                                       low_conf=self._track_low)
                except Exception as e:
                    print(f"[ENGINE] Direct ANE runner unavailable ({e}); using Ultralytics CoreML path.")
            if self._runner is None:
                from ultralytics import YOLO
                self._ultra_model = YOLO(str(pkg))
        else:
            from ultralytics import YOLO
            self._ultra_model = YOLO(weights)
            self.device = self.backend if self.backend != Backend.CPU else "cpu"

        self.tracker_cfg = tracker_cfg

    def _load_tracker_args(self) -> SimpleNamespace:
        cfg_path = Path(self.tracker_cfg)
        if not cfg_path.exists():
            import ultralytics
            cfg_path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / "bytetrack.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        args = SimpleNamespace(**cfg)
        self._track_low = float(getattr(args, "track_low_thresh", 0.1))
        return args

    def tracker_for(self, stream_id: int = 0):
        """Get or lazily create an independent BYTETracker for a stream."""
        if stream_id not in self._trackers:
            from ultralytics.trackers.byte_tracker import BYTETracker
            self._trackers[stream_id] = BYTETracker(self._load_tracker_args())
        return self._trackers[stream_id]

    def track(self, frame: np.ndarray, stream_id: int = 0) -> list[Detection]:
        """Detect + track one frame. Each stream_id keeps independent tracker state."""
        t0 = time.perf_counter()
        tracker = self.tracker_for(stream_id)
        if self._runner is not None:
            dets_np = self._runner.predict_dets(frame)   # raw (N,6) ndarray
            from ultralytics.engine.results import Boxes
            dets_in = Boxes(dets_np, orig_shape=(frame.shape[0], frame.shape[1]))
        else:
            # Ultralytics-native path: pass the Boxes object the tracker expects.
            t1 = time.perf_counter()
            dets_in = self._predict(frame)[0].boxes.cpu().numpy()
            self.pre_ms = 0.9 * self.pre_ms + 0.1 * (time.perf_counter() - t1) * 1000
        self.infer_ms = 0.9 * self.infer_ms + 0.1 * (time.perf_counter() - t0) * 1000

        tracked = tracker.update(dets_in, frame)
        # tracked: (N,8): x1,y1,x2,y2,track_id,conf,cls,idx
        out = []
        for row in tracked:
            out.append(Detection(
                xyxy=np.asarray(row[:4], dtype=np.float32),
                conf=float(row[5]), cls_id=int(row[6]), track_id=int(row[4])))
        return out

    def _predict(self, frame: np.ndarray):
        kwargs = dict(imgsz=self.imgsz, conf=self._track_low, iou=self.iou,
                      classes=self.classes, verbose=False)
        if self.backend == Backend.COREML:
            return self._ultra_model.predict(frame, **kwargs)
        half = self.backend == Backend.MPS
        return self._ultra_model.predict(frame, device=self.device, half=half, **kwargs)

    def describe(self) -> str:
        if self.backend == Backend.COREML:
            path = "direct-tensor" if self._runner is not None else "ultralytics"
            return f"CoreML/ANE {self.imgsz[0]}x{self.imgsz[1]} fp16 ({path})"
        if self.backend == Backend.MPS:
            return "MPS fp16"
        return self.backend.upper()

    @property
    def class_names(self) -> dict:
        """Class-id -> name for the loaded model (COCO or custom)."""
        if self._runner is not None:
            return dict(self._runner.names)
        if self._ultra_model is not None:
            return dict(getattr(self._ultra_model, "names", {}) or {})
        return {}
