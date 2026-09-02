#!/usr/bin/env python3
"""
download_samples.py
Automatically download and validate open-source sample test videos containing
pedestrians and vehicles for YOLOv8 object detection verification.
"""

import os
import sys
import time
import requests
import cv2

VIDEO_SOURCES = [
    {
        "filename": "samples/video1.mp4",
        "description": "Pedestrians, Bicycles, and Cars (Mixed Traffic Scene)",
        "urls": [
            "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/person-bicycle-car-detection.mp4",
            "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4",
        ]
    },
    {
        "filename": "samples/video2.mp4",
        "description": "Street Car Flow & Road Scene",
        "urls": [
            "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/car-detection.mp4",
            "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4",
        ]
    }
]


def download_file(urls, target_path, expected_min_bytes=500_000):
    """Download a file from a list of candidate URLs with streaming and retry."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    if os.path.exists(target_path) and os.path.getsize(target_path) >= expected_min_bytes:
        print(f"[SKIP] '{target_path}' already exists ({os.path.getsize(target_path):,} bytes).")
        return True

    for url in urls:
        print(f"[DOWNLOAD] Fetching {url} -> {target_path} ...")
        try:
            start_time = time.time()
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total_length = int(r.headers.get("content-length", 0))
                downloaded = 0
                with open(target_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=128 * 1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_length > 0:
                                percent = (downloaded / total_length) * 100
                                sys.stdout.write(f"\r  Progress: {percent:.1f}% ({downloaded / 1024 / 1024:.2f} MB)")
                                sys.stdout.flush()

            elapsed = time.time() - start_time
            file_size = os.path.getsize(target_path)
            print(f"\n[DONE] Saved '{target_path}' ({file_size / 1024 / 1024:.2f} MB in {elapsed:.2f}s)")

            if file_size >= expected_min_bytes:
                return True
            else:
                print(f"[WARN] File size ({file_size} bytes) too small, trying next URL...")
        except Exception as e:
            print(f"\n[ERROR] Failed to download from {url}: {e}")
            if os.path.exists(target_path):
                os.remove(target_path)

    return False


def validate_video(file_path):
    """Validate video integrity using OpenCV."""
    print(f"[VALIDATE] Checking video integrity: {file_path}")
    if not os.path.exists(file_path):
        print(f"  [FAIL] File does not exist: {file_path}")
        return False

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        print(f"  [FAIL] OpenCV could not open {file_path}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"  [FAIL] OpenCV failed to decode the first frame of {file_path}")
        return False

    print(f"  [SUCCESS] Decoded successfully!")
    print(f"    - Resolution : {width}x{height}")
    print(f"    - FPS        : {fps:.2f}")
    print(f"    - Frames     : {frame_count}")
    print(f"    - Duration   : {duration:.2f}s")
    print(f"    - Channels   : {frame.shape[2]} (RGB/BGR)")
    return True


def main():
    print("=" * 60)
    print(" YOLOv8 Video Test Assets Downloader & Validator")
    print("=" * 60)

    all_valid = True
    for item in VIDEO_SOURCES:
        path = item["filename"]
        desc = item["description"]
        print(f"\nTarget: {path} ({desc})")
        ok = download_file(item["urls"], path)
        if not ok:
            print(f"[CRITICAL] Unable to download {path}")
            all_valid = False
            continue

        valid = validate_video(path)
        if not valid:
            all_valid = False

    print("\n" + "=" * 60)
    if all_valid:
        print("[SUMMARY] All test video assets are ready and verified!")
        return 0
    else:
        print("[SUMMARY] Some video assets failed verification!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
