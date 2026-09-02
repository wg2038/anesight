#!/bin/bash
# 一键运行脚本 — YOLO 智能监控分析（Apple Silicon 版）
# 用法:
#   ./run.sh                          # 预览（balanced 模式）
#   ./run.sh --mode eco               # 省电值守：实时节奏 + 能效核
#   ./run.sh --mode turbo             # 全性能：yolov8m + 大画布 + 性能核
#   ./run.sh --no-gui video.mp4 ...   # 任意参数透传给 run_detector.py
cd "$(dirname "$0")"

if [ $# -eq 0 ]; then
    SRC="test.mp4"                       # 本地素材优先
    [ -f "$SRC" ] || SRC="samples/video1.mp4"   # 回退: download_samples.py 的公开素材
    [ -f "$SRC" ] || { echo "请先运行 python download_samples.py 或指定视频文件"; exit 1; }
    exec ./.venv/bin/python run_detector.py --videos "$SRC" --no-save --help-keys
else
    exec ./.venv/bin/python run_detector.py "$@"
fi
