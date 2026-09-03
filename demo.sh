#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  ANESight 一键演示启动器
#  用法:  ./demo.sh            （交互菜单）
#         ./demo.sh --section bench   （直接进某板块，参数透传）
#  首次运行自动完成: 虚拟环境 → 依赖安装 → 演示素材 → 启动演示台
# ─────────────────────────────────────────────────────────────
set -e
cd "$(dirname "$0")"

# 1. 虚拟环境（缺则建）
if [ ! -x .venv/bin/python ]; then
    echo "◈ 首次运行：创建虚拟环境..."
    if command -v uv >/dev/null 2>&1; then
        uv venv .venv --python python3.12 2>/dev/null || uv venv .venv
    else
        python3 -m venv .venv
    fi
fi
PY=.venv/bin/python

# 2. 依赖（只装缺的，装过则秒过）
if ! "$PY" -c "import rich, cv2, torch, ultralytics, coremltools" 2>/dev/null; then
    echo "◈ 安装依赖（首次约 1-5 分钟，之后跳过）..."
    if command -v uv >/dev/null 2>&1; then
        uv pip install -r requirements.txt --python "$PY"
    else
        "$PY" -m pip install -q --upgrade pip
        "$PY" -m pip install -q -r requirements.txt
    fi
fi

# 3. 演示素材（本地视频优先，否则拉公开示例）
if [ ! -f test.mp4 ] && ! ls samples/*.mp4 >/dev/null 2>&1; then
    echo "◈ 下载演示素材（公开示例视频）..."
    "$PY" download_samples.py || echo "⚠ 素材下载失败，可手动放置任意 mp4 后重试"
fi

# 4. 启动演示台（首次分析视频时会自动导出 ANE 模型，约数秒）
exec "$PY" demo.py "$@"
