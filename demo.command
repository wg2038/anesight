#!/bin/bash
# macOS 双击即开：Finder 里双击本文件 → 自动打开终端
# 直接进入 test.mp4 实况分析；Ctrl+C 停止后统计总结保留在窗口中
"$(dirname "$0")/demo.sh" "$@"
echo ""
echo "◈ 演示结束。按回车关闭窗口..."
read
