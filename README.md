# YOLO 智能监控分析平台 — Apple Silicon 深度优化版

基于 Ultralytics YOLOv8 + ByteTrack 的**多场景**实时视频分析系统，专为 Apple Silicon（M1/M2/M3/M4）深度调优：
**推理跑在神经网络引擎（ANE）上，视频编码走 VideoToolbox 硬件，GPU 留给渲染，CPU 只做调度**。

三大场景（对标 [Ultralytics Solutions](https://docs.ultralytics.com/solutions) 全部能力并超越）：

| 场景 | 能力 |
|---|---|
| `road` 道路交通 | 方向车流计数、测速、热力图、两轮车（电动车）统计 |
| `parking` 停车场 | **车位级占用检测**（单格/网格批量）、进出场计数、停车时长、周转率 |
| `factory` 厂区安防 | **禁区入侵**、**滞留告警**、**人车接近预警**（叉车-行人防碰撞）、布防时段、Webhook 推送 |

在 Apple M3 (16GB) 上，对 1280×674 视频的端到端处理速度 **137 FPS**（原基线 28.8 FPS，**提升 4.8 倍**）。

## 工作原理

整个系统围绕 Apple Silicon 的**异构计算分工**设计：ANE 做推理、VideoToolbox 做编解码、GPU 做渲染、CPU 只做调度。

```
视频源 (文件/摄像头/RTSP)
   │  线程化捕获：文件源带缓冲队列+可跳转；直播源"最新帧优先"防积压
   ▼
预处理 letterbox (CPU, ~1.4ms)          ┌─────────────────────────────┐
   原图 → 画布缩放+填充 + BGR→RGB       │ 首次运行自动完成：            │
   ▼                                    │  YOLO .pt                    │
NCHW float 张量 (0-255, 不缩放)          │    │ Ultralytics 导出        │
   ▼                                    │    ▼ CoreML fp16 mlprogram   │
Apple Neural Engine (~4-20ms)           │    ▼ spec 补丁: PIL→Tensor   │
   fp16 权重 + 内嵌 NMS                  │    ▼ 缓存 *_ane_t.mlpackage  │
   ▼                                    └─────────────────────────────┘
(1,N,80) 置信度矩阵 + (N,4) xywh
   │  反 letterbox（比例/填充逆变换回原图坐标）+ 置信度过滤
   ▼
每流独立 BYTETracker（两段关联：高分匹配 + 低分补配）
   ▼
分析层：方向 tripwire / 车位状态机 / 禁区 / 接近检测 / 测速 / 热力图
   ▼
渲染 (GPU) + VideoToolbox H.264 输出 + 告警总线 (CSV/Webhook)
```

几个关键设计决策：

1. **ANE 张量直连（提速 2.4 倍的关键）**：Ultralytics 导出的 CoreML 模型以 PIL Image 为输入，
   每帧要经历多次内存拷贝和 PIL 编解码（~10ms 纯开销）。本系统在导出后对模型 spec 打补丁，
   把输入改写为原始 float 张量（/255 缩放已烘焙在计算图内，letterbox 逆变换在引擎侧完成），
   令 ANE 推理降至 ~4ms/帧。这是本仓库与常规 `model.export()` 用法的核心差异。
2. **矩形画布**：ANE 对固定形状优化，960×960 方形画布对横屏视频有近半是 padding；
   `512x960` 匹配 1.9:1 素材后快 2.6 倍。竖屏素材传 `960x512`。
3. **手动 ByteTrack 而非 `model.track`**：`model.track(persist=True)` 全局只有一份追踪状态，
   多路视频会串台，且其 conf 参数会截断低分候选池。本系统直接驱动 `BYTETracker.update()`，
   每路一份状态、喂入完整置信度池（ByteTrack 论文的标准两段关联），追踪 ID 更稳。
4. **位图级准确率对齐**：ANE fp16 与 MPS fp32 的检测结果 mean IoU 0.983、帧均数量差 0.2，
   精度无损。
5. **几何层全部纯逻辑**：tripwire 滞回带（6px，红绿灯停车抖动零误报）、车位状态机、
   禁区滞留、告警去重均不依赖 I/O，可离线单元测试（tests/ 共 114 项断言）。

## 终端演示台

```bash
python demo.py        # 交互菜单
```

六个板块，全部真实执行、无模拟数据：

| 板块 | 实际做什么 |
|---|---|
| ① 系统体检 | 实时探测芯片/P-E核/ANE/VideoToolbox/ANE 模型缓存/视频源 |
| ② 引擎基准 | 真跑 ANE / MPS / CPU 检测+追踪对比，实时进度条 + FPS 条形图 |
| ③ 场景实况 | 真分析视频流，终端实时遥测：FPS、类别构成、线计数、车位、告警流 |
| ④ 工作原理 | ANE 直连数据流图 + 五个关键设计决策 |
| ⑤ 质量门禁 | 真跑 5 套测试，通过/失败/耗时一目了然 |
| ⑥ 能耗模式 | eco/balanced/turbo 配置与本机实测数据 |

也支持脚本化：`python demo.py --section live --video xx.mp4 --scenario factory --max-frames 300`

## 三大场景用法

```bash
# ── 道路交通（默认）：方向车流 + 测速 + 热力图
python run_detector.py test.mp4 --heatmap --speed --mpp 0.02

# ── 停车场：网格车位一键生成 + 车位级占用 + 门禁进出线
python run_detector.py parking.mp4 --scenario parking \
    --slot-grid "60,440,230,120,4,3" \
    --slot "1080,140,1240,290:P-EXEC" \
    --line "60,420,780,420:gate:car,bus,truck" \
    --snapshots
# 车位占用/释放实时告警（含停车时长），HUD 显示 PARKING 5/13，汇总 JSON 输出周转数据

# ── 厂区安防：禁区入侵 + 滞留 + 人车接近 + 时段布防 + Webhook
python run_detector.py factory.mp4 --scenario factory \
    --restricted "1000,100,1280,674:high-voltage" \
    --restricted "0,0,420,140:warehouse-door:person,car" \
    --proximity-px 100 --loiter-sec 15 \
    --armed-hours "8-18" --webhook http://ops.local/hook --snapshots
```

场景自动识别：给了 `--slot*` → parking；给了 `--restricted/--proximity-px` → factory；否则 road。
`--scenario` 显式指定永远生效。

**告警体系**：入侵/滞留为 `critical`，接近/车位为 `warning`；同类告警自动去重（`--alert-cooldown` 秒）；
事件写入 `alerts_*.csv`，可选 POST JSON 到企业微信/钉钉/Slack 网关（`--webhook`）；
`--snapshots` 自动保存告警瞬间画面。

**自定义模型即插即用**：任何 Ultralytics 训练的 .pt（如安全帽/PPE 检测模型）直接 `--model ppe.pt`
即可接入全部管线——类别名从模型元数据自动读取，ANE 自动导出，框色自动分配，无需改代码。

## 性能模式：节能 / 均衡 / 全性能

`--mode` 一键切换，实测于 Apple M3（test.mp4 全片 10811 帧，无 GUI）：

| 模式 | 模型 | 画布 | 行为 | 处理速度 | CPU 占用 | 适用 |
|---|---|---|---|---|---|---|
| `eco` | yolov8n | 512x960 | **按源视频实时节奏处理** + UTILITY QoS（调度偏好能效核） | 12.0 FPS（=源帧率） | **18%** | 长期值守、笔记本电池、监控大屏 |
| `balanced`（默认） | yolov8s | 512x960 | 全速处理 | 137 FPS | 93% | 批量分析、最快出结果 |
| `turbo` | yolov8m | 704x1280 | 全速 + 最高精度 + 12Mbps 输出 + USER_INTERACTIVE QoS（锁性能核） | 43 FPS | 68% | 精度优先（小目标/电动车多）、画质存档 |

节能模式的原理：ANE 本身已是 Apple Silicon 能效最高的计算单元，而**真正的耗电大户是无意义的全速空转**——
12fps 的摄像头不需要 137fps 的推理。eco 模式按源帧率节奏处理（16GB M3 实测 CPU 占用 93% → 18%），
并让调度器把编排工作偏向能效核。

```bash
python run_detector.py test.mp4 --mode eco           # 省电值守（实时节奏）
python run_detector.py test.mp4 --mode turbo         # 全性能（yolov8m + 704x1280）
python run_detector.py test.mp4 --mode turbo --model yolov8s.pt --imgsz 896x1536  # 任意组合
python run_multistream.py cam1 cam2 --mode eco       # 多路值守
```

> 模式只是默认值：显式传入的 `--model / --imgsz / --realtime / --skip-frames` 永远优先。
> turbo 极限玩法：`--model yolov8m.pt --imgsz 896x1536`（33 FPS，1536 宽画布小目标最清晰）。

## 性能实测（Apple M3，512×960 画布，含追踪）

| 后端 | 延迟 | 吞吐 | 相对加速 |
|---|---|---|---|
| **CoreML/ANE fp16（张量直连）** | **6.7 ms/帧** | **148.5 FPS** | **8.2×** |
| MPS fp16 (PyTorch Metal) | 55.2 ms/帧 | 18.1 FPS | 1.0× |
| CPU | 76.5 ms/帧 | 13.1 FPS | 0.7× |

> ANE 直连路径的关键：导出的 CoreML 模型默认以 PIL Image 为输入，每帧要经过
> 多次内存拷贝和 PIL 编解码（~10ms 开销）。本项目在导出后对模型 spec 打补丁，
> 把输入改写为原始 float 张量，letterbox 前处理与坐标反变换全部在引擎内完成，
> ANE 推理降至 ~4ms。检测质量与 MPS fp32 对比：**mean IoU 0.983**，计数一致。

模型精度/速度可选（均已自动导出缓存 ANE 版本）：

| 模型 | ANE 推理 | 追踪模式 | 适用 |
|---|---|---|---|
| yolov8n | ~3.4 ms | ~80 FPS | 多路摄像头、边缘设备 |
| **yolov8s（默认）** | ~5 ms | ~148 FPS | 均衡 |
| yolov8m | ~9 ms | ~60 FPS | 最高精度 |

## 电动车 / 两轮车识别说明

COCO 数据集（YOLOv8 的训练集）**没有"电动车"类**——电动车在画面中会以这两种类别出现：

| COCO 类别 | 典型对应 | 最低显示置信度 |
|---|---|---|
| `motorcycle` (3) | 电动车（踏板/跨骑）、摩托车 | **0.30** |
| `bicycle` (1) | 电动车（弯梁/简易款）、自行车 | **0.30** |

系统将二者统一为 **TW（two-wheeler，两轮车）** 类别单独统计（HUD 的 `TW:` 行与报告的
`two-wheelers (e-bike)`），不与机动车 `V:` 计数混淆。两轮车因目标小、距离远，检出置信度
天然低于汽车，因此使用专属低阈值（0.30）——远处小电动车在 0.3~0.45 区间也会被保留。

提升两轮车检出率的手段：

```bash
# 1. 提高推理画布（小目标检出 +51%，ANE 下仍 ~8ms/帧）
python run_detector.py test.mp4 --imgsz 704x1280

# 2. 进一步降低显示阈值（会引入更多远处弱检出）
python run_detector.py test.mp4 --conf 0.35

# 3. 给两轮车单独设检测线（斑马线/非机动车道）
python run_detector.py test.mp4 --line "0,300,600,300:e-bike-lane:motorcycle,bicycle"
```

> 速度估计提示：远景目标的 px 位移对应实际速度偏大（透视原理），单点 `--mpp` 标定下
> 远处两轮车的 km/h 读数会偏高，属预期；精确测速需分区标定或单应性变换。

## 核心特性

1. **后端自动择优**：CUDA → CoreML/ANE → MPS → CPU，无需配置；CoreML 模型首次运行自动导出并缓存（`*_ane_t.mlpackage`）。
2. **手动 ByteTrack 管理**：每路视频流独立追踪器状态，共享同一编译模型（`model.track` 无法做到），且向追踪器喂入低置信度候选池（ByteTrack 论文标准两段关联），ID 更稳定。
3. **方向感知 tripwire 计数**：带滞回带（默认 6px，红绿灯停车抖动零误报）、每方向每车只计一次、IN/OUT 双向统计、可按类别过滤。
4. **多边形区域统计**：任意多边形区域，实时占用计数 + 累计访问人数。
5. **运动热力图**：衰减累积 + 半分辨率 JET 着色，可视化车流/人流热区。
6. **速度估计**：轨迹窗口位移 → px/s；配合 `--mpp`（米/像素标定）输出 km/h。
7. **硬件编解码**：输出视频默认走 `ffmpeg h264_videotoolbox`（H.264/yuv420p 全兼容），无 ffmpeg 时回退 cv2。
8. **线程化捕获**：文件源带缓冲队列 + 可跳转（seek）；摄像头/RTSP 源采用"最新帧优先"策略，永不积压。
9. **会话日志**：每次穿越事件写入 CSV（含视频时间戳、车速），最终统计写入 JSON。
10. **多路监控台**（`run_multistream.py`）：N 路输入平铺上墙，独立计数，聚合吞吐 3 路 >100 FPS。
11. **交互控制**：`空格` 暂停、`.` 逐帧、`t` 轨迹、`h` 热力图、`z` 区域、`l` 检测线、`s` 截图、`q` 跳过、`Esc` 退出。
12. **类内稳定投票**：15 帧多数投票消除 car/truck/bus 标签闪烁；嵌套框抑制（≥70% 包含）去重。
13. **两轮车（电动车）专项**：bicycle+motorcycle 双类别捕获、0.30 专属阈值、TW 独立统计，详见上文说明。

## 快速开始

```bash
# 1. 环境（uv 或 venv 均可）
uv venv .venv --python python3.12
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. 测试素材（公开示例视频，含行人/自行车/汽车）
python download_samples.py        # 下载到 samples/video1.mp4, video2.mp4

# 3. 运行（首次运行自动下载 .pt 权重并导出缓存 ANE 模型）
python run_detector.py samples/video1.mp4 --heatmap     # 完整管线预览
python run_detector.py samples/video1.mp4 --no-gui      # 无窗口批处理
python benchmark.py                                     # 本机硬件基准
```

> 模型权重与 ANE 缓存（`*.pt` / `*.mlpackage`）不入库：`.pt` 首次运行时由
> Ultralytics 自动下载，ANE 版由本系统自动导出。测试录像同样不入库（隐私），
> 请用自备视频或 `download_samples.py` 的公开素材。

验证（5 套测试共 114 项断言）：

```bash
python tests/test_analytics.py && python tests/test_io.py && \
python tests/test_classes.py && python tests/test_perf.py && python tests/test_scenarios.py
```

## 常用示例

```bash
# 全功能：热力图 + 速度(km/h) + 自定义区域 + 事件截图
python run_detector.py test.mp4 --heatmap --speed --mpp 0.02 \
    --zone "880,280,1260,660:curb" --snapshots

# 自定义检测线（p1->p2 左侧为 IN；可限定类别）
python run_detector.py video.mp4 \
    --line "100,400,900,400:gate:car,bus,truck" --line "400,100,400,600:exit"

# 最高精度模型
python run_detector.py test.mp4 --model yolov8m.pt

# 三路监控（文件/摄像头/RTSP 混合）
python run_multistream.py test.mp4 0 rtsp://camera.local/stream --heatmap

# 摄像头实时
python run_detector.py 0 --speed --mpp 0.02
```

## 项目结构

```
├── run_detector.py        # 主入口：三场景管线（道路/停车场/厂区）
├── run_multistream.py     # 多路监控台（N 路平铺、独立追踪、聚合统计）
├── benchmark.py           # 基准工具（detect / track 两种模式）
├── traffic/
│   ├── engine.py          # 后端选择 + ANE 张量直连 + 手动 ByteTrack + 自定义模型
│   ├── capture.py         # 线程化捕获（文件队列 / 实时丢帧 / 线程安全 seek）
│   ├── writer.py          # VideoToolbox 硬件编码输出（cv2 回退）
│   ├── analytics.py       # 方向 tripwire / 多边形区域 / 热力图 / 速度 / 轨迹簿
│   ├── parking.py         # 停车场：车位状态机 + 网格生成 + 周转统计
│   ├── factory.py         # 厂区：禁区入侵/滞留 + 人车接近 + 布防时段
│   ├── alerts.py          # 告警总线：去重 / CSV / Webhook
│   ├── annotate.py        # HUD 仪表盘与所有叠加渲染
│   ├── perf.py            # 性能模式（eco/balanced/turbo）+ QoS 调度
│   └── logio.py           # CSV 事件/告警日志 + JSON 会话摘要 + 截图
├── tests/                 # 5 套测试：几何/IO/类别/性能模式/场景逻辑
├── custom_bytetrack.yaml  # ByteTrack 调参（长缓冲抗遮挡）
├── download_samples.py    # 测试素材下载
└── output/                # 标注视频 / 事件与告警 CSV / 会话 JSON / 截图
```

## 关键参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--scenario` | 自动 | `road` / `parking` / `factory`（按配置自动推断） |
| `--model` | 按模式 | 任意 YOLOv8/v11 .pt（含自定义模型），ANE 版自动导出缓存 |
| `--mode` | balanced | `eco` / `balanced` / `turbo` 性能模式 |
| `--imgsz` | 512x960 | 推理画布，横屏视频用 `HxW` 矩形最省 ANE 算力 |
| `--line` | 道路场景内置 | `x1,y1,x2,y2[:名称[:类别1,类别2]]`，可重复（原始像素坐标） |
| `--zone` | 无 | 通用多边形区域 `x1,y1,...[:名称]`（4 数=矩形） |
| `--slot` / `--slot-grid` | 无 | 停车车位：单格 / 网格批量 `x,y,w,h,列,行[:前缀]` |
| `--restricted` | 无 | 禁区 `x1,y1,...[:名称[:触发类别]]`，默认 person 触发 |
| `--proximity-px` | 0 | 人车接近告警距离（px，厂区推荐 100~150） |
| `--loiter-sec` / `--armed-hours` | 20 / 24h | 滞留阈值 / 布防时段 `8-18`（支持 `22-6` 跨夜） |
| `--webhook` / `--alert-cooldown` | 无 / 30s | 告警 JSON 推送地址 / 去重间隔 |
| `--mpp` | 无 | 米/像素标定，启用后速度显示 km/h |
| `--no-save` / `--no-gui` / `--no-log` | — | 关闭视频输出 / 窗口 / 日志 |

## 设计说明

- **为什么矩形画布快 2.6 倍**：ANE 对固定形状优化，横屏视频 960×960 方形画布有近一半是 padding；`512x960` 匹配 16:9/1.9:1 素材。竖屏素材建议 `960x512`（宽高互换）。
- **为什么手动追踪器**：Ultralytics 的 `model.track(persist=True)` 全局只有一份追踪状态，多流会串台；且其 `conf` 参数会截断低分池。本引擎直接驱动 `BYTETracker.update()`，每流一份状态、喂完整置信度池。
- **嵌套抑制**：卡车车斗/驾驶室常被重复检出，包含度 ≥70% 的内框直接剔除（向量化实现）。

## 验证

```bash
python tests/test_analytics.py   # 35 项：tripwire 方向/滞回/去重/类别过滤、区域、热力图、速度、抑制、投票
python tests/test_io.py          # 10 项：捕获读写、seek 竞态、两种编码器输出可解码
python tests/test_classes.py     # 11 项：类别体系完整性（含两轮车阈值）
python tests/test_perf.py        # 18 项：性能模式解析、显式参数优先、QoS
python tests/test_scenarios.py   # 40 项：车位状态机/网格、入侵/滞留/接近、布防时段、告警去重
python benchmark.py              # 输出本机各后端实时基准
```
