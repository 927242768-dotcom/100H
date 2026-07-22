# RKNN / NPU 接入说明

这份说明对应当前仓库里新接入的第一版 `RKNN/NPU` 路线：

1. `YOLOv8 车牌检测` 走 `RKNNLite + NPU`
2. `车牌字符识别` 先继续复用现有 `HyperLPR`

这样做的目的，是先尽快把 `NPU 检测 + 现有 OCR` 这条链路跑通，验证板端性能收益，再决定是否把 OCR 模型也继续迁移到 RKNN。

## 当前已完成的代码改动

当前已经新增：

- [detector.py](/D:/100H/competition_solution/arm/python/detector.py)
  - 新增 `RknnLiteDetector`
  - 支持加载 `.rknn` 模型
  - 支持 YOLOv8 风格输出解码
  - 支持 NMS
  - 支持可选复用 `HyperLPR` 做 OCR
- [pipeline.py](/D:/100H/competition_solution/arm/python/pipeline.py)
  - 新增 `--detector rknn`
  - 新增 RKNN 相关命令行参数
  - 保留现有 `mock` / `hyperlpr` 路线不变
- [export_plate_onnx.py](/D:/100H/yolov8-plate/export_plate_onnx.py)
  - 用于把 `yolov8-plate/weights/yolov8s.pt` 导出成 ONNX

## 第一步：导出 ONNX

在 `yolov8-plate` 目录下执行：

```bash
cd /home/linaro/yolov8-plate
python3 export_plate_onnx.py --weights ./weights/yolov8s.pt --imgsz 640
```

默认会在 `weights` 目录下生成对应的 `.onnx` 文件。

如果你是在 Windows 侧先导出，再传到板子上，也可以在 Windows 里执行同一个脚本。

## 第二步：用 RKNN Toolkit 转成 `.rknn`

这一步需要在你已经装好 `RKNN Toolkit` 的环境里完成。

目标产物建议命名为：

```text
yolov8s.rknn
```

推荐放到板子上的任一路径，例如：

```text
/userdata/yolov8-plate/weights/yolov8s.rknn
```

当前 `RknnLiteDetector` 会优先尝试这些默认路径：

- `/userdata/yolov8-plate/yolov8s.rknn`
- `/userdata/yolov8-plate/weights/yolov8s.rknn`
- `/home/linaro/yolov8-plate/yolov8s.rknn`
- `/home/linaro/yolov8-plate/weights/yolov8s.rknn`
- `/userdata/models/yolov8_plate.rknn`

如果都不是你的实际路径，就在命令里显式传：

```bash
--rknn-model /你的/实际/yolov8s.rknn
```

## 第三步：板端实时运行

### 方案 A：NPU 检测 + HyperLPR OCR

这是当前最推荐的第一版。

```bash
cd /home/linaro/competition_arm/python
export DISPLAY=:0
export XAUTHORITY=/home/linaro/.Xauthority
export LD_LIBRARY_PATH=/userdata/HyperLPR/HyperLPR/Prj-Linux/hyperlpr3/lib:/userdata/HyperLPR/HyperLPR/3rdparty_hyper_inspire_op/MNN-2.2.0/linux/lib:$LD_LIBRARY_PATH

sudo -E python3 pipeline.py \
  --resource-root /sys/bus/pci/devices/0002:21:00.0 \
  --camera 0 \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-backend v4l2 \
  --fpga-width 112 \
  --fpga-height 64 \
  --morph-cfg 0xD0 \
  --threshold-mode percentile \
  --threshold-percentile 78 \
  --mask-cleanup open_close \
  --mask-kernel 3 \
  --mask-min-area 48 \
  --mask-max-area-ratio 0.35 \
  --display-mode outline \
  --detector rknn \
  --detector-source full \
  --detector-interval 2 \
  --detector-search-input-width 960 \
  --detector-track-input-width 640 \
  --rknn-model /userdata/yolov8-plate/weights/yolov8s.rknn \
  --rknn-input-size 640 \
  --rknn-conf-threshold 0.20 \
  --rknn-nms-threshold 0.45 \
  --hyperlpr-max-num 8 \
  --hyperlpr-threads 1 \
  --fullscreen
```

### 方案 B：只看 NPU 车牌框

如果你先只想验证 NPU 检测有没有框出来，不想把 OCR 也带上：

```bash
cd /home/linaro/competition_arm/python
export DISPLAY=:0
export XAUTHORITY=/home/linaro/.Xauthority

sudo -E python3 pipeline.py \
  --resource-root /sys/bus/pci/devices/0002:21:00.0 \
  --camera 0 \
  --camera-width 1280 \
  --camera-height 720 \
  --camera-backend v4l2 \
  --fpga-width 112 \
  --fpga-height 64 \
  --morph-cfg 0xD0 \
  --threshold-mode percentile \
  --threshold-percentile 78 \
  --mask-cleanup open_close \
  --mask-kernel 3 \
  --mask-min-area 48 \
  --mask-max-area-ratio 0.35 \
  --display-mode outline \
  --detector rknn \
  --detector-source full \
  --rknn-model /userdata/yolov8-plate/weights/yolov8s.rknn \
  --rknn-input-size 640 \
  --rknn-conf-threshold 0.20 \
  --rknn-nms-threshold 0.45 \
  --rknn-disable-hyperlpr-ocr \
  --fullscreen
```

## 新增命令行参数

`pipeline.py` 当前已经支持这些 RKNN 参数：

- `--rknn-model`
  - RKNN 模型路径
- `--rknn-input-size`
  - RKNN YOLO 输入尺寸，默认 `640`
- `--rknn-conf-threshold`
  - 检测置信度阈值
- `--rknn-nms-threshold`
  - NMS 阈值
- `--rknn-core-mask`
  - NPU core mask，支持 `auto`、`0`、`1`、`2`、`0_1`、`0_1_2`
- `--rknn-labels`
  - 类别名称，默认是 `单层车牌,双层车牌`
- `--rknn-disable-hyperlpr-ocr`
  - 关闭 HyperLPR OCR，只保留 RKNN 车牌框检测

## 当前这版的定位

这不是最终版，而是一个“先把 NPU 利用起来”的过渡版本：

- 优点
  - 车牌检测已经开始从 CPU 挪到 NPU
  - 可以快速验证 YOLO 车牌检测模型在 RK3568 上的效果
  - 不需要立刻把 OCR 模型也迁移成 RKNN
- 局限
  - OCR 仍然在 CPU 上
  - 最终极限性能还不是最优
  - YOLO 的输出格式如果和当前 RKNN 导出方式不一致，可能还需要微调解码逻辑

## 下一步建议

如果这条线跑通，下一步优先做：

1. 固定 RKNN 导出参数
2. 在板子上抓一份真实输出张量形状
3. 根据真实输出微调 `RknnLiteDetector` 解码
4. 评估是否还需要把字符识别模型也转成 RKNN

## 可选行人检测

行人检测使用第二个独立 RKNN 模型和异步线程。默认不启用，因此原有车牌检测、HyperLPR OCR、FPGA 预处理、摄像头/SD 输入与 HDMI 显示不增加开销。启用后只保留模型中的 `person` 类，并用绿色框显示“行人 + 置信度”。

当前解码器要求 Ultralytics 原始单输出模型，ONNX 输出应为 `(1, 84, 8400)`。不要直接使用 Rockchip Model Zoo 的多分支优化 ONNX。

在 Windows 导出 ONNX：

```powershell
cd D:\100H\yolov8-plate
$env:YOLO_CONFIG_DIR='D:\100H\yolov8-plate\.ultralytics'
.\.venv\Scripts\python.exe .\export_plate_onnx.py --weights .\weights\yolov8n.pt --imgsz 640 --output-dir .\weights
```

在已安装 `rknn-toolkit2==2.3.2` 的 Linux x86_64 环境转换：

```bash
python3 tools/convert_yolov8_person_to_rknn.py \
  --onnx /path/to/yolov8n.onnx \
  --output /path/to/yolov8n.rknn \
  --target rk3568
```

把模型传到板端，例如 `/userdata/yolov8-person/yolov8n.rknn`，然后在原运行命令末尾增加：

```bash
  --person-model /userdata/yolov8-person/yolov8n.rknn \
  --person-model-classes 80 \
  --person-input-size 640 \
  --person-conf-threshold 0.25 \
  --person-nms-threshold 0.45 \
  --person-interval 3 \
  --person-hold-seconds 0.60
```

如果使用的是只含行人的单类模型，把 `--person-model-classes` 改为 `1`。`--person-interval` 越小，跟随越及时，但会占用更多 NPU 时间；RK3568 同时运行车牌和行人两个模型时建议先从 `3` 开始实测。
