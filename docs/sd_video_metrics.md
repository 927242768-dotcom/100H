# SD卡视频输入与自动化测试

## 1. 部署文件

把下面三个文件同步到板端 `/home/linaro/competition_arm/python/`：

- `arm/python/pipeline.py`
- `arm/python/evaluation_metrics.py`
- `arm/python/detector.py`（如果板端版本不是当前仓库版本）

SD卡挂载位置可先用下面命令确认：

```bash
lsblk -f
findmnt
```

以下示例假设视频位于 `/media/linaro/SDCARD/test.mp4`。

## 2. SD视频实时显示

`realtime` 模式不会故意快于视频原始帧率，视频结束后程序正常退出。添加 `--video-loop` 可循环播放。

```bash
cd /home/linaro/competition_arm/python
export DISPLAY=:0
export XAUTHORITY=/home/linaro/.Xauthority
export LD_LIBRARY_PATH=/userdata/HyperLPR/HyperLPR/Prj-Linux/hyperlpr3/lib:/userdata/HyperLPR/HyperLPR/3rdparty_hyper_inspire_op/MNN-2.2.0/linux/lib:$LD_LIBRARY_PATH

sudo -E python3 pipeline.py \
  --resource-root /sys/bus/pci/devices/0002:21:00.0 \
  --input-video /media/linaro/SDCARD/test.mp4 \
  --video-playback realtime \
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
  --detector-interval 1 \
  --rknn-model /userdata/yolov8-plate/yolov8s.rknn \
  --rknn-input-size 640 \
  --rknn-conf-threshold 0.15 \
  --rknn-nms-threshold 0.45 \
  --hyperlpr-threads 1 \
  --metrics-dir /userdata/competition_metrics/sd_video_realtime \
  --fullscreen
```

## 3. 最大吞吐率测试

`fast` 模式不等待视频原始帧率，`headless` 关闭HDMI绘制，适合测试整条处理链最大吞吐率。设置预热帧后，正式报告优先使用 `summary.json` 中的 `runtime.measured_throughput_fps`；未设置预热帧时可使用 `runtime.throughput_fps`。不要使用界面倍增后的FPS。

```bash
sudo -E python3 pipeline.py \
  --resource-root /sys/bus/pci/devices/0002:21:00.0 \
  --input-video /media/linaro/SDCARD/test.mp4 \
  --video-playback fast \
  --fpga-width 112 \
  --fpga-height 64 \
  --morph-cfg 0xD0 \
  --threshold-mode percentile \
  --threshold-percentile 78 \
  --mask-cleanup open_close \
  --mask-kernel 3 \
  --mask-min-area 48 \
  --mask-max-area-ratio 0.35 \
  --detector rknn \
  --detector-source full \
  --detector-interval 1 \
  --rknn-model /userdata/yolov8-plate/yolov8s.rknn \
  --rknn-input-size 640 \
  --rknn-conf-threshold 0.15 \
  --rknn-nms-threshold 0.45 \
  --hyperlpr-threads 1 \
  --metrics-dir /userdata/competition_metrics/sd_video_fast \
  --metrics-warmup-frames 10 \
  --headless
```

## 4. 输出文件

指定 `--metrics-dir` 后自动生成：

- `frames.csv`：逐帧源时间、真实FPS、候选框数、车牌数、行人数和各阶段耗时。
- `detections.csv`：检测框、置信度、OCR文字、车牌类型及标注匹配结果。
- `summary.json`：吞吐率、FPGA/车牌/行人延迟P50/P95、OCR输出率及可选准确率。

CSV使用UTF-8 BOM编码，可直接使用Excel打开。

## 5. 带标注准确率测试

标注JSON示例：

```json
{
  "frames": [
    {
      "frame": 1,
      "plates": [
        {
          "box": [120, 240, 180, 56],
          "text": "苏ED51712",
          "type": "绿牌新能源"
        }
      ]
    },
    {
      "frame": 2,
      "plates": []
    }
  ]
}
```

`box`固定采用 `[x, y, width, height]`。空画面也应写入对应帧并将 `plates` 设为空数组，这样才能统计误检。运行时增加：

```bash
  --metrics-ground-truth /media/linaro/SDCARD/test_ground_truth.json \
  --metrics-iou-threshold 0.50
```

提供标注后，`summary.json`会增加：

- `precision`、`recall`、`f1`
- `mean_matched_iou`
- `ocr_exact_accuracy`
- `type_exact_accuracy`

异步检测结果会按照提交检测时的原始视频帧号进行匹配，不会错误地对齐到检测完成时已经显示的后续帧。
