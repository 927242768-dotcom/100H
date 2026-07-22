# 实时帧率与延迟优化

当前实时链路采用三条相互解耦的执行路径：

1. 摄像头线程持续读取 V4L2，只保留最新帧，避免处理速度低于摄像头时积压旧画面。
2. FPGA 线程只处理最新的 `112x64` 灰度帧，PCIe 写入、启动、等待和掩码读回不再阻塞 HDMI。
3. RKNN 检测线程继续采用最新任务覆盖策略；连续命中同一车牌时复用最近一次成功 OCR，减少 HyperLPR 重复推理。

RKNN 路线直接从摄像头原图一次 letterbox 到模型的 `640x640`，不再经过 `1280 -> 960 -> 640` 两次缩放。两个 NPU 结果之间使用低分辨率稀疏光流更新车牌框位置，降低移动时的拖尾。

异步检测返回时，结果坐标实际属于约数百毫秒前的提交帧。管线现在保存该源帧，并用带前后向一致性检查的光流把新框直接补偿到当前 HDMI 帧；光流不可靠时自动退回上一版跟随逻辑。OCR 成功时还会使用 HyperLPR 的车牌内框校准 YOLO 外框，并按相对坐标复用校准结果。两项优化都不修改检测置信度、模型输入尺寸或检测频率。

每个 RKNN 检测器会复用 letterbox、RGB 和浮点输入缓冲区，避免每轮推理重新分配约数 MB 的临时数组；输入布局、归一化数值和模型输出语义保持不变。`outline`/`camera` 模式也不再生成未被显示的全分辨率掩码。

## 日志指标

- `real_fps`：程序实际 HDMI 主循环帧率；屏幕上的 `FPS` 仍按比赛展示要求显示为两倍。
- `fpga_ms`：一次 FPGA 往返处理耗时。FPGA 已异步，因此该值不会直接阻塞 HDMI。
- `det_ms`：一次 RKNN 检测及必要 OCR 的总耗时，是判断模型瓶颈的主要指标。
- `det=1`：检测线程正在工作，不代表 HDMI 主循环被阻塞。
- `fpga_async=1`：FPGA 后台线程正在处理最新任务。

## 新增参数

- `--camera-buffered`：关闭最新帧采集，恢复传统同步 `cap.read()`，仅用于兼容性排查。
- `--disable-box-tracking`：关闭光流跟随，直接显示 NPU 原始框，用于 A/B 对比。
- `--disable-detection-lag-compensation`：关闭检测源帧到当前帧的延迟补偿，仅用于定位回退测试。
- `--rknn-disable-ocr-box-refinement`：继续识别文字，但关闭 HyperLPR 对车牌框的二次校准。
- `--rknn-ocr-cache-seconds 2.0`：成功 OCR 的默认复用时间。
- `--rknn-ocr-cache-iou 0.50`：复用 OCR 结果所需的最低框重叠度。

当前常用启动命令无需增加参数即可启用最新帧、异步 FPGA、OCR 缓存和框跟随。不要添加 `--camera-buffered` 或 `--disable-box-tracking`。

## 双 RKNN 模型优化

车牌和行人模型同时启用时，默认通过共享锁串行调用 RK3568 的单核 NPU。锁只覆盖 `rknn.inference()`，车牌 OCR 和后处理在 CPU 执行期间，另一个模型仍可进入 NPU，从而避免两个 RKNN 上下文同时争抢 NPU。若需要与旧行为 A/B 对比，可临时增加 `--rknn-allow-concurrent-inference`。

行人模型在 NMS 前只保留 COCO 类别 `0`，其余 79 类不会进入坐标恢复和 NMS。该优化不改变行人分数和框位置，同时减少后处理负担，并避免其他类别与行人做跨类别 NMS。

日志中的 `person_ms` 是行人预处理、等待 NPU、推理及后处理总耗时。串行调度后应同时观察 `det_ms`、`person_ms` 和 `real_fps`，不能只比较单个模型耗时。

## 摄像头高速模式

部分 USB 摄像头在 `1280x720` 默认 YUYV 模式下只能输出约 5 FPS。可先执行 `v4l2-ctl --device /dev/video0 --list-formats-ext`，确认设备支持 `MJPG 1280x720` 后，在启动命令中增加：

```bash
--camera-fourcc MJPG \
--camera-fps 30
```

启动日志会打印实际协商结果，例如 `mode=1280x720@30.0 fourcc=MJPG`。如果仍显示约 `5.0` 或格式不是 `MJPG`，说明摄像头没有接受该模式；删除这两个参数即可恢复原采集行为。

## 模型侧上限

当前 `yolov8s.rknn` 由非量化 ONNX 默认转换得到，文件约 7.45 MB。若日志中的 `det_ms` 仍长期很高，下一步应使用板端真实车牌画面制作校准集并生成 INT8 RKNN：

```powershell
python .\convert_plate_to_rknn.py `
  --onnx .\weights\yolov8s.onnx `
  --output .\weights\yolov8s_int8.rknn `
  --target rk3568 `
  --do-quantization `
  --dataset .\calibration.txt
```

`calibration.txt` 每行写一张校准图片路径，建议使用 100 至 300 张来自实际摄像头、覆盖远近距离和各种车牌类型的图片。INT8 可以明显降低 NPU 推理时间，但必须用真实场景校准并对识别率做 A/B 测试，不能直接覆盖当前模型。
