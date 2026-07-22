# RK3568_MES2L100H 比赛工程起步版

这个目录是我基于你当前资料先搭好的第一版比赛工程骨架，目标是先把整条链路拆清楚并落成可继续开发的工程：

1. `USB 摄像头 -> ARM 采集`
2. `ARM -> PCIe BAR -> FPGA`
3. `FPGA 预处理 -> BAR 输出缓冲`
4. `ARM 读回结果 -> YOLO -> HDMI 显示`

当前版本采用的是“先跑通、再提速”的策略：

1. 第一阶段优先使用 `单 BAR(resource0)` 共享窗口方案，快速完成联调。
2. 第二阶段再把大帧数据搬运切到现有 `pcie_dma_test_100h` 的 DMA 通道，提高实时性。

## 目录说明

1. `docs`
   方案、地址映射、和现有例程的对接说明。
2. `fpga/src`
   自定义 FPGA 逻辑骨架，包含寄存器、输入输出帧缓冲、预处理状态机。
3. `arm/python`
   ARM 侧用户态原型，负责摄像头采集、PCIe BAR 交互、候选区域提取、YOLO 接口和 HDMI 显示。

## 当前默认假设

1. FPGA 作为 PCIe Endpoint，RK3568 作为 Root Complex。
2. 你板子当前枚举到的 Endpoint 只暴露了一个有效 BAR，也就是 `resource0`，大小 `64 KB`。
3. 因此第一版 FPGA 预处理采用 `单 BAR 头部寄存器 + 单 BAR 输出窗口` 方案。
4. ARM 上保留原始彩色帧给 YOLO，FPGA 负责生成二值掩码/候选区域先验。

## 推荐开发顺序

1. 先看 `docs/vendor_integration.md`，确认你要怎么把自定义逻辑接到现有 PCIe 例程。
2. 在 FPGA 侧先完成 BAR0 输入缓冲、BAR1 控制写入、BAR2 输出读取。
3. 在板子 Linux 上用 `arm/python/pipeline.py` 验证摄像头、PCIe BAR、预处理和显示链路。
4. 最后把 YOLO 模型切到 `RKNNLite` 或你手头已有的 RK3568 YOLO 推理工程。

## SD 卡图片输入

先在板端确认 SD 卡挂载点：

```bash
lsblk -f
findmnt -t vfat,exfat,ext4
```

静态图片与摄像头共用同一条 FPGA、RKNN、OCR 和 HDMI 显示链路。把原命令中的
`--camera 0` 保留或删除均可，再增加图片参数即可：

```bash
sudo -E python3 pipeline.py \
  --resource-root /sys/bus/pci/devices/0002:21:00.0 \
  --sd-image /media/linaro/SDCARD/test.jpg \
  --fpga-width 112 \
  --fpga-height 64 \
  --morph-cfg 0x94 \
  --threshold-mode percentile \
  --threshold-percentile 78 \
  --mask-cleanup off \
  --detector rknn \
  --detector-source full \
  --detector-interval 1 \
  --rknn-model /userdata/yolov8-plate/yolov8s.rknn \
  --rknn-input-size 640 \
  --rknn-conf-threshold 0.15 \
  --rknn-nms-threshold 0.45 \
  --fullscreen
```

`--sd-image` 是 `--input-image` 的别名。程序会重复处理该图片，适合对比 FPGA
配置、检测阈值和 OCR 结果；不传该参数时仍从摄像头读取。

## SD卡视频与自动化测试

主程序现已支持 `--input-video/--sd-video`、视频原速或极速测试、循环播放，以及CSV/JSON指标输出。
完整部署命令、标注格式和指标说明见 `docs/sd_video_metrics.md`。

## 行人违法行为标记

可选的机动车道/禁入区判定会对进入区域的行人连续确认，并在HDMI、日志和自动测试结果中标记违法行为。
完整参数和区域标定方法见 `docs/pedestrian_violation.md`。

## 二维 FPGA 预处理实验版

`fpga/src/rk3568_traffic_preprocess_fpga_v2.v` 在原有 BAR 协议上增加了二维 3x3
高斯滤波、Sobel 边缘和二维开闭运算。推荐先使用以下配置做 A/B 测试：

1. `0xD0`：二维高斯 + 先开后闭，不启用 Sobel。
2. `0x94`：二维高斯 + Sobel + 闭运算，优先保留弱车牌边缘。
3. `0xD4`：二维高斯 + Sobel + 先开后闭，去噪更强但可能损失小车牌边缘。

启用 Sobel 时，ARM 会自动根据 Sobel 幅值图计算阈值，默认限制在 24 到 192；
可用 `--sobel-threshold-min`、`--sobel-threshold-max` 调整。硬件已经执行形态学时，
建议加 `--mask-cleanup off`，避免 ARM 重复计算。

FPGA 预处理主要改善掩码和 ROI 质量，并减少 ARM 掩码清理开销。当前
`--detector-source full` 仍会对整帧执行 RKNN，因此总帧率提升不会等于 FPGA
处理速度提升；确认 ROI 稳定后可测试 `--detector-source hybrid`，同时保留周期性
全帧搜索以避免漏检。

本次改动前的 Git 回退标签为 `pre-sd-fpga-pipeline-20260722`。

本机 PDS 工程位于
`D:/100H/fpga_work/fpga_demo_100h/pcie_dma_test_100h/pcie_dma_test.pds`。
工程已加入 v2 源文件，并在
`hdl/pcie_dma_ctrl/ips2l_pcie_dma_rx_top.v` 中将 BAR0 预处理实例切换为
`rk3568_traffic_preprocess_fpga_v2`。旧模块仍完整保留；若板端 A/B 测试不理想，
将该实例名改回 `rk3568_traffic_preprocess_fpga` 即可恢复旧 FPGA 行为，软件则可从
上述 Git 标签检出原版本。

PDS 2022.2-SP6.4 最终报告为 `All Constraints Met`：250 MHz `pclk` 的 WNS 为
`+0.558 ns`，125 MHz `pclk_div2` 的 WNS 为 `+0.492 ns`。资源占用约为 LUT
15.2%、寄存器 3.7%、DRM 44.8%。正式烧录文件为：

`D:/100H/fpga_work/fpga_demo_100h/pcie_dma_test_100h/generate_bitstream/pcie_dma_test.sbit`

旧版烧录文件保留在：

`D:/100H/fpga_work/fpga_demo_100h/pcie_dma_test_100h/generate_bitstream/bak/pcie_dma_test.sbit`
