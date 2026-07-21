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
