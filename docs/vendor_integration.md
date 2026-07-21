# 与 `pcie_dma_test_100h` 的对接说明

## 1. 已确认的例程关键信息

我已经从你提供的 `pcie_dma_test_100h` 例程里确认了几个关键事实：

1. `ips2l_pcie_dma_ram.v` 当前配置是 `12 bit 地址宽度 + 128 bit 数据宽度`。
2. 单个 BAR RAM 容量是 `2^(12 + 4) = 65536 byte`，也就是 `64 KB`。
3. `ips2l_pcie_dma_wr_ctrl.v` 里写地址最终取的是 `wr_addr[ADDR_WIDTH+3:4]`，说明软件侧 BAR 偏移必须按 `16 byte` 对齐考虑。
4. 你板子当前枚举结果显示 Endpoint 只暴露了一个有效 BAR，也就是 `resource0`。

## 2. 这意味着什么

### 第一版分辨率不要太大

因为单个 BAR 窗口只有 `64 KB`，当前建议：

1. 输入到 FPGA 的灰度图固定缩放到 `320x180`
2. 总字节数 `57600`
3. 再加 `0x100` 状态头，仍然能放进 `64 KB`

### 原始彩色帧继续留在 ARM

更合理的做法是：

1. ARM 保留原始彩色帧。
2. FPGA 处理降采样灰度帧，返回二值掩码或候选区域提示。
3. ARM 用这些候选区域去驱动 YOLO。

## 3. 推荐接法

第一版直接做成单 BAR：

1. 主机把控制寄存器写到 `BAR0 + 0x000 ~ 0x070`
2. 主机把灰度图写进 `BAR0 + 0x100`
3. FPGA 内部把状态头映射到 `BAR0 + 0x000 ~ 0x0FF`
4. FPGA 内部把输出掩码映射到 `BAR0 + 0x100`

## 4. 你需要重点改的原厂文件

1. `.../pcie_dma_test_100h/hdl/pcie_dma_ctrl/ips2l_pcie_dma_rx_top.v`
2. `.../pcie_dma_test_100h/hdl/pcie_dma_test.v`
3. `.../pcie_dma_test_100h/hdl/pcie_dma_ctrl/ips2l_pcie_dma_controller.v`

## 5. 第一版最稳的改法

1. 保留原来的 PCIe 收发和 BAR 基础逻辑。
2. 新增自定义模块，把单个 BAR0 同时作为“寄存器头部 + 输入窗口 + 输出窗口”。
3. 先不要碰 DMA controller。
4. 等 BAR 路线联通后，再让 DMA controller 接管连续帧搬运。

## 6. 最小联调闭环

1. 先不用摄像头，ARM 写一张本地灰度图到 BAR0。
2. FPGA 做简单二值化。
3. ARM 从 BAR2 读回掩码并保存成图片。
4. 这一步跑通后再接 USB 摄像头和 YOLO。
