# FPGA 正式烧录文件

`pcie_dma_test.sbit` 是 PDS 2022.2-SP6.4 生成并用于当前 RK3568 + PG2L100H 板卡的正式烧录文件。

对应设计：

- 器件：PG2L100H-6-FBG484
- PCIe Endpoint
- BAR0 单窗口图像输入、配置与掩码输出
- FPGA 预处理：3×3 高斯、Sobel、阈值化、二维形态学
- 推荐输入：112×64 灰度图

烧录前请确认开发板型号和 FPGA 器件一致。文件 SHA-256 见根目录 `ASSET_SHA256SUMS.txt`。
