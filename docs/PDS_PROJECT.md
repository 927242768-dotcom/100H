# PDS 工程与可复现性

## 已随仓库提供

- `fpga/pds/pcie_dma_test_100h/pcie_dma_test.pds`：本次上板工程入口。
- `fpga/pds/pcie_dma_test_100h/fdc/pcie_dma_test.fdc`：板级与时序约束。
- `fpga/pds/pcie_dma_test_100h/hdl/pcie_dma_test.v`：工程顶层。
- `fpga/pds/pcie_dma_test_100h/ipcore/pcie_test/pcie_test.idf`：PCIe IP 参数配置。
- `fpga/bitstream/pcie_dma_test.sbit`：已经通过实现并上板使用的正式烧录文件。
- `fpga/src/`：本项目自行开发的图像预处理 RTL。

## PDS 版本

使用 PDS `2022.2-SP6.4`，器件为 `PG2L100H-6-FBG484`。

最终报告为 `All Constraints Met`：

- 250 MHz `pclk` WNS：`+0.558 ns`
- 125 MHz `pclk_div2` WNS：`+0.492 ns`
- LUT 使用率约 15.2%
- 寄存器使用率约 3.7%
- DRM 使用率约 44.8%

## 为什么没有公开完整 IP RTL

原工程中的 PCIe IP 与部分 DMA 示例 RTL 带有 PANGO 私有源码声明，明确禁止未经书面授权复制或公开。因此公共仓库不能重新分发这些厂商文件。

克隆仓库后有两种使用方式：

1. **直接使用**：烧录仓库中的 `fpga/bitstream/pcie_dma_test.sbit`，不需要重新综合。
2. **重新构建**：安装对应版本 PDS，从官方开发板资料或 PDS IP Compiler 生成 PCIe 示例工程，再按 `fpga/pds/pcie_dma_test_100h/BAR0_INTEGRATION.md` 接入本仓库的 `rk3568_traffic_preprocess_fpga_v2.v`。

## PDS 路径说明

原始工程文件曾引用本机路径：

```text
../../../competition_solution/fpga/src/rk3568_traffic_preprocess_fpga_v2.v
```

仓库中的工程应改为相对于 `fpga/pds/pcie_dma_test_100h` 的：

```text
../../src/rk3568_traffic_preprocess_fpga_v2.v
```

如果 PDS 打开后仍显示旧路径，请在工程源文件列表中移除旧条目并添加上述仓库内文件。
