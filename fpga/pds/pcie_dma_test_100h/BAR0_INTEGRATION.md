# BAR0 图像预处理集成

本说明只描述项目新增逻辑，不重新分发 PANGO 私有 PCIe/DMA 源码。

在 PDS 官方 PCIe DMA 示例工程的 BAR0 接口位置，将原 BAR0 RAM 实例替换为：

```verilog
rk3568_traffic_preprocess_fpga_v2 #(
    .ADDR_WIDTH(ADDR_WIDTH)
) u_rk3568_traffic_preprocess_fpga (
    .clk(clk),
    .rst_n(rst_n),
    .i_bar0_wr_en(bar0_wr_en),
    .i_bar0_wr_addr(bar0_wr_addr),
    .i_bar0_wr_data(bar0_wr_data),
    .i_bar0_wr_be(bar0_wr_byte_en),
    .i_bar0_rd_clk_en(i_bar0_rd_clk_en),
    .i_bar0_rd_addr(i_bar0_rd_addr),
    .o_bar0_rd_data(o_bar0_rd_data)
);
```

同时将以下文件加入工程源文件列表：

```text
../../src/preprocess_register_bank.v
../../src/rk3568_traffic_preprocess_fpga_v2.v
```

当前软件和正式 `.sbit` 使用单 BAR 共享窗口：

- `0x000` 至 `0x0ff`：寄存器和状态头。
- `0x100` 起：输入灰度帧；处理完成后从同一窗口读取掩码。
- 安全输入尺寸：`112 × 64`，宽度必须为 16 的整数倍。

重新构建后，应先运行 `arm/python/quickcheck.py` 验证状态签名 `0x54504650`，再运行完整管线。
