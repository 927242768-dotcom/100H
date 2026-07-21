# ARM 侧 Python 联调说明

这个目录用于板子 Linux 侧快速联调。当前已经打通：

1. USB 摄像头采图
2. 通过 PCIe BAR0 把灰度图送入 FPGA
3. 读回 FPGA 生成的二值掩码
4. 在 ARM 侧做候选框筛选
5. 调用 HyperLPR 做车牌识别
6. 直接输出到 HDMI 实时显示

## 当前脚本

1. `camera_probe.py`
   最小摄像头检查，抓拍一张图保存到本地。
2. `quickcheck.py`
   Python 版 BAR0 最小闭环检查。
3. `pipeline.py`
   摄像头 + FPGA 预处理 + HyperLPR + HDMI 显示主循环。
4. `fpga_client.py`
   FPGA BAR0 寄存器和帧缓冲访问封装。
5. `detector.py`
   检测器接口层，当前支持 `MockDetector` 和 `HyperLprDetector`。

## 当前 FPGA 限制

当前这版 FPGA 工程里，`BAR0` 安全帧区大约只有 `7936` 字节。

因此现在建议的 FPGA 输入分辨率先用：

- `112x64`
- `96x64`
- `64x64`

先不要直接用 `320x180`，否则会超出当前 BAR0 可用帧区。

## 每次热下载新 sbit 后都要执行

由于板载 Flash 还没固化成功，当前仍然走 `sbit` 热下载。

每次下载新 `sbit` 后，在板子上执行：

```bash
sudo sh -c 'echo 1 > /sys/bus/pci/devices/0002:21:00.0/remove'
sudo sh -c 'echo 1 > /sys/bus/pci/rescan'
sudo sh -c 'echo 1 > /sys/bus/pci/devices/0002:21:00.0/enable'
sudo setpci -s 0002:21:00.0 COMMAND=0006
```

然后先检查签名：

```bash
sudo busybox devmem 0xf0200000 32
```

正常应该看到：

```bash
0x54504650
```

## 1. 先检查摄像头

```bash
python3 camera_probe.py --camera 0 --width 1280 --height 720 --output camera_probe.png
```

如果成功，会生成 `camera_probe.png`。

## 2. 再跑 FPGA 最小闭环

```bash
sudo python3 quickcheck.py --resource-root /sys/bus/pci/devices/0002:21:00.0
```

## 3. 先用 headless 模式验证链路

如果你还在 SSH 里调试，建议先用 `headless + save-dir`：

```bash
sudo python3 pipeline.py \
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
  --headless \
  --max-frames 120 \
  --save-dir pipeline_out \
  --save-every 20
```

运行后会在 `pipeline_out` 里保存：

- `frame_*.png`
- `mask_*.png`
- `display_*.png`

## 4. HyperLPR 接入说明

`pipeline.py` 现在支持：

- `--detector mock`
  不做二阶段识别，只看 FPGA 候选框。
- `--detector hyperlpr`
  对 FPGA 候选框逐个调用 HyperLPR 做车牌检测与识别。

当前 `HyperLprDetector` 会优先尝试下面这些板端默认路径：

- `libhyperlpr3.so`
  - `/userdata/HyperLPR/HyperLPR/Prj-Linux/hyperlpr3/lib/libhyperlpr3.so`
  - `/userdata/HyperLPR/HyperLPR/build/linux/install/hyperlpr3/lib/libhyperlpr3.so`
- `r2_mobile`
  - `/userdata/HyperLPR/HyperLPR/Prj-Linux/hyperlpr3/resource/models/r2_mobile`
  - `/userdata/HyperLPR/HyperLPR/build/linux/install/hyperlpr3/resource/models/r2_mobile`
- `libMNN.so`
  - `/userdata/HyperLPR/HyperLPR/3rdparty_hyper_inspire_op/MNN-2.2.0/linux/lib/libMNN.so`
  - `/userdata/MNN/build-linux/libMNN.so`

如果你板子上的实际路径不同，可以显式传：

- `--hyperlpr-lib`
- `--hyperlpr-model-dir`
- `--mnn-lib`

常用调参项：

- `--detector-max-rois 4`
  每帧最多送多少个 FPGA 候选框给 HyperLPR。
- `--detector-source full`
  优先跑全帧检测，通常比只喂小 ROI 更容易真正识别到车牌。
- `--detector-interval 3`
  每隔 3 帧才真正跑一次 HyperLPR，其余帧复用上次结果，能明显减轻掉帧。
- `--detector-input-width 960`
  全帧检测前先把宽度压到 960，再把结果映射回原图。
- `--detector-min-score 0.50`
  只绘制高于这个分数的车牌结果。
- `--hyperlpr-max-num 6`
  HyperLPR 每帧最多返回多少个车牌。
- `--hyperlpr-threads 1`
  RK3568 上先建议从 1 线程起步。
- `--hyperlpr-no-half`
  如果后续怀疑 FP16 稳定性，再临时关掉。

## 5. HDMI 实时显示

`pipeline.py` 当前支持 4 种显示模式：

- `--display-mode outline`
  默认模式，只在原图上画真正识别出的车牌框；只有显式加 `--draw-roi` 才会显示 ROI 框。
- `--display-mode overlay`
  原图叠加掩码。
- `--display-mode camera`
  只显示摄像头原图。
- `--display-mode mask`
  只显示 FPGA 掩码。

其他常用参数：

- `--fullscreen`
  全屏显示。
- `--display-width 1920 --display-height 1080`
  按 HDMI 分辨率缩放窗口。
- `--draw-roi`
  额外显示候选区域框。
- `--text-font /path/to/platech.ttf`
  指定中文绘制字体；留空时会优先查找 HyperLPR 自带 `platech.ttf`。
- `--text-font-size 28`
  中文绘制字体大小。
- `--hide-status`
  隐藏底部状态栏。
- `--camera-backend v4l2`
  摄像头长时间运行更稳。
- `--camera-read-retries 8 --camera-retry-delay 0.2`
  读帧失败后自动重试，避免瞬时异常直接退出。
- `--threshold-mode percentile`
  按当前画面亮度自适应调 FPGA 阈值。
- `--mask-cleanup open_close`
  ARM 侧对 FPGA 掩码再做一次轻量开闭运算清理。
- `--mask-max-area-ratio 0.35`
  去掉覆盖过大的误检区域，避免整屏被染色。

### 纯 FPGA HDMI 命令

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
  --fullscreen
```

### FPGA + HyperLPR + HDMI 命令

先把运行库路径带上：

```bash
cd /home/linaro/competition_arm/python
export DISPLAY=:0
export XAUTHORITY=/home/linaro/.Xauthority
export LD_LIBRARY_PATH=/userdata/HyperLPR/HyperLPR/Prj-Linux/hyperlpr3/lib:/userdata/HyperLPR/HyperLPR/3rdparty_hyper_inspire_op/MNN-2.2.0/linux/lib:$LD_LIBRARY_PATH
```

然后运行：

```bash
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
  --detector hyperlpr \
  --detector-source full \
  --detector-interval 5 \
  --detector-input-width 960 \
  --detector-min-score 0.45 \
  --hyperlpr-max-num 6 \
  --hyperlpr-threads 1 \
  --fullscreen
```

如果你的安装路径不是默认值，再补：

```bash
  --hyperlpr-lib /实际/libhyperlpr3.so \
  --hyperlpr-model-dir /实际/r2_mobile \
  --mnn-lib /实际/libMNN.so
```

## 6. 车牌文本显示说明

当前版本会优先使用 Pillow + TrueType 字体直接绘制中文。

如果板子环境里没有 Pillow，会退化成 ASCII 显示，并在终端打印警告。这种情况下安装：

```bash
sudo -E python3 -m pip install pillow
```

正常情况下：

- 车牌框上方显示：`苏ED51712`
- 底部状态栏上方显示：`绿牌新能源, 苏ED51712`
- 多车牌时会按 `A | B | C` 的形式拼接显示

## 7. 第二版预处理实验参数

当前第二版 V2.1-lite 已经把 `morph_cfg` 重新定义成：

- bit0: `invert`
- bit1: `passthrough_gray`
- bit2: `enable_sobel`
- bit5:4: `denoise_mode`
  - `00`: off
  - `01`: gauss3x3
- bit7:6: `morph_mode`
  - `00`: off
  - `01`: open
  - `10`: close
  - `11`: open_then_close

当前实现的是轻量可综合版 V2.1-lite：

1. `gauss` 先做成按字内邻域的轻量平滑
2. `sobel` 先做成按字内邻域的轻量梯度
3. `open/close` 先做成按字内邻域的轻量形态学

你现在可以直接试这些值：

- `0x0000`
  原始阈值二值化
- `0x0010`
  `gauss3x3 + threshold`
- `0x0004`
  `sobel + threshold`
- `0x0050`
  `gauss3x3 + open`
- `0x0090`
  `gauss3x3 + close`
- `0x00D0`
  `gauss3x3 + open_then_close`

## 8. SSH 和 HDMI 的区别

如果你只是通过 SSH 运行，而板子当前没有图形桌面会话，`cv2.imshow()` 没法直接把窗口送到 HDMI。

所以实际联调建议分两步：

1. 先在 SSH 里用 `--headless` 验证链路
2. 再到板端图形终端，或者先正确设置 `DISPLAY` / `XAUTHORITY`，再运行 HDMI 实时显示命令

如果 `DISPLAY` 没配好，`pipeline.py` 会明确提示当前不能直接输出到 HDMI。
