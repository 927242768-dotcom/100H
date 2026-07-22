# 行人违法行为识别与标记

## 功能说明

系统在已有RKNN行人检测和光流跟随基础上增加“行人进入机动车道/禁入区域”判定：

1. 使用可配置多边形定义机动车道或行人禁入区。
2. 使用行人框底部中央的脚点判断是否进入区域，避免仅上半身与区域重叠造成误报。
3. 默认连续2次行人检测命中后才触发违法告警。
4. 默认连续2次检测离开区域后解除，检测短暂中断时最多保留0.75秒。
5. 普通行人显示绿色框，违法行人显示加粗橙红框，并标记“违法行为：行人进入机动车道”。
6. HDMI顶部显示违法告警，状态栏、日志和自动测试结果记录违法数量。

该功能默认关闭，不会影响原有车牌识别、普通行人检测或摄像头/SD输入。只有增加 `--enable-pedestrian-violation` 后才启用。

## 板端部署

把下列文件同步到 `/home/linaro/competition_arm/python/`：

- `pipeline.py`
- `pedestrian_violation.py`
- `evaluation_metrics.py`

## 运行参数

在当前已经能够运行车牌和行人检测的命令末尾增加：

```bash
  --enable-pedestrian-violation \
  --pedestrian-zone "0.05,0.55;0.95,0.55;1.0,1.0;0.0,1.0" \
  --pedestrian-zone-coordinates normalized \
  --pedestrian-violation-confirmations 2 \
  --pedestrian-violation-clear-hits 2 \
  --pedestrian-violation-hold-seconds 0.75
```

如果不传 `--pedestrian-zone`，系统自动使用上面这组默认梯形区域。

## 区域标定

归一化坐标范围为0到1，不受摄像头分辨率影响：

- 左上角为 `0,0`。
- 右下角为 `1,1`。
- 顶点按顺时针或逆时针排列。
- 区域可以是三角形、四边形或更多顶点的多边形。

启用后HDMI会显示橙色区域边界。根据实际道路画面调整 `--pedestrian-zone`，让边界只覆盖机动车道，不覆盖人行道和安全区域。例如道路位于画面右侧时可使用：

```bash
--pedestrian-zone "0.48,0.42;0.95,0.42;1.0,1.0;0.35,1.0"
```

现场调试完成后如不希望显示区域边界，可增加：

```bash
--hide-pedestrian-zone
```

## 关键调节参数

- `--pedestrian-violation-confirmations 2`：增大可减少误报，但告警更慢。
- `--pedestrian-violation-clear-hits 2`：增大可减少人员在边界附近导致的告警闪烁。
- `--pedestrian-violation-match-iou 0.15`：相邻检测结果的轨迹匹配阈值。
- `--pedestrian-violation-hold-seconds 0.75`：检测短暂中断时保留违法轨迹的时间。
- `--pedestrian-foot-y-ratio 0.95`：脚点在行人框纵向的位置，通常保持0.90到0.98。

推荐先保持默认值，只调整区域多边形。区域正确后再根据实际误报和告警速度调节确认次数。

## 自动测试输出

同时启用 `--metrics-dir` 后：

- `frames.csv` 增加 `violation_count`。
- `detections.csv` 增加 `kind=violation` 的违法记录。
- `summary.json` 增加 `pedestrian_violation_output_frames` 和 `pedestrian_violation_rows`。

答辩演示建议同时展示普通行人在安全区域不告警、进入机动车道后触发告警、离开后自动解除三个过程。
