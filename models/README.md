# 部署模型

- `plate/yolov8s.rknn`：RK3568 车牌框检测，默认类别为单层车牌、双层车牌。
- `person/yolov8n.rknn`：RK3568 行人检测，用于可选的机动车道/禁入区违法判断。
- `hyperlpr/r2_mobile/`：HyperLPR OCR 使用的 MNN 模型。

程序会优先从仓库内寻找车牌 RKNN 与 HyperLPR 模型；也可以通过命令行参数覆盖路径。

模型完整性可使用根目录 `ASSET_SHA256SUMS.txt` 校验。
