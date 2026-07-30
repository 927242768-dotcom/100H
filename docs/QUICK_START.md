# 快速开始

仓库包含 ARM/FPGA 源码、正式烧录文件、RKNN/MNN 模型和离线依赖。

板端执行：

```bash
python3 arm/python/install_rknnlite_from_wheel.py third_party/wheels/*.whl --clean
```

然后进入 `arm/python` 运行 `pipeline.py`。完整参数见根目录 README。
