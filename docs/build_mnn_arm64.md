# 构建 Linux ARM64 版 MNN

适用场景：

1. 板子上的 `HyperLPR` 只能找到 `x86-64` 版 `libMNN.so`
2. 板子空间较小，不适合继续在板子上下载和重编 MNN
3. 需要在 PC / WSL / Ubuntu 主机上先交叉编译出 `aarch64` 版 `libMNN.so`

## 目标

最终只需要得到一个能在 RK3568 Linux 上使用的：

- `libMNN.so`

然后把它替换到板子上的：

```text
/userdata/HyperLPR/HyperLPR/3rdparty_hyper_inspire_op/MNN-2.2.0/linux/lib/libMNN.so
```

## 推荐主机环境

推荐使用：

1. Ubuntu PC
2. 或 Windows + WSL2 Ubuntu

## 1. 安装交叉编译工具

在 Ubuntu / WSL 中执行：

```bash
sudo apt update
sudo apt install -y git cmake python3 gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

## 2. 获取 MNN 源码

为了尽量和 HyperLPR 当前依赖版本对齐，优先使用 `2.2.0`：

```bash
git clone https://github.com/alibaba/MNN.git
cd MNN
git checkout 2.2.0
```

如果主机环境访问 GitHub 失败，也可以直接复用板子上已经拉下来的源码：

```bash
scp -r linaro@<板子IP>:/userdata/HyperLPR/HyperLPR/build/linux/_deps/mnn-src ./MNN
cd MNN
```

这套源码就是你前面在板子上跑 `HyperLPR` 构建脚本时自动下载到 `_deps/mnn-src` 的那份，适合当前“板子空间紧张、主机网络受限”的情况。

如果你**不想从板子上取源码**，也可以直接走 `Windows 主机拿源码 -> 复制到 Ubuntu 虚拟机`：

1. 在 `Windows` 上下载 `MNN 2.2.0` 源码压缩包，或在 `Windows` 上先执行：

```powershell
git clone --branch 2.2.0 --depth 1 https://github.com/alibaba/MNN.git D:\100H\MNN
```

2. 如果 `Windows` 上也不方便用 `git`，就直接在浏览器里下载：

```text
https://github.com/alibaba/MNN/archive/refs/tags/2.2.0.zip
```

下载后解压到：

```text
D:\100H\MNN
```

3. 再把这个目录整体复制到 `Ubuntu 虚拟机`，例如放到：

```bash
~/MNN
```

4. 然后从 `~/MNN` 继续后面的补丁和交叉编译步骤。

如果你的 `Ubuntu 虚拟机` 无法直接 `ssh/scp` 到板子，也可以走 `Windows 主机中转`：

1. 先在板子上打包源码：

```bash
cd /userdata/HyperLPR/HyperLPR/build/linux/_deps
tar czf /home/linaro/mnn-src.tar.gz mnn-src
```

2. 用 `MobaXterm` 左侧 `SFTP` 面板把 `/home/linaro/mnn-src.tar.gz` 下载到 Windows，例如：

```text
D:\100H\mnn-src.tar.gz
```

3. 再把这个压缩包复制到虚拟机里，解压后继续后面的补丁和交叉编译步骤：

```bash
mkdir -p ~/MNN
tar xzf ~/mnn-src.tar.gz --strip-components=1 -C ~/MNN
cd ~/MNN
```

## 3. 先打 ARM 汇编兼容补丁

如果你直接在 ARM/Linux 上编 MNN 2.2.0，可能会遇到：

```text
MNNGemmInt8AddBiasScale_ARMV86_Unit.S
mov v1.4s,v0.4s
operand mismatch
```

为避免同类问题，先执行补丁脚本：

```bash
python3 /mnt/d/100H/competition_solution/tools/patch_mnn_armv86_mov.py \
  ./source/backend/cpu/arm/arm64/MNNGemmInt8AddBiasScale_ARMV86_Unit.S
```

如果你使用的是 `WSL`，上面的 `/mnt/d/100H/...` 路径可以直接用。

如果你使用的是 `Ubuntu 虚拟机`，它不是 `WSL`，通常没有 `/mnt/d/...` 这种路径。这时有两种方式：

1. 先把 `patch_mnn_armv86_mov.py` 复制到虚拟机里，再用虚拟机本地路径执行
2. 或者直接在虚拟机里用下面这条命令完成同样的补丁：

```bash
perl -0pi.bak -e 's/\bmov\s+(v\d+)\.4s,\s*(v\d+)\.4s\b/mov $1.16b, $2.16b/g; s/\bmov\s+(\\[A-Za-z0-9_]+\\\(\))\.4s,\s*(\\[A-Za-z0-9_]+\\\(\))\.4s\b/mov $1.16b, $2.16b/g' \
  ./source/backend/cpu/arm/arm64/MNNGemmInt8AddBiasScale_ARMV86_Unit.S
```

如果交叉编译时又遇到 `CPUFixedPoint.hpp` 里 `std::int32_t` / `std::int16_t` 相关报错，说明当前编译环境需要额外补上 `cstdint` 头文件。可以继续执行：

```bash
python3 /mnt/d/100H/competition_solution/tools/patch_mnn_cpufixedpoint.py \
  ./source/backend/cpu/CPUFixedPoint.hpp
```

如果你使用的是 `Ubuntu 虚拟机`，也可以直接用一条命令改：

```bash
perl -0pi.bak -e 's/#include <stdint.h>/#include <stdint.h>\n#include <cstdint>/' \
  ./source/backend/cpu/CPUFixedPoint.hpp
```

## 4. 交叉编译 ARM64 版 libMNN.so

使用我准备好的工具链文件：

```bash
cmake -S . -B build-arm64 \
  -DCMAKE_TOOLCHAIN_FILE=/mnt/d/100H/competition_solution/tools/mnn_aarch64_toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DMNN_BUILD_SHARED_LIBS=ON \
  -DMNN_BUILD_TOOLS=OFF \
  -DMNN_BUILD_CONVERTER=OFF \
  -DMNN_BUILD_TRAIN=OFF \
  -DMNN_BUILD_DEMO=OFF \
  -DMNN_BUILD_BENCHMARK=OFF \
  -DMNN_BUILD_TEST=OFF \
  -DMNN_EVALUATION=OFF \
  -DMNN_OPENMP=OFF \
  -DMNN_USE_THREAD_POOL=ON \
  -DMNN_ARM82=OFF \
  -DMNN_SME2=OFF \
  -DMNN_SUPPORT_BF16=OFF \
  -DMNN_KLEIDIAI=OFF
```

如果你用的是 `Ubuntu 虚拟机`，请先把 `mnn_aarch64_toolchain.cmake` 复制进虚拟机，然后把命令里的工具链路径改成虚拟机本地路径，例如：

```bash
cmake -S . -B build-arm64 \
  -DCMAKE_TOOLCHAIN_FILE=~/mnn_aarch64_toolchain.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DMNN_BUILD_SHARED_LIBS=ON \
  -DMNN_BUILD_TOOLS=OFF \
  -DMNN_BUILD_CONVERTER=OFF \
  -DMNN_BUILD_TRAIN=OFF \
  -DMNN_BUILD_DEMO=OFF \
  -DMNN_BUILD_BENCHMARK=OFF \
  -DMNN_BUILD_TEST=OFF \
  -DMNN_EVALUATION=OFF \
  -DMNN_OPENMP=OFF \
  -DMNN_USE_THREAD_POOL=ON \
  -DMNN_ARM82=OFF \
  -DMNN_SME2=OFF \
  -DMNN_SUPPORT_BF16=OFF \
  -DMNN_KLEIDIAI=OFF
```

然后编译：

```bash
cmake --build build-arm64 -j8
```

## 5. 验证产物

编译完成后检查：

```bash
file build-arm64/libMNN.so
```

正常应该看到类似：

```text
ELF 64-bit LSB shared object, ARM aarch64
```

## 6. 拷回板子

把它传到板子上，例如：

```bash
scp build-arm64/libMNN.so linaro@<板子IP>:/home/linaro/
```

## 7. 在板子上替换旧库

先备份旧的 x86 库：

```bash
mv /userdata/HyperLPR/HyperLPR/3rdparty_hyper_inspire_op/MNN-2.2.0/linux/lib/libMNN.so \
   /userdata/HyperLPR/HyperLPR/3rdparty_hyper_inspire_op/MNN-2.2.0/linux/lib/libMNN.so.x86_64.bak
```

再替换成新的 ARM64 库：

```bash
cp /home/linaro/libMNN.so \
   /userdata/HyperLPR/HyperLPR/3rdparty_hyper_inspire_op/MNN-2.2.0/linux/lib/libMNN.so
```

然后确认：

```bash
file /userdata/HyperLPR/HyperLPR/3rdparty_hyper_inspire_op/MNN-2.2.0/linux/lib/libMNN.so
```

## 8. 重新编译 HyperLPR

回到 HyperLPR 根目录重新执行：

```bash
cd /userdata/HyperLPR/HyperLPR
rm -rf build/linux
mkdir -p build/linux
cd build/linux
cmake -DCMAKE_BUILD_TYPE=Release -DLINUX_FETCH_MNN=OFF -DBUILD_SHARE=ON -DBUILD_SAMPLES=OFF -DBUILD_TEST=OFF ../..
make -j2
make install
```

如果这一步通过，再继续编 `Prj-Linux`。

## 9. 如果 Prj-Linux 链接 PlateRecDemo 时找不到 libMNN.so

典型报错长这样：

```text
/usr/bin/ld: warning: libMNN.so, needed by .../libhyperlpr3.so, not found
/usr/bin/ld: .../libhyperlpr3.so: undefined reference to `MNN::...`
```

这说明：

1. `libhyperlpr3.so` 已经编出来了
2. 但是 `Prj-Linux` 在链接 `PlateRecDemo` 时，没有把 `MNN` 运行库一起带上

这时需要改板子上的：

```text
/userdata/HyperLPR/HyperLPR/Prj-Linux/CMakeLists.txt
```

把 `MNN` 的库目录和链接项补进去。可以按下面这个思路修改：

```cmake
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/hyperlpr3/include)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/hyperlpr3/lib)
link_directories(/userdata/HyperLPR/HyperLPR/3rdparty_hyper_inspire_op/MNN-2.2.0/linux/lib)

add_executable(PlateRecDemo plate_rec_demo.cpp)

target_link_libraries(PlateRecDemo
    hyperlpr3
    MNN
    ${OpenCV_LIBS}
)

set_target_properties(PlateRecDemo PROPERTIES
    BUILD_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/hyperlpr3/lib;/userdata/HyperLPR/HyperLPR/3rdparty_hyper_inspire_op/MNN-2.2.0/linux/lib"
)
```

注意：

1. `hyperlpr3` 要放在 `MNN` 前面
2. 只设置 `LD_LIBRARY_PATH` 适合运行时，不足以解决当前这个链接阶段报错

改完后重新编：

```bash
cd /userdata/HyperLPR/HyperLPR/Prj-Linux
rm -rf build
bash build.sh
find build -type f -executable | sort
```

如果编成功，再运行：

```bash
cd /userdata/HyperLPR/HyperLPR/Prj-Linux/build
export LD_LIBRARY_PATH=../hyperlpr3/lib:/userdata/HyperLPR/HyperLPR/3rdparty_hyper_inspire_op/MNN-2.2.0/linux/lib:$LD_LIBRARY_PATH
./PlateRecDemo ../hyperlpr3/resource/models/r2_mobile ../hyperlpr3/resource/images/test_img.jpg
```

## 10. 如果新编出来的 libMNN.so 仍然提示 GLIBC / GLIBCXX 版本不兼容

典型报错长这样：

```text
libmvec.so.1 not found
undefined reference to `_ZGVnN4v_logf@GLIBC_2.38'
undefined reference to `std::ios_base_library_init()@GLIBCXX_3.4.32'
```

这说明：

1. 你在 `Ubuntu/WSL` 上交叉编出来的 `libMNN.so` 架构虽然是 `ARM aarch64`
2. 但它链接的是主机环境里更高版本的 `glibc/libstdc++`
3. 板子系统版本更低，所以 `HyperLPR` 主库虽然能编过去，`Prj-Linux` 在最终链接 `PlateRecDemo` 时还是会炸

这时最稳的方案不是继续折腾交叉编译参数，而是：

**直接在板子上本机编译一份最小化的 MNN 动态库**

这样编出来的 `libMNN.so` 会天然匹配板子自己的 `glibc/libstdc++` 版本。

### 板子本机最小编译建议

1. 用 `MobaXterm` 把 `Windows` 上的 `D:\100H\MNN.tar.gz` 上传到板子，例如：

```text
/userdata/MNN.tar.gz
```

2. 在板子上解压：

```bash
cd /userdata
rm -rf MNN
tar xzf MNN.tar.gz
cd MNN
```

3. 打补丁：

```bash
perl -0pi.bak -e 's/\bmov\s+(v\d+)\.4s,\s*(v\d+)\.4s\b/mov $1.16b, $2.16b/g; s/\bmov\s+(\\[A-Za-z0-9_]+\\\(\))\.4s,\s*(\\[A-Za-z0-9_]+\\\(\))\.4s\b/mov $1.16b, $2.16b/g' \
  ./source/backend/cpu/arm/arm64/MNNGemmInt8AddBiasScale_ARMV86_Unit.S

perl -0pi.bak -e 's/#include <stdint.h>/#include <stdint.h>\n#include <cstdint>/' \
  ./source/backend/cpu/CPUFixedPoint.hpp
```

4. 用尽量小的配置本机编译：

```bash
cmake -S . -B build-linux \
  -DCMAKE_BUILD_TYPE=Release \
  -DMNN_BUILD_SHARED_LIBS=ON \
  -DMNN_BUILD_TOOLS=OFF \
  -DMNN_BUILD_CONVERTER=OFF \
  -DMNN_BUILD_TRAIN=OFF \
  -DMNN_BUILD_DEMO=OFF \
  -DMNN_BUILD_BENCHMARK=OFF \
  -DMNN_BUILD_TEST=OFF \
  -DMNN_EVALUATION=OFF \
  -DMNN_OPENMP=OFF \
  -DMNN_USE_THREAD_POOL=ON
```

5. 为了减小内存和空间压力，用单线程编译：

```bash
cmake --build build-linux -j1
```

6. 编好后确认：

```bash
file /userdata/MNN/build-linux/libMNN.so
```

7. 然后再把这份 **板子本机编出来的** `libMNN.so` 替换到：

```bash
/userdata/HyperLPR/HyperLPR/3rdparty_hyper_inspire_op/MNN-2.2.0/linux/lib/libMNN.so
```

8. 最后重新编：

```bash
cd /userdata/HyperLPR/HyperLPR
rm -rf build/linux
mkdir -p build/linux
cd build/linux
cmake -DCMAKE_BUILD_TYPE=Release -DLINUX_FETCH_MNN=OFF -DBUILD_SHARE=ON -DBUILD_SAMPLES=OFF -DBUILD_TEST=OFF ../..
make -j2
make install

cd /userdata/HyperLPR/HyperLPR/Prj-Linux
rm -rf hyperlpr3 build
cp -r ../build/linux/install/hyperlpr3 ./
bash build.sh
```

### 什么时候应该切到这条路线

只要你已经看到下面任意一种符号版本报错，就可以直接切到“板子本机编 MNN”：

1. `GLIBC_2.38`
2. `GLIBC_2.39`
3. `GLIBCXX_3.4.32`
4. `libmvec.so.1 not found`

## 说明

这套方案的核心是：

1. 不在板子上下载和重编 MNN
2. 只在主机上产出一个 `ARM64` 的 `libMNN.so`
3. 板子上只做替换和重新链接 HyperLPR

这样最省板子空间，也最适合当前环境。
