# 依赖、来源与本机布局

用户指定 FlashInfer 0.6.17 wheel 的隔离安装、SM12.1 FA2 prefill 适配与对照测试见
[本轮复现流程](../../../docs/guides/windows-flashinfer-comparison.md)。该实验依赖与下面保留的基础环境记录分别管理。

## 已验证环境

| 类别 | 版本或路径（相对仓库根目录） | 用途 |
| --- | --- | --- |
| OS / Python | Windows ARM64 / CPython 3.13.16，.venv/Scripts/python.exe | 本机进程，不是 x64 Python 仿真 |
| GPU | RTX Spark N1X，SM12.1，CUDA_VISIBLE_DEVICES=0 | GPU 1 不作为本轮 CUDA GPU 使用 |
| MSVC | 工具目录 .port/msvc/Contents/VC/Tools/MSVC/14.51.36231 | Hostarm64/arm64/cl.exe 与 link.exe |
| 实际编译器 | 19.51.36260，_MSC_VER=1951，_MSC_FULL_VER=195136260 | 14.51 工具集的真实版本 |
| Windows SDK | 10.0.26100.0 | 从本机 Visual Studio 环境取得 ARM64 headers/libs |
| CUDA SDK | C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4 | version.json 的 SDK 版本为 13.4.0 |
| CUDA nvcc | 13.4.46 | 组件版本与 SDK 发行版本不同 |
| CMake / Ninja | 4.4.4 / Python ninja 1.13.2 | CMake 为系统工具，不是 Python cmake 包 |
| Rust | 1.92.0，aarch64-pc-windows-msvc | .port/rust-toolchain/bin，Cargo.lock 锁定 crate |
| xgrammar C++ 源码 | 0.2.8，.port/xgrammar-0.2.8 | engine 静态编译 core，不使用 Python wheel 内的 headers |
| GoogleTest | v1.14.0，.port/googletest | 仅 scheduler C++ 测试需要 |

完整 Python 快照（219 个实际解析到的 distribution，包括共享环境中的其他工具）
见 [版本文本](https://github.com/jinkun_microsoft/windows-tokenspeed/blob/9a50ac82b59fc47044090e4441c05c88173715c1/docs/validation/windows-arm64-fp8-20261027/inventory/python-environment.txt)
和 [路径、wheel tag、Requires-Dist 元数据](https://github.com/jinkun_microsoft/windows-tokenspeed/blob/9a50ac82b59fc47044090e4441c05c88173715c1/docs/validation/windows-arm64-fp8-20261027/inventory/python-environment.json)。
它是观测快照，不是应直接 pip install 的最小 lock 文件。

## Python 构建与运行依赖

| 包 | 本机实际版本 | 角色 |
| --- | --- | --- |
| setuptools / wheel | 80.10.2 / 0.48.0 | wheel 构建 backend |
| scikit-build-core / nanobind | 1.1.1 / 3.1.0 | scheduler CMake backend 与绑定 |
| tokenspeed-spdlog | 1.15.1 | C++ headers / CMake package |
| PyTorch | 2.14.0a0+git20b0f57，cp313-win_arm64，CUDA 13.4 | 必须有原生 ARM64 CUDA 构建 |
| torchvision | 0.29.0a0+10f68db | 共享环境中的 PyTorch 配套包 |
| apache-tvm-ffi | 0.1.13.post3，cp313-win_arm64 | CUDA TVM FFI ABI、headers 和 tvm_ffi.lib |
| triton-windows | 3.8.0.post29 | SM12x JIT 内核，经 tokenspeed_triton 边界使用 |
| flashinfer-python | 0.7.2 | 头文件、打包数据及可用的 Python 功能 |
| transformers / compressed-tensors | 5.12.0 / 0.17.0 | 模型及压缩检查点加载 |
| numpy / packaging / psutil | 2.5.4 / 26.3 / 7.0.0 | 数组、版本及系统信息 |
| msgspec / pyzmq / pydantic | 0.22.0 / 26.4.0 / 2.14.1 | 控制协议、IPC、配置 |
| nvidia-ml-py | 13.615.71 | NVML 查询 |
| pytest / pytest-timeout | 9.1.1 / 2.4.0 | 回归测试 |

声明依赖的权威来源是 python/pyproject.toml、tokenspeed-engine/pyproject.toml、
tokenspeed-scheduler/pyproject.toml 和 tokenspeed-kernel/python/requirements/windows-cuda.txt。
kernel 的 dense-triton profile 只选择 windows-cuda.txt，不安装 Linux 全量 vendor
依赖，也跳过 setup.py 内部的嵌套 pip 安装；这些 ARM64 依赖需提前准备。

主包声明 setuptools==83.0.0，但这次 --no-build-isolation 实际使用 80.10.2。
这证明本机已准备环境能构建，不证明已验证隔离构建及声明依赖的全量解算。
本机 Python xgrammar 为 0.2.1；engine 静态 core 使用单独的 0.2.8 源码。
主包声明 Python xgrammar==0.2.8，而验证关闭 grammar_backend，因此不声称
Python grammar 路径已验证。主包还声明未移植的 SMG、mooncake 等服务依赖。
不要通过一次无限制 pip install -U 把已验证的 ARM64 包替换掉。

## 已有 venv 与新机器准备

已有环境 .venv/Lib/site-packages/local-runtime.pth 指向
../../vllm-bench/.venv/Lib/site-packages（相对此 official-repos 工作区）。当前 torch、TVM FFI 等包从该共享目录解析；
本仓库四个 wheel 安装在自己的 .venv 中，优先使用本地安装版本。

另一台机器需要原生 ARM64 Python，以及等价 ARM64 torch/TVM FFI 等 wheel。
本机这些 wheel 的准备位置为 ../../N1X_ARM_Package_Support/wheelhouse，属于本机
前提，不是可联网下载的公共安装源。不要用 x64 wheel 替代，也不要声称已验证
干净机器安装。本次未重新构建这些第三方依赖。

有受信 ARM64 wheelhouse 时，可在项目 venv 内按实际所需包安装：

```powershell
$py = '.venv/Scripts/python.exe'
$wheelhouse = '.port/wheelhouse' # 已复制的 ARM64 torch / TVM FFI wheel
& $py -m pip install --no-deps --no-index --find-links $wheelhouse 'torch==2.14.0a0+git20b0f57' 'apache-tvm-ffi==0.1.13.post3'
if ($LASTEXITCODE -ne 0) { throw 'ARM64 dependency installation failed' }
```

此命令仅安装指定依赖；其余运行依赖按 metadata 和实际功能补齐。源码归档、
wheel tag 与构建身份均需核对。不要把整个观测快照作为新机器必须安装的清单。

## 工具链来源与校验

MSVC 官方入口为 https://aka.ms/vs/18/stable/channel 。历史选择的 package URL、
版本和 SHA256 位于
[MSVC payload 清单](https://github.com/jinkun_microsoft/windows-tokenspeed/blob/9a50ac82b59fc47044090e4441c05c88173715c1/docs/validation/windows-arm64-20261026/msvc-packages.json)。
将 VSIX 作为 ZIP 解压到 .port/msvc，保留 Contents/VC/... 布局，合并编译器、
CRT ARM64 Desktop、CRT Headers 和语言资源；Store payload 不是 Desktop 构建必需。
工具目录名不能替代编译器版本检查。raw VSIX 不包含 nvcc 需要的 Auxiliary
环境脚本，需本机安装的 VS 提供官方 vcvarsall.bat，runner 才能复制恢复。

Rust 归档来源为
https://static.rust-lang.org/dist/rust-1.92.0-aarch64-pc-windows-msvc.tar.xz ，
其 SHA256 为 cd9f4ceb634329b920285998d451cb16df59d7ba6e7fade49c39c9194a6981b8。
解包安装到 .port/rust-toolchain；Cargo 缓存使用 .port/cargo。
xgrammar 下载 URL 与固定 SHA256 由
tokenspeed-engine/crates/ts-grammar/CMakeLists.txt 的 FetchContent 声明管理；
预解压到 .port/xgrammar-0.2.8 后设置 FETCHCONTENT_SOURCE_DIR_XGRAMMAR 可离线复用。
Cargo 首次 fetch 和 CMake 首次 FetchContent 需要网络或完整缓存。

[下载缓存审计](https://github.com/jinkun_microsoft/windows-tokenspeed/blob/9a50ac82b59fc47044090e4441c05c88173715c1/docs/validation/windows-arm64-fp8-20261027/inventory/download-cache-audit.json)
记录当前八个归档的 URL、实际哈希和预期哈希；本轮全部匹配。新下载仍需重新
校验，不能从一个编译成功的 cl.exe 反推所有归档都可信。CUDA 的 13.4.92
探测归档、.port/cuda-patch* 和旧 14.44 frontend helper 均不是最终构建输入。

Windows scheduler 使用系统 bcrypt/CNG，不需要为本流程额外安装 OpenSSL DLL。
CUDA 库使用 SDK 的 arm64 libs、系统驱动和 ARM64 MSVC runtime。安装程序和
本机 VS/SDK 不是四个项目 wheel 的一部分。
