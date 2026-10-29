# WSL2 ARM64 官方源码构建与 Windows 对比

本流程在 2026-10-28 重启 Windows 后执行。旧的重启前记录保留在
[原报告](../validation/windows-wsl2-perf-20261028/README.md)，不累计为本轮测试。
本轮结果、原生 ELF 审计、逐轮吞吐与输出 IDs 见
[重启后报告](../validation/windows-wsl2-perf-20261028-after-reboot/README.md)。

## 环境和源码

| 项目 | 实际环境 |
| --- | --- |
| WSL | 3.0.3.0，内核 6.18.54.1-microsoft-standard-WSL2 |
| Ubuntu | 26.04 ARM64，glibc 2.43 |
| GPU | RTX Spark N1X，SM12.1，驱动 615.89 |
| Python | 隔离 CPython 3.13.16，与 Windows 相同 |
| PyTorch | 官方 PyPI 2.14.0，torch.__version__ 为 2.14.0+cu130 |
| PyTorch CUDA 运行库 | 探测到实际加载 CUDA 13.0；详见 runtime-toolchain.json |
| AOT CUDA 编译器 | 官方 cuda-toolkit 13.4.0rc1，nvcc 13.4.46 |
| C++ / Rust | GCC 15.2.0 / Rust 1.92.0 |
| Linux 工作目录 | /opt/tokenspeed-work，位于此仓库 .port/wsl2-ubuntu 的 VHDX 内 |
| 官方源码 | https://github.com/lightseekorg/tokenspeed，固定 864bcdd9ef3c5f1512bbe293d970d73eb29d2c94 |

Linux 使用官方源码，与 Windows 移植 9a50ac82 的共同上游基线对齐。
工作目录、venv 和模型放在 Linux 文件系统，Windows 的 win_arm64 wheel 不安装到 Linux。
模型与测试脚本保持一致；新旧两轮的报告文件分别保存。

## 已解决的安装问题

Ubuntu 首次软件包配置曾中断，通过 dpkg --configure -a 恢复，随后完成构建工具安装。
CUDA 13.0 nvcc 与 Ubuntu 26.04 的 glibc 数学声明发生 rsqrt/rsqrtf noexcept 冲突。
使用独立官方 CUDA 13.4 SDK 后，SM121a 编译及 4097 元素 GPU 向量加法通过。
PyPI 对应版本名为 13.4.0rc1，其 nvcc 组件为 13.4.46rc1，工具输出为 V13.4.46。
独立 SDK 必须包含 cuRAND 头文件；首次缺少它导致采样相关编译失败，日志予以保留。

cuSPARSELt 0.8.1 的供应商 WHEEL 内部标签为 manylinux2014_sbsa，pip check 报平台不支持；
已检查 libcusparseLt.so.0 的实际 ELF 为 AArch64，并通过实际 CUDA/BF16 探测。
没有改写该供应商 wheel 元数据；环境不能宣称 pip check 全绿。

## 安装前提与构建命令

以下在新建的 Ubuntu 中使用 root 执行。root 用于自动化安装，Python 包均放在隔离 venv。
Ubuntu 镜像和 MSI 来源、SHA256 见原复现说明；本轮无需再次安装 Windows 可选功能。

```bash
apt-get update
apt-get install -y --no-install-recommends build-essential python3-venv python3-dev cmake ninja-build pkg-config libssl-dev libnuma-dev curl ca-certificates unzip patchelf
python3 -m venv /opt/tokenspeed-work/bootstrap-venv
/opt/tokenspeed-work/bootstrap-venv/bin/python -m pip install --index-url https://pypi.org/simple uv
/opt/tokenspeed-work/bootstrap-venv/bin/uv python install 3.13.16 --install-dir /opt/tokenspeed-work/python
/opt/tokenspeed-work/bootstrap-venv/bin/uv venv --seed --python /opt/tokenspeed-work/python/cpython-3.13.16-linux-aarch64-gnu/bin/python3.13 /opt/tokenspeed-work/venv
git clone https://github.com/lightseekorg/tokenspeed.git /opt/tokenspeed-work/source
git -C /opt/tokenspeed-work/source checkout --detach 864bcdd9ef3c5f1512bbe293d970d73eb29d2c94
```

本机从此前已验证来源的本地官方 clone 复制 Git 对象，随后将 origin 设置为上述官方 URL。
Rust 从 static.rust-lang.org 下载 1.92.0-aarch64-unknown-linux-gnu 归档，校验官方 SHA256，
安装到 /opt/tokenspeed-work/rust-toolchain；Cargo cache 在 /opt/tokenspeed-work/cargo。

先在 venv 中安装官方 kernel 的 cuda.txt、cuda-thirdparty.txt，以及主包 pyproject 声明的
运行依赖；三个项目原生包由本工作区源码构建。实际版本快照和安装日志见本轮报告。
独立 CUDA SDK 的安装命令为：

```bash
/opt/tokenspeed-work/bootstrap-venv/bin/uv pip install --python /opt/tokenspeed-work/venv/bin/python --target /opt/tokenspeed-work/cuda-sdk-13.4 --index-url https://pypi.org/simple --prerelease=allow "cuda-toolkit[nvcc,cccl,cudart,cublas,curand]==13.4.0rc1"
```

构建前的关键环境如下。CUDA 编译器使用独立 SDK；PyTorch 仍可能按其 ELF RPATH
加载自身的 CUDA 13.0 库，因此报告同时保存编译版本和实际加载版本。

```bash
export VIRTUAL_ENV=/opt/tokenspeed-work/venv
export CUDA_HOME=/opt/tokenspeed-work/cuda-sdk-13.4/nvidia/cu13
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$VIRTUAL_ENV/bin:/opt/tokenspeed-work/rust-toolchain/bin:$PATH"
PYTHON_BASE=$(python -c "import sys; print(sys.base_prefix)")
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$PYTHON_BASE/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="$CUDA_HOME/lib:/usr/lib/wsl/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export CARGO_HOME=/opt/tokenspeed-work/cargo
export PYO3_PYTHON="$VIRTUAL_ENV/bin/python"
export CMAKE_GENERATOR=Ninja
export CMAKE_BUILD_PARALLEL_LEVEL=2
export CARGO_BUILD_JOBS=2
export MAX_JOBS=2
export CUDA_VISIBLE_DEVICES=0
export TOKENSPEED_KERNEL_BACKEND=cuda
export TOKENSPEED_CUDA_ARCH=121
export PYTHONPATH=
```

此 SDK wheel 的库目录为 lib，部分库只有版本化名称。FlashInfer JIT 还需要常规的
lib64 和未版本化链接；本机通过以下 SDK 内符号链接解决，未修改库二进制：

```bash
ln -sfn lib "$CUDA_HOME/lib64"
for library in cudart cublas cublasLt; do
  ln -sfn "lib${library}.so.13" "$CUDA_HOME/lib/lib${library}.so"
done
ln -sfn libcurand.so.10 "$CUDA_HOME/lib/libcurand.so"
```

创建链接前应核对实际 SDK 文件；本机 cuRAND 的 SONAME 为 10，其余上述库为 13。

把校验过的 xgrammar 0.2.8 源码解包到 /opt/tokenspeed-work/vendor/xgrammar-0.2.8，
并设置 FETCHCONTENT_SOURCE_DIR_XGRAMMAR 指向它。使用独立 scheduler build-dir。

```bash
export FETCHCONTENT_SOURCE_DIR_XGRAMMAR=/opt/tokenspeed-work/vendor/xgrammar-0.2.8
cd /opt/tokenspeed-work/source
python -m pip wheel ./tokenspeed-scheduler --no-build-isolation --no-deps --wheel-dir /opt/tokenspeed-work/wheels --config-settings=build-dir=/opt/tokenspeed-work/build/scheduler
python -m pip wheel ./tokenspeed-engine --no-build-isolation --no-deps --wheel-dir /opt/tokenspeed-work/wheels
python -m pip wheel ./tokenspeed-kernel/python --no-build-isolation --no-deps --wheel-dir /opt/tokenspeed-work/wheels
python -m pip wheel ./python --no-build-isolation --no-deps --wheel-dir /opt/tokenspeed-work/wheels
# 检查目录恰好只有本次四个候选 wheel，再安装。
python -m pip install --no-deps --no-index /opt/tokenspeed-work/wheels/*.whl
```

## 验证和配对性能测试

安装后逐文件比较 wheel 与 site-packages，核对项目原生文件为 ELF64/AArch64，并保存 SHA256。
官方本地 kernel wheel 虽标记 py3-none-any，实际包含 AArch64 原生库；本次只验证
Linux ARM64 使用，不能据此作为跨平台通用 wheel 分发。
然后验证同一 BF16 检查点的三次顺序请求和一次双请求 batch，保存全部 token IDs。
性能测试复用 tools/perf/benchmark_tokenspeed.py，参数与原复现说明一致：
输入 128/512、输出 128、batch 1/4、3 次预热、10 次实测、seed=20261028、Triton attention、
eager、关闭 prefix cache、KV offload、prefill graph，context=704、KV token 容量=2816。
consistency-policy=record 保留逐轮 ID 差异；性能完成状态不代表模型输出数值等价。

按上文设置环境后，在 Linux 中运行下列命令。repo_mount 指向 Windows 目标仓库
在 WSL 中的挂载位置；模型、依赖和官方源码仍位于 Linux 文件系统。

```bash
repo_mount=/mnt/c/Users/user/projects/official-repos/windows-tokenspeed
export HF_HUB_OFFLINE=1
python "$repo_mount/tools/perf/benchmark_tokenspeed.py" \
  --platform wsl2-arm64 --attention-backend triton --consistency-policy record \
  --model /opt/tokenspeed-work/models/Qwen3-0.6B \
  --source-root /opt/tokenspeed-work/source \
  --upstream-base 864bcdd9ef3c5f1512bbe293d970d73eb29d2c94 \
  --input-lengths 128 512 --batch-sizes 1 4 --output-length 128 --warmup 3 --rounds 10 \
  --report "$repo_mount/.port/wsl2-artifacts/perf.json"
```

Windows 基线与 WSL2 基准在同一次 Windows 启动内顺序运行，避免 GPU 竞争。
旧的重启前 Windows 数据不用于本次配对比较。两侧 PyTorch、Triton、运行库及平台实现
存在差异，结果描述两套实际软件栈，不把全部差异归因于操作系统。
