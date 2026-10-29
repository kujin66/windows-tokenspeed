# Windows ARM64 构建产物、依赖与测试总表

本表汇总 MSVC 14.51 + CUDA SDK 13.4.0 的当前 TokenSpeed Windows ARM64 产物。
最新实际复验完成于 **2026-10-27 12:26（UTC+08:00）**：四个 wheel 与安装
文件一致，六个项目原生二进制均为 ARM64，186 项安装包测试及十次真实推理通过。
本次文档整理额外核对当前文件哈希与依赖；没有重新编译或重复运行上述推理。

重建及故障处理入口为
[仓库 skill](../../skills/tokenspeed-windows-arm64-build/SKILL.md)，分为
[依赖准备](../../skills/tokenspeed-windows-arm64-build/references/dependencies.md)、
[完整构建流程](../../skills/tokenspeed-windows-arm64-build/references/build.md)、
[已解决问题](../../skills/tokenspeed-windows-arm64-build/references/troubleshooting.md)
和 [测试命令](../../skills/tokenspeed-windows-arm64-build/references/validation.md)。
原 [Windows ARM64 指南](windows-arm64.md) 保留背景和已有命令。

## 源码与环境身份

| 项目 | 本轮身份 |
| --- | --- |
| 上游 | https://github.com/lightseekorg/tokenspeed |
| 基线 commit | 864bcdd9ef3c5f1512bbe293d970d73eb29d2c94 |
| Windows 源码 | 产物构建时使用本地移植修改；上传提交保留这些修改及随后格式调整，基线 SHA 不能单独代表全部修改 |
| OS / Python | Windows ARM64 / 原生 CPython 3.13.16 |
| GPU | NVIDIA RTX Spark N1X，SM12.1，GPU 0 |
| MSVC | 工具集 14.51；实际 C/C++ 编译器 19.51.36260 ARM64 |
| 编译器宏 | _MSC_VER=1951，_MSC_FULL_VER=195136260 |
| Windows SDK | 10.0.26100.0 |
| CUDA SDK / nvcc | 13.4.0（version.json）/ 13.4.46（组件版本） |
| Rust / CMake / Ninja | 1.92.0 aarch64-pc-windows-msvc / 4.4.4 / 1.13.2 |
| PyTorch / Triton | 2.14.0a0+git20b0f57（CUDA 13.4 ARM64）/ triton-windows 3.8.0.post29 |
| TVM FFI / FlashInfer | 0.1.13.post3（ARM64）/ flashinfer-python 0.7.2 |
| Transformers / compressed-tensors | 5.12.0 / 0.17.0 |

构建环境、依赖声明差异和来源说明见 skill 的依赖参考。特别是本机
setuptools=80.10.2，而主包 build-system 声明 83.0.0；--no-build-isolation 使用
实际已安装版本。本机 Python xgrammar=0.2.1，engine 静态 C++ core 为 0.2.8；
本轮推理关闭 grammar，未验证完整 Python grammar 依赖路径。

## 目录与用途

下表和机器清单用 &lt;repo&gt; 表示仓库根目录、&lt;projects&gt; 表示其父目录，
便于在用户指定的工作区或移动后的 checkout 中定位。

| 路径 | 用途 |
| --- | --- |
| &lt;repo&gt;/dist | 四个最终 wheel；SHA256SUMS.txt 是当前包哈希清单 |
| &lt;repo&gt;/.venv/Scripts/python.exe | 构建、安装与测试解释器 |
| &lt;repo&gt;/.venv/Lib/site-packages | 四个项目包的实际安装目录 |
| &lt;repo&gt;/.venv/Lib/site-packages/local-runtime.pth | 共享相邻项目的依赖 |
| &lt;projects&gt;/vllm-bench/.venv/Lib/site-packages | torch、TVM FFI 等共享 ARM64 依赖来源 |
| &lt;projects&gt;/N1X_ARM_Package_Support/wheelhouse | 本机第三方 ARM64 wheel 的准备位置，非公共下载服务 |
| &lt;repo&gt;/.port/msvc/Contents/VC/Tools/MSVC/14.51.36231 | 便携 MSVC，实际 compiler 版本以 banner 为准 |
| &lt;repo&gt;/.port/msvc/Contents/VC/Auxiliary/Build/vcvarsall.bat | nvcc 需要的官方 VS 布局脚本副本 |
| C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.4 | 已安装 CUDA SDK；链接 arm64 子目录 |
| &lt;repo&gt;/.port/rust-toolchain / .port/cargo | Rust 工具链 / Cargo cache |
| &lt;repo&gt;/.port/xgrammar-0.2.8 / .port/googletest | engine C++ grammar 源码 / scheduler 测试源码 |
| &lt;repo&gt;/.port/scheduler-wheel-msvc1451 | Ninja 的 scheduler wheel 构建目录 |
| &lt;repo&gt;/tokenspeed-engine/target/release | 默认 Cargo release 中间产物，最终以 wheel 为交付对象 |
| &lt;repo&gt;/tokenspeed-kernel/python/tokenspeed_kernel/thirdparty/cuda/objs | CUDA 各组二进制及 build-identity.json |
| &lt;repo&gt;/.port/triton / .port/flashinfer / .port/autotune | JIT 和 autotune cache，不作为项目 wheel 交付 |
| &lt;repo&gt;/.port/*.log、*.json、*.xml | 本机原始构建/测试记录，历史探测亦在此，不全部代表当前方案 |
| &lt;repo&gt;/docs/validation/windows-arm64-fp8-20261027 | 最后修复与当前独立复验的公开证据 |

路径与实际安装关系的完整清单见
[artifacts.json](../validation/windows-arm64-fp8-20261027/inventory/artifacts.json)。
不要将 .port 中的 cuda-patch、旧 14.44 库、split 构建实验或旧 kernel wheel
误作最终产物。当前方案直接使用真实 14.51 frontend 与 CUDA 13.4.46 nvcc。

## 当前 wheel

所有文件均在 &lt;repo&gt;/dist。最后三种原生包使用 cp313-cp313-win_arm64 tag；
tokenspeed 本身是 Python runtime，因此 py3-none-any 是预期的打包方式。

| 文件名 | 字节数 | SHA256 |
| --- | ---: | --- |
| tokenspeed-0.1.1-py3-none-any.whl | 2405652 | eb3801db695525d89ee737ffaf26f857db629afcde40c2ab40742efecb587f77 |
| tokenspeed_engine-0.1.0-cp313-cp313-win_arm64.whl | 5172123 | 4f74aa78ce36e354a6ffdb5850c3fb44457e1b0130710af3ef03a44ea21ffa29 |
| tokenspeed_kernel-0.1.4.dev20261027+git864bcdd9-cp313-cp313-win_arm64.whl | 3727442 | 98529ea58b82d1ab54d91987dfd2d4c8f911cae9d790a8b5904b960ca15cb7fd |
| tokenspeed_scheduler-0.1.42-cp313-cp313-win_arm64.whl | 492747 | 81f9532d3d4ae755605c2e1824a8f309f5ec463e1f7052dd36b515e737798ea4 |

安装采用 --force-reinstall --no-deps --no-index，从明确列出的本地 wheel 安装。
这避免为 Windows dense profile 拉取未移植的上游 Linux/SMG 依赖，但意味着
四个 wheel 不能单独构成已验证的全新机器完整安装包。

## 六个项目原生二进制

下列路径相对 &lt;repo&gt;/.venv/Lib/site-packages。全部 PE machine=0xaa64，linker
14.51，安装内容与 wheel 内文件逐字节相同。逐文件 SHA256、字节数、wheel
tag、CUDA flags 和安装路径一并保存在 artifacts.json。

| 安装文件 | 字节数 | 功能 |
| --- | ---: | --- |
| tokenspeed_engine/_native.cp313-win_arm64.pyd | 3169280 | Rust/Python engine 绑定 |
| tokenspeed_engine/bin/ts-host.exe | 9502208 | 原生 rank host，嵌入当前 Python |
| tokenspeed_scheduler/tokenspeed_scheduler_ext.cp313-win_arm64.pyd | 1235968 | C++ scheduler 绑定 |
| tokenspeed_kernel/thirdparty/cuda/objs/rmsnorm_fused_parallel/rmsnorm_fused_parallel.so | 829440 | CUDA RMSNorm |
| tokenspeed_kernel/thirdparty/cuda/objs/sampling_chain/sampling_chain.so | 815104 | CUDA 链式采样校验 |
| tokenspeed_kernel/thirdparty/cuda/objs/cublas_fp8_tensor/cublas_fp8_tensor.so | 66048 | CUDA cuBLASLt tensor-scaled FP8 |

.so 沿用上游命名，实际是 Windows PE DLL，不是 Linux ELF。NVFP4 矩阵乘通过
Triton tl.dot_scaled JIT 到原生 FP4 Tensor Core 指令，不对应第七个项目 AOT 库。
Python 的 IsWow64Process2 结果为 process_machine=0x0、native_machine=0xaa64，
证明测试进程原生运行；这与项目二进制 PE 检查共同支持原生 ARM64 结论。

CUDA 组编译采用 C++17、O3、/MD、/EHsc、/utf-8、/Zc:__cplusplus、
/Zc:preprocessor、--use-local-env，实际目标 compute_121a / sm_121a。
链接 CUDA arm64 的 cudart/cuda、TVM FFI；FP8 库额外链接 cuBLASLt，显式关闭
cuBLASLt fast accumulation。完整 flags、headers 和 lib 搜索路径见身份 JSON，
不要只凭文件名判断主机编译器。

## 测试结果与证据

| 本次独立复验 | 结果 | 证据 |
| --- | --- | --- |
| Python native / 四个 wheel / 六个 PE | 全通过，无 WOW64 | [架构与安装审计](../validation/windows-arm64-fp8-20261027/reconfirmation/native-artifacts-verification.json) |
| 缺 VS 布局负探测及自动修复后实编译 | 预期失败可复现，修复后编译及执行成功 | [工具链回归](../validation/windows-arm64-fp8-20261027/reconfirmation/nvcc1451-layout-regression.json) |
| 新编译 CUDA 向量加法 | 4097 元素正确，真实 1951 / runtime13040 | [GPU smoke](../validation/windows-arm64-fp8-20261027/reconfirmation/nvcc1451-cuda-smoke-result.json) |
| 安装包运行时回归 | 160 通过，0 失败/错误/跳过 | [JUnit](../validation/windows-arm64-fp8-20261027/reconfirmation/reconfirm-installed-runtime.xml) |
| 安装包 GPU / 缓存回归 | 26 通过，0 失败/错误/跳过 | [JUnit](../validation/windows-arm64-fp8-20261027/reconfirmation/reconfirm-installed-gpu.xml) |
| BF16 / NVFP4 真实推理 | 各五次通过，各请求 16 token，基线 token ID 不变 | 下节两个报告 |

186 项不同安装包测试由 FP8 80、NVFP4 25、插件 34、dummy drafter 2、host 19
和物理 GPU/cache 26 组成；不将源码与安装包重复执行、旧组件测试或 smoke
重复累计。原 46 项 FP8/MLA 失败均有对应通过 ID，见
[修复结果](../validation/windows-arm64-fp8-20261027/results.json)。

历史单组件 14.51 验证还包括 scheduler C++ 661、scheduler Python 280、原生
RMSNorm 24 和 chain 72 项通过。它们保留在
[此前组件验证](../validation/windows-arm64-native-20261027/README.md)，不代表
最新 wheel 的所有功能又重新跑了一遍。旧报告的失败数和哈希属于当时状态；
当前产物以本表及最新复验为准。

## 模型与实际推理

| 项目 | BF16 | NVFP4 |
| --- | --- | --- |
| 检查点 | Qwen/Qwen3-0.6B | mgoin/Qwen3-0.6B-NVFP4 |
| 本机路径 | &lt;repo&gt;/.port/models/Qwen3-0.6B | &lt;projects&gt;/vllm-bench/models/Qwen3-0.6B-NVFP4 |
| revision | c1899de289a04d12100db370d81485cdf75e47ca | 545def2dd2492f4abd75435ccde6dd93d1d93e34 |
| model.safetensors SHA256 | f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b | a2e29806fc05893751b60ff4144a48f966e6c1d9a4a09ff9c5ea73dbccf80cc7 |
| 权重格式 | BF16 | nvfp4-pack-quantized，16 元素分组 |
| 请求 | 三次顺序、两次批量请求 | 三次顺序、两次批量请求 |
| 输出例子 | 1, 2, 3, 4, 5, 6 | 12345，随后解释下一个数字 |
| 完整响应/token ID | [BF16 JSON](../validation/windows-arm64-fp8-20261027/reconfirmation/reconfirm-bf16-inference.json) | [NVFP4 JSON](../validation/windows-arm64-fp8-20261027/reconfirmation/reconfirm-nvfp4-inference.json) |

脚本为 tools/windows/validate_inference.py，采用单 GPU、eager、Triton attention、
greedy，context512、max_num_seqs=2、max_total_tokens=512、显存比例0.25。
关闭 grammar、prefill graph、KV offload、retraction snapshots。每模型五次
请求各 16 completion token；同一检查点输出稳定且与前次基线完全一致，不
要求 BF16 和 NVFP4 两个不同检查点之间输出相同。

## 可机器读取的清单

| 文件 | 内容 |
| --- | --- |
| [最新复验 results.json](../validation/windows-arm64-fp8-20261027/reconfirmation/results.json) | 测试统计、时间、工具链、产物及证据 SHA256 |
| [inventory/artifacts.json](../validation/windows-arm64-fp8-20261027/inventory/artifacts.json) | 四个 wheel 和六个原生文件的路径、字节数、哈希及编译身份 |
| [inventory/python-environment.json](../validation/windows-arm64-fp8-20261027/inventory/python-environment.json) | 219 个实际解析到的 distribution 的版本、位置、tag、Requires-Dist |
| [inventory/python-environment.txt](../validation/windows-arm64-fp8-20261027/inventory/python-environment.txt) | 便于阅读的观测版本快照，不是安装 lock |
| [inventory/download-cache-audit.json](../validation/windows-arm64-fp8-20261027/inventory/download-cache-audit.json) | 八个归档的公开来源、预期/实际 SHA256，本轮全部匹配 |
| [inventory/source-inputs.json](../validation/windows-arm64-fp8-20261027/inventory/source-inputs.json) | 关键构建配置、Cargo.lock 和验证脚本的 SHA256 |
| [格式化前源码快照](../validation/windows-arm64-fp8-20261027/inventory/source-inputs-before-formatting.json) | 保留提交检查之前的源码身份，与上传时的格式化后快照区分 |

上传前按 AGENTS.md 执行完整 pre-commit 检查，规范了源码格式及证据副本的
空白/文件结尾。证据 SHA256 已按公开副本更新；测试执行时间和原 wheel 字节
保持原记录，并未将格式检查写成重新编译或重新推理。固定 clang-format18.1.8
使用独立 x64 Python3.12.15 hooks 环境，项目构建及推理仍为原生 ARM64。

## 结论边界

已证明本机原生 Windows ARM64、RTX Spark N1X SM12.1、单 GPU、eager
BF16/NVFP4 在所列工具链与共享依赖环境下通过验证。SM100/103 回归使用模拟
平台元数据，并非对应物理硬件测试。尚未证明全新机器独立安装全部依赖、
完整 MLA/MoE 服务、HTTP/gRPC/SMG、多 GPU/NCCL、CUDA graph、卸载、压力
和性能指标。四个项目 wheel 的 14.51 结论不意味着 CPython 及全部第三方包
也由本仓库用 14.51 重编译。
