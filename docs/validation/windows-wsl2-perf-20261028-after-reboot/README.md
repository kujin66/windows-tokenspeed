# Windows ARM64 与 WSL2 配对性能测试（重启后）

WSL2/Ubuntu 安装、官方源码构建、原生 ELF 审计和 BF16 真实推理已完成。
Windows 与 WSL2 在同一次 Windows 启动内顺序测量；重启前的结果保留为历史记录，没有混入本轮。

[构建与复现](../../guides/wsl2-arm64-build.md) · [完整结果](results.json) · [逐 case 比较](comparison.json)

## 配置

同一块 RTX Spark N1X / SM12.1 / 驱动 615.89；Qwen3-0.6B BF16，Triton attention，greedy，eager，单 GPU。
输入 128/512、输出固定 128 token、batch 1/4；每 case 3 次预热和 10 次实测。
关闭 prefix caching、KV offload、prefill graph；context=704、KV token 容量=2816、seed=20261028。
每侧 100 个实测请求、12,800 个实测输出 token；预热另计。测量完整阻塞 Engine.generate，包含 prefill/decode/IPC。

## 吞吐结果

| 输入 token | Batch | Windows 输出 tok/s | WSL2 输出 tok/s | WSL2 / Windows | WSL2 相对 Windows |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 53.19 | 65.01 | 1.222x | +22.22% |
| 128 | 4 | 215.64 | 244.94 | 1.136x | +13.59% |
| 512 | 1 | 55.00 | 64.06 | 1.165x | +16.49% |
| 512 | 4 | 215.88 | 244.66 | 1.133x | +13.33% |

## 输出一致性

每个请求的输出长度均严格校验；全部逐轮 token IDs 在两侧 perf.json 中。
consistency-policy=record 保留重复请求和跨平台差异；下表的“跨平台”比较每个 case 的参考输出。

| 输入 token | Batch | Windows 重复一致 | WSL2 重复一致 | 跨平台参考 IDs 相同 |
| ---: | ---: | --- | --- | --- |
| 128 | 1 | True | True | True |
| 128 | 4 | True | True | False |
| 512 | 1 | True | True | True |
| 512 | 4 | True | True | False |

## 构建和验证

Linux 使用官方 864bcdd9 基线，Windows 为在此基线上移植的 9a50ac82。
Linux 四个 wheel 与安装文件逐字节匹配，27 个项目 ELF 均为 AArch64；记录见 wsl2/native-artifacts.json。
Linux 编译器为 GCC 15.2 / Rust 1.92 / nvcc 13.4.46。PyTorch 编译 CUDA 为 13.0，实际加载的运行库也为 13.0；不能把 nvcc 版本当作 PyTorch 运行库版本。
另外通过 4097 元素 CUDA 向量加法、BF16 矩阵乘，以及模型 3 次顺序加 2 次 batch 请求，后者输出 ID 一致。
Windows 复用已审计的四个 wheel，本轮重新执行性能测试；此前 202 项 pytest 结果不重复累计。

## 依赖和适用范围

CUDA 13.0 编译器与 Ubuntu 26.04 头文件冲突，改用独立官方 CUDA 13.4.0rc1 SDK；随后补齐 cuRAND 头文件及常规 lib64/未版本化库链接。失败与修复日志均保留。
供应商 cuSPARSELt wheel 的 sbsa 内部标签使 pip check 报平台不支持；实际 ELF 为 AArch64。本次没有改写该包元数据。
官方本地 kernel wheel 标为 py3-none-any，但包含 AArch64 原生库；仅验证本机 Linux ARM64 使用，不能据此当作跨平台通用 wheel。
两侧 PyTorch、Triton、运行库及平台实现有差异，结果是两套实际软件栈的定长吞吐对比，不等同于单独测量操作系统开销。
未测 HTTP 服务、TTFT/ITL、CUDA graph、多 GPU、NVFP4 或长时间压力。

## 文件

- windows/perf.json 和 wsl2/perf.json：完整逐轮数据、环境版本、模型及脚本哈希。
- wsl2/native-artifacts.json：安装字节比较、ELF 架构、wheel/文件 SHA256 与编译身份。
- wsl2/runtime-toolchain.json、wsl2/cuda-sdk-packages.json：实际加载库与独立 SDK。
- setup/ 与 wsl2/logs/：安装、失败、修复和源码构建日志。
- evidence-sha256.json：原始及脱敏副本 SHA256。

原始日志和 Linux wheel 保存在仓库 .port/wsl2-artifacts，Windows wheel 在 dist。
公开副本已去掉终端颜色、替换本机用户路径与主机名，并统一 UTF-8/LF；旧证据没有覆盖。
