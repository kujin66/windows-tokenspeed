# Windows ARM64 FlashInfer 0.6.17 与 Triton / WSL2 对照

[复现流程](../../guides/windows-flashinfer-comparison.md) · [完整结果](results.json)

使用用户指定的 flashinfer_python-0.6.17+g2ea3c24b5dac-py3-none-any.whl，SHA256 为 47db3e3972fda64719ceae8490d1934362380cef62f33975cb593ad086e0bcb6。它没有预编译 GPU DLL，本机实际生成并验证 ARM64 DLL。

## 实验路径

- Windows FlashInfer：FlashInfer FA2 paged prefill + TRT-LLM decode + RMSNorm + SwiGLU。
- Windows Triton：同一个0.6.17 wheel、相同 MoE 兼容处理，attention/RMSNorm/SwiGLU 保持 Triton。
- WSL2 FlashInfer：官方源码基线、Linux0.7.2，加同一 SM12.1 FA2 paged prefill provider；decode/norm/SwiGLU 使用 FlashInfer。
- 其余 GEMM、QK norm/RoPE/KV-write prologue、采样等沿用各栈现有实现。没有声称所有 GPU 算子都替换成 FlashInfer。

Qwen3-0.6B BF16，同一块 RTX Spark N1X / SM12.1 / 驱动615.89，单 GPU。输入128/512、输出固定128、batch1/4，每 case3次预热和5次实测。context704、KV token2816、seed20261028，关闭 prefix cache、KV offload、prefill graph；decode graph capture sizes=[1,4]。每配置50个实测请求、6400个实测输出 token，预热另计。

## 吞吐结果

单位为输出 tok/s。百分比方向均为 Windows FlashInfer 相对另一列，正值表示 Windows FlashInfer 更快。

| 模式 | 输入 | Batch | Windows Triton | Windows FlashInfer | WSL2 FlashInfer | Win FI / Win Triton | Win FI / WSL2 FI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eager | 128 | 1 | 49.62 | 64.88 | 73.49 | +30.75% | -11.72% |
| eager | 128 | 4 | 177.59 | 249.92 | 271.39 | +40.73% | -7.91% |
| eager | 512 | 1 | 43.64 | 57.92 | 59.40 | +32.72% | -2.50% |
| eager | 512 | 4 | 184.65 | 256.88 | 272.37 | +39.12% | -5.68% |
| graph | 128 | 1 | 145.33 | 136.77 | 140.99 | -5.89% | -3.00% |
| graph | 128 | 4 | 600.93 | 593.53 | 544.32 | -1.23% | +9.04% |
| graph | 512 | 1 | 130.20 | 127.89 | 128.32 | -1.77% | -0.34% |
| graph | 512 | 4 | 516.95 | 494.58 | 477.95 | -4.33% | +3.48% |

## 输出一致性

| 模式 | 输入 | Batch | Win FI 重复一致 | Win Triton 重复一致 | WSL2 FI 重复一致 | Win FI/Triton 参考相同 | 跨平台 FI 参考相同 |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| eager | 128 | 1 | True | True | True | False | False |
| eager | 128 | 4 | True | True | True | False | False |
| eager | 512 | 1 | True | True | True | True | True |
| eager | 512 | 4 | True | True | True | True | True |
| graph | 128 | 1 | True | True | True | False | False |
| graph | 128 | 4 | False | False | True | False | False |
| graph | 512 | 1 | True | True | True | True | True |
| graph | 512 | 4 | True | True | True | True | True |

全部逐轮 IDs 保留在各 JSON。每请求均检查输出长度128及 completion_tokens；吞吐完成状态不代表数值等价。参考比较取第一个实测输出，重复一致性包含3次预热和5次实测。
本轮 Windows graph 的输入128/batch4 在 FlashInfer 和 Triton 两组均出现重复输出差异；部分跨后端/跨平台参考 IDs 也不同。原因尚未定位，不能把吞吐结果作为数值等价或稳定性证明。

## 遇到的问题与实际适配

1. 当前 TokenSpeed 导入新版 per-channel FP8 MoE API，0.6.17没有这些符号。两组 Windows 实验副本一致禁用 FlashInfer MoE 注册，限定 dense 模型，不伪造缺失接口。首次失败见 *moe-import-failure*。
2. TRT-LLM prefill 的 fmhaRunner.cuh 明确只接受 SM100/103/107，SM12.1失败；decode数值探测通过不能代表prefill可用。保留 *trtllm-prefill-failure*，不绕过架构检查。
3. 在 kernel registry 中新增仅 SM12.1/BF16/head_dim128/causal full-context 的 FA2 paged prefill provider。Windows与WSL2使用同一源文件；显式拒绝量化、LSE、sinks、滑窗等未实现选项。prefill graph仍关闭；规划包含 CPU 元数据同步和逐调用 wrapper 构建，属于本次适配开销。
4. Windows FA2 JIT 的长对象文件名超过路径限制，nvcc无法打开输出。将对象文件名改为源路径稳定哈希，未改CUDA计算代码、编译选项或架构判断。原始wheel完整保留。失败见 *long-path-failure*，补丁见 windows-jit-object-paths.patch。

## 验证与依赖文件

- GPU基础验证：1/4/128/512行的 RMSNorm、融合残差 RMSNorm、SwiGLU 与 FP32 参考比较，另验证 CUDA Graph；native PE machine=0xaa64。
- attention decode：16 Q heads /8 KV heads /head_dim128 /KV128，对照真实 GPU FP32 SDPA。
- FA2 prefill：128-token普通prefill、混合Q/KV长度与部分尾页/乱序页表、512-token prefill；对照显式 bottom-right causal mask 的 FP32 SDPA。Windows与WSL2分别验证。
- final-native-audit.json：wheel逐文件对比（仅一处已记录 JIT 文件名修复）、JIT DLL 的 ARM64/MSVC14.51 身份、SHA256与本地二进制备份。缓存也可能含失败路径生成的DLL；PE架构正确不代表每条路径均可用，以实际通过的测试和模型路径为准。
- 原始wheel在 .port/wheelhouse，隔离包在 .venv/flashinfer0617；本地 DLL 备份在 .port/flashinfer-comparison/native-dlls，源码/补丁和依赖版本文件纳入仓库。

## 范围与复现限制

这不是原始 wheel 在未修改 TokenSpeed 上的即装即用结果，而是有明确适配和实际数值验证的实验路径。基础 .venv、共享包与原安装 wheel 保持不变。WSL2采用官方基线的隔离 kernel 副本，新增同一FA2 provider；不把实验结果称为上游对SM12.1的完整支持。
Windows使用0.6.17，Linux使用0.7.2；PyTorch/CUDA/Triton也不同。跨平台对比仍代表两套软件栈，不是纯OS效应。5轮样本、未锁定时钟或做多轮ABBA，运行间波动需保留。计时包含生成后立即进行的长度/元数据校验，排除加载和显式预热；未测HTTP、TTFT/ITL、MoE/NVFP4、多GPU或长期压力。
scripts/ 是源快照，依赖原 .port 位置和当前本机布局，按复现说明准备；失败与修复记录分开保存。公开副本已统一UTF-8/LF，替换本机用户路径、主机名及内部域名，元数据只保留包头。evidence-sha256.json记录原始及公开副本哈希。
