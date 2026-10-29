# Windows ARM64 FlashInfer 0.6.17 对照流程

本轮使用用户提供的 flashinfer_python-0.6.17+g2ea3c24b5dac-py3-none-any.whl，
SHA256 为 47db3e3972fda64719ceae8490d1934362380cef62f33975cb593ad086e0bcb6。
它包含 Windows JIT 支持，没有预编译 GPU DLL；本机通过 MSVC 14.51、CUDA 13.4.46
生成 ARM64 DLL。原始 wheel 备份在仓库 .port/wheelhouse，包安装在
.venv/flashinfer0617，JIT 短路径为 projects/fij0617，以避免 Windows 路径长度限制。

[当前 non-eager 测试结果与方法](windows-wsl2-engine-backend-retest.md)
记录具体环境、计时边界、逐轮数据和输出一致性。
[原 Windows 构建说明](windows-arm64-reproduction.md) 和
[WSL2 构建说明](wsl2-arm64-build.md) 仍作为基础环境前提。

## 为什么需要实验副本

原移植会在 Windows 强制选择 Triton RMSNorm/SwiGLU。只安装 FlashInfer wheel
不会改变这两处选择。本次复制已安装 tokenspeed_kernel 到两个隔离目录：

- .venv/flashinfer0617-kernel-triton：Triton 对照。
- .venv/flashinfer0617-kernel：FlashInfer 对照，移除两个 facade 的 Windows 强制分支。

FlashInfer 0.6.17 缺少新版 per-channel FP8 MoE API，TokenSpeed 的通用模型 executor
却会在 dense 模型启动时导入 MoE 模块。两组实验副本
均显式禁用 FlashInfer MoE 注册，保留其不可用行为；没有用其他操作冒充缺失的接口。
本流程验证 dense Qwen3-0.6B，不声明 MoE、NVFP4 或所有 FlashInfer 功能可用。

两个实验副本之间仅有 RMSNorm/SwiGLU facade 差异；原生 TokenSpeed 二进制逐字节
保持一致。FlashInfer attention 通过明确的 attention_backend=flashinfer 选择。
这个 wheel 的 TRT-LLM prefill 限制在 SM100/103/107，本机 SM12.1 会报 Unsupported architecture。
本轮通过 kernel registry 增加 SM12.1 的 FlashInfer FA2 paged prefill provider；decode
继续使用通过数值验证的 FlashInfer TRT-LLM 路径。适配器只支持本次 BF16、head_dim=128、
causal full-context 组合，并明确拒绝量化、LSE、sinks、滑窗等未实现选项。
FA2 prefill 规划会把长度/页表元数据同步到 CPU；性能包含这部分适配开销。
GEMM、QK norm/RoPE/KV-write prologue、采样等其余路径保持项目自身的选择。

Windows JIT 的长对象文件名还会超过路径长度限制。本轮仅将对象文件名改成稳定路径哈希，
不改变 CUDA 计算代码、编译选项或架构检查。原始 wheel 保持不变，差异补丁随报告保存。

## 在此工作区复现

先查看engine measurement sources/ 中的源快照；以下命令使用本轮保留在仓库 .port 的脚本。
若从 Git checkout 恢复，将相应快照按原名称复制回 .port，并先准备基础 venv。
这些命令是当前共享依赖布局的复现步骤，不是干净机器的完整依赖安装配方。

```powershell
& ./.venv/Scripts/python.exe -m pip install --no-deps --no-index --target .venv/flashinfer0617 .port/wheelhouse/flashinfer_python-0.6.17+g2ea3c24b5dac-py3-none-any.whl
& ./.venv/Scripts/python.exe .port/setup_flashinfer_overlay.py
& ./.venv/Scripts/python.exe .port/prepare_flashinfer_dense_compat.py
& ./.venv/Scripts/python.exe .port/install_fa2_adapter.py
& ./.venv/Scripts/python.exe .port/fix_flashinfer_object_paths.py
```

准备脚本创建新目录，适用于首次准备；已有实验副本时先检查状态，避免覆盖本轮证据。
当前准备脚本位于 [tools/windows/flashinfer](../../tools/windows/flashinfer/)，
源码还原工具会将它们复制到 .port。这些脚本不依赖任何旧测试文件。
复测已有环境时直接运行基准，不重复覆盖准备目录。

launcher 设置独立 PYTHONPATH、FlashInfer JIT/autotune 缓存和额外 linker 搜索目录。
后者指向本机共享 venv 的 torch/lib、tvm_ffi/lib。没有修改共享包或系统配置。


## Non-eager measurements

Use tools/perf/benchmark_engine_matrix.py with --engine tokenspeed, explicit
--backend triton or flashinfer, --execution graph, input lengths 128/512,
batches 1/4, output length 128, three warmups and five measured calls.
See the [complete comparison](windows-wsl2-engine-backend-retest.md) for all
arguments, repeat aggregation, timing boundaries and package differences.
