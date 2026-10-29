# Windows ARM64：插件导入与 NVFP4 修复验证

此为前次检查快照。文中剩余 46 项失败的后续修复见
[FP8 最终验证报告](../windows-arm64-fp8-20261027/README.md)。

前次两项 FlashMLA/DSV4 插件导入错误已修复；Qwen3-0.6B 的 BF16 和
compressed-tensors NVFP4 检查点均已通过重建、安装后的 wheel 推理验证。
本机为原生 Windows ARM64，RTX Spark N1X，SM12.1，MSVC 14.51、
CUDA SDK 13.4.0（nvcc 组件 13.4.46）。

## 修复内容

- FlashMLA 缺失时，公共入口可导入，调用不可用入口时明确失败；
  DSV4 的 FlashMLA CUDA 实现仅在依赖存在时注册。
- 普通 MTP 模型解析不再提前导入 Inkling 及其 CuTe 专用内核。
- 增加 `nvfp4-pack-quantized` 密集线性层方案，接受对称 4-bit 浮点、
  group size 16、动态局部激活。权重保持 E2M1 打包格式，块缩放保持 E4M3。
- 融合的 QKV、gate/up 投影分别使用各自的激活和权重全局缩放值，
  正确处理检查点存储的倒数语义，不以共享最大值覆盖不同缩放。
- SM120/SM121 使用统一内核注册与调度入口 `gemm.nvfp4_linear`，
  在线量化激活并通过原生 FP4 Tensor Core 指令计算。批大小是运行时参数。

## 已安装 wheel 的验证

| 检查 | 结果 |
| --- | --- |
| 四个 wheel 与实际安装文件比对 | 通过 |
| 五个项目 PE 二进制 | 全部 ARM64、链接器版本 14.51 |
| Python 进程架构 | ARM64，未使用 WOW64 仿真 |
| NVFP4 配置、边界值、零块、舍入、矩阵乘、融合缩放 | 25 项通过 |
| 运行时插件（含前次两项失败） | 34 项通过 |
| Dummy drafter | 2 项通过 |
| Python/Rust 宿主接口 | 19 项通过 |
| 变化批大小时不重新编译 GPU 程序 | 1 项通过 |
| NVFP4 模型推理 | 3 次顺序请求、2 次批量请求通过 |
| BF16 模型推理回归 | 3 次顺序请求、2 次批量请求通过 |

核心已安装包回归为 **80 项通过、0 项失败**，另有一项批大小编译缓存验证。
每个模型的五次请求都生成 16 个 token，同一模型内的贪心 token ID 一致。

NVFP4 GPU 计算逐项对比独立 E2M1 解码及动态激活量化参考，覆盖 FP16/BF16、
不同 batch、K/N 尾部、不同融合投影全局缩放和 bias。
`native-nvfp4-mm.ptx` 来自实际编译缓存，包含：

```ptx
mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3
```

这条计算路径没有把整模型权重展开成 BF16，也不依赖 FlashInfer 的 Linux JIT。
BF16 与 NVFP4 是不同检查点，测试分别验证重复性，不要求两者文本相同。
当前 NVFP4 短提示的输出为 `12345`，后续为 `The number after 12345 is`。
这是功能验证，不是模型质量或性能基准。

## 扩展测试中仍存在的问题

另外运行的 FP8/MLA 扩展组合为 **95 项通过、46 项失败**，完整结果保留在
`remaining-fixes-regression.log`，没有将这些失败计入通过结果：

- 36 项依赖当前 SM12.1 Windows 配置中未提供的 SM100/103 FP8 专用后端注册。
- 10 项进入 DeepSeek/MLA 模型导入路径，需要当前缺少的 CuTe `cutlass` 包。

本次未实现 FlashMLA 本体、DeepSeek/MoE、多 GPU、HTTP/SMG、CUDA graph 或
卸载功能，也没有验证全新机器上的完整依赖安装。原生执行、单 GPU NVFP4
与 BF16 的验证结论不能扩大为全部上游模型和后端均已支持。

## 产物与复现

新 runtime 和 kernel wheel 已写入仓库 `dist/` 并强制重装到项目 `.venv`。
engine、scheduler 的原生 14.51 构建保持原有内容。当前四个 wheel 的 SHA256
位于 `dist/SHA256SUMS.txt`，详细结果见本目录 `results.json`。

NVFP4 模型：`mgoin/Qwen3-0.6B-NVFP4`，revision
`545def2dd2492f4abd75435ccde6dd93d1d93e34`。检查点 SHA256 也记录在结果中。

[构建与复现命令](../../guides/windows-arm64.md)。从仓库根目录执行：

```powershell
$env:PYTHONPATH = ''
$env:HF_HUB_OFFLINE = '1'
$py = '.venv/Scripts/python.exe'
& $py tools/windows/run.py $py tools/windows/check_native_artifacts.py
& $py tools/windows/run.py $py tools/windows/validate_inference.py --model ../vllm-bench/models/Qwen3-0.6B-NVFP4 --report .port/nvfp4-inference-validation.json
& $py tools/windows/run.py $py tools/windows/validate_inference.py
```
