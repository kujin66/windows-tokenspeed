# Windows ARM64：剩余 46 项 FP8/MLA 失败修复

本轮独立复验记录见 [原生架构、工具链与实际推理复验](reconfirmation/README.md)。
产物路径、完整依赖快照与可复用 skill 入口见
[构建产物与测试总表](../../guides/windows-arm64-build-record.md)。

**前次 46 项失败已全部清零。** 最终安装包的 160 项运行时回归和
26 项 GPU/缓存测试全部通过，失败和跳过均为零。原失败用例的每个 ID
均在本次安装包 JUnit 结果中找到对应的通过记录，详见 `results.json`。

环境为 Windows ARM64、RTX Spark N1X（SM12.1）、MSVC 14.51
（编译器 19.51.36260）、CUDA SDK 13.4.0（nvcc 组件 13.4.46）。

## 修复

1. 为 SM100 系列补齐不限行数的可移植 FP8 token 量化注册。缺少 TRT-LLM
   时仍能选择已有 FP8 GEMM；存在 vendor 量化时保留其较高优先级，
   小 batch 继续使用原有高优先级量化实现。
2. 新增直接链接 CUDA SDK cuBLASLt 的原生 FP8 库，支持 E4M3、每个输入
   一个 FP32 device scale、FP16/BF16 输出。无需 FlashInfer Linux JIT。
   显式关闭 fast accumulation，隔离不同设备和 CUDA stream 的 scratch，
   验证矩阵布局、尺寸、dtype 和设备，并正确处理带 padding 的输出行。
3. CuTe 专用融合、TRT-LLM 和 DeepGEMM 的导入及注册检查实际依赖是否
   存在。普通 DeepSeek/MLA 类导入和配置校验不再被未使用的依赖阻断；
   强制调用不可用操作时明确报错，不注册虚假的可执行后端。
4. 两个大 batch 量化测试原先强制要求 TRT-LLM。现在按实际已注册的
   vendor 能力检查应走的量化路径，同时保留数值参考比较。没有删除
   原失败用例，也没有将其标记为 skip。

## 验证

| 检查 | 结果 |
| --- | --- |
| 源代码运行时回归 | 160 通过、0 失败、0 跳过 |
| 已安装 wheel 运行时回归 | 160 通过、0 失败、0 跳过 |
| 原生 cuBLASLt FP8 GPU 数值、out/bias、多 stream | 21 通过 |
| 大 batch token 量化，含零值、极小值、IEEE 舍入参考 | 4 通过 |
| NVFP4 batch 变化不重新编译 | 1 通过 |
| 四个 wheel 与实际安装文件比对 | 通过 |
| 六个项目原生 PE 二进制 | 全部 ARM64，链接器版本 14.51 |
| Python 进程 | ARM64 原生执行，无 WOW64 仿真 |
| NVFP4 检查点 | 3 次顺序请求、2 次批量请求通过 |
| BF16 检查点 | 3 次顺序请求、2 次批量请求通过 |

GPU 和缓存测试也对安装包执行，禁用了会把源代码目录插入路径的测试
conftest。每个模型的五次请求都生成 16 个 token，并且 token ID 与修复前
相同。源代码与安装包测试是两次验证，不将它们重复累计为不同用例；
本次安装包共 **186 项测试通过，另有十次真实推理通过**。

编译日志为 `cublas-fp8-build.log`，安装包架构审计为
`native-artifacts-verification.json`。内核包现在包含 RMSNorm、链式采样和
cuBLASLt FP8 三个 ARM64 PE CUDA 库，保留上游 `.so` 命名方式。

## 产物与范围

更新后的 kernel wheel 已强制安装到项目 `.venv`，四个 wheel 位于
仓库 `dist/`，最新 SHA256 位于 `dist/SHA256SUMS.txt` 和本目录
`results.json`。其他三个 wheel 内容保持原样，当前安装内容均通过审计。

本次物理 GPU 验证目标为 SM12.1，单 GPU、eager BF16/NVFP4。
SM100/103 回归采用模拟平台元数据，并非在对应物理 GPU 上测得结果。
模型类导入和配置检查通过，也不等于已验证完整 DeepSeek/MLA/MoE 服务。
未新增 HTTP/SMG、多 GPU、CUDA graph 或卸载功能的验证结论。

构建、安装及模型复验命令见 [Windows ARM64 指南](../../guides/windows-arm64.md)。
核心回归可从仓库根目录重跑：

```powershell
$env:PYTHONPATH = ''
$py = '.venv/Scripts/python.exe'
& $py tools/windows/run.py $py -m pytest test/runtime/test_compressed_tensors_fp8.py test/runtime/test_compressed_tensors_nvfp4.py test/runtime/test_runtime_plugins.py test/runtime/execution/test_dummy_drafter.py test/runtime/test_rust_host.py -q
& $py tools/windows/run.py $py tools/windows/check_native_artifacts.py
```
