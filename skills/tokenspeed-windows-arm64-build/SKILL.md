---
name: tokenspeed-windows-arm64-build
description: "在本仓库使用 MSVC 14.51 和 CUDA SDK 13.4.0 构建、排查、安装及验证 Windows ARM64 TokenSpeed wheel，包含依赖准备、nvcc 主机探测、原生架构审计和 BF16/NVFP4 推理。"
---

# TokenSpeed Windows ARM64 构建与验证

源码还原与依赖准备见 [构建说明](../../docs/guides/windows-arm64-build-details.md)。
当前性能结果见 [non-eager 对比](../../docs/guides/windows-wsl2-engine-backend-retest.md)，
Linux 环境准备见 [WSL2 构建](../../docs/guides/wsl2-arm64-build.md)。

在包含 Windows 移植修改的 TokenSpeed checkout 中工作。仓库根目录为本文件的
两级父目录；命令中的路径相对仓库根目录。遵守该 checkout 的 AGENTS.md，
优先使用项目 .venv 的解释器。不要把此流程直接套到未移植的上游源码。

## 按任务读取

- 准备工具链、恢复环境或核对依赖：读 [依赖与来源](references/dependencies.md)。
- 构建、安装 wheel 或更换编译器：读 [构建流程](references/build.md)。
- nvcc 拒绝主机、DLL/导入或版本选择失败：读 [故障处理](references/troubleshooting.md)。
- 判断是否原生、复跑 GPU/模型测试或提交验证结论：读 [测试与证据](references/validation.md)。
- 查看当前产物、哈希、安装路径、测试结果：读
  [产物与验证总表](../../docs/guides/windows-arm64-build-details.md)。

这些参考文件属于仓库 skill，引用同一仓库中的源码和证据；移动或安装本 skill
时需保留仓库，或调整引用。用户要求的输出位置为仓库 skills，不自动安装到
用户全局技能目录，也不修改全局技能配置。

## 执行要点

1. 先识别任务是复验现有产物还是重编译。复验未变的 wheel 无需重建；代码或
   工具链改变时重建受影响包，再验证安装内容。四个包均放在 dist 中。
2. 确认 Windows ARM64 Python、GPU 0 的 SM12.1、MSVC 14.51 和 CUDA SDK
   13.4.0。目录名 14.51.36231 与实际编译器 19.51.36260 不同；以 cl 输出、
   编译身份和运行时宏为证。SDK 13.4.0 的 nvcc 组件版本为 13.4.46。
3. 通过 tools/windows/run.py 准备环境。nvcc 依赖便携 VS 布局中的官方
   VC/Auxiliary/Build/vcvarsall.bat；runner 会从本机 VS 复制缺失文件。
   保持真实 19.51 前端与主机编译，不伪造宏或修改 NVIDIA 二进制。
4. 使用 Ninja；清除显式继承的 Visual Studio generator，使用独立构建目录。
   仅改变 PATH 不能证明 CMake 没有选中已安装的 14.44。检查 CMake 编译器
   路径和 wheel 内 PE 的 ARM64 machine / linker 版本。
5. 本机已有依赖通过 local-runtime.pth 共享。--no-build-isolation 与 --no-deps
   是已准备环境的构建策略，不能当作公共 PyPI 上的完整安装支持。
6. CUDA profile 明确选 dense-triton，目标 121，frontend helper 为空。
   三个原生 CUDA 库加 Triton 内核覆盖本机 dense 推理；不得将可选依赖导入
   成功解释为完整 MLA/MoE、多 GPU 或服务后端已移植。
7. 编译主机探测回归会短暂移动仓库内的 vcvarsall.bat，期间不并行启动
   使用同一工具链的 nvcc 构建。真实模型顺序运行，避免重复启动占用内存。
8. 安装包验证清空源码 PYTHONPATH；kernel GPU 测试禁用会插入源码的
   conftest。同时验证数值、原生 PE 和逐文件安装一致性。
9. 保存新的时间、日志、JUnit、推理 JSON、SHA256 和验证范围到 docs/validation。
   不覆盖历史证据；不把过往测试与本轮同名用例重复累计。公开日志替换本机
   用户路径、主机名及私有索引地址，处理 UTF-16 BOM 后再写成 UTF-8。

## 完成标准

报告实际执行的构建/复验、wheel 和六个原生二进制的位置与 SHA256、工具链
身份、测试通过/失败/跳过数，以及模型请求数和 token 一致性。以本次生成的
证据为准，不累计独立测试轮次的通过数。

当前模型性能范围是单 GPU dense BF16、decode CUDA Graph，具体后端、版本和局限以当前报告为准。
