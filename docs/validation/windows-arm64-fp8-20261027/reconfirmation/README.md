# Windows ARM64 安装产物独立复验

完成时间：2026-10-27 12:26（UTC+08:00）。

对当前四个 wheel 及其已安装文件重新检查，并重新执行编译探测、GPU 测试和
实际模型推理。本轮未重新构建内容未变的 wheel；构建身份记录、PE 头及文件
SHA256 一并核验。完成时间及证据摘要见 [results.json](results.json)。

| 检查 | 本轮结果 |
| --- | --- |
| Python 进程 | 原生 ARM64，IsWow64Process2 为 0x0 / 0xaa64，无仿真 |
| 四个 wheel 与安装文件 | 逐文件一致，wheel SHA256 与原产物一致 |
| 六个项目原生二进制 | Windows ARM64 PE，machine 0xaa64，链接器 14.51 |
| CUDA SDK / nvcc 组件 | 13.4.0 / 13.4.46 |
| 实际 CUDA 主机编译器 | MSVC 14.51，19.51.36260，_MSC_VER=1951 |
| nvcc 主机探测修复回归 | 缺少 VS 布局时复现错误，自动恢复官方脚本后直接编译通过 |
| 新编译 CUDA 向量加法 | 4097 个元素全部正确，runtime 版本 13040 |
| 已安装包运行时回归 | 160 通过，0 失败，0 跳过 |
| 已安装包 GPU/缓存回归 | 26 通过，0 失败，0 跳过 |
| Qwen3-0.6B NVFP4 | 3 次顺序请求、2 次批量请求，各生成 16 token |
| Qwen3-0.6B BF16 | 3 次顺序请求、2 次批量请求，各生成 16 token |

两个模型各自的全部请求与前次基线保持相同 greedy token ID。两个检查点
不同，不要求 BF16 和 NVFP4 之间的输出相同。共计 186 项不同安装包测试
和 10 次实际推理；编译烟雾测试单独记录，不重复计数。

证据文件：

- [原生架构与安装文件审计](native-artifacts-verification.json)
- [nvcc 布局修复与实编译结果](nvcc1451-layout-regression.json)
- [运行时 JUnit](reconfirm-installed-runtime.xml) / [日志](reconfirm-installed-runtime.log)
- [GPU JUnit](reconfirm-installed-gpu.xml) / [日志](reconfirm-installed-gpu.log)
- [NVFP4 推理](reconfirm-nvfp4-inference.json) / [日志](reconfirm-nvfp4-inference.log)
- [BF16 推理](reconfirm-bf16-inference.json) / [日志](reconfirm-bf16-inference.log)

物理硬件验证范围为 RTX Spark N1X（SM12.1）、单 GPU、eager BF16/NVFP4。
SM100/103 回归使用模拟平台元数据，并未在对应物理硬件执行。完整 MLA/MoE
服务、HTTP/SMG、多 GPU、CUDA graph、卸载、压力及性能测试不在本轮结论内。
当前项目 venv 通过 local-runtime.pth 共享相邻项目的 ARM64 依赖，因此尚未
证明全新机器上能独立完成全部依赖安装。

复跑命令和构建方式见 [Windows ARM64 指南](../../../guides/windows-arm64.md)。
