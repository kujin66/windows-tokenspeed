# Windows ARM64 原生安装包复验

此为前次检查快照。插件导入与 NVFP4 的后续修复见
[最新验证报告](../windows-arm64-nvfp4-20261027/README.md)。

结论：当前 `dense-triton` 单 GPU BF16 版本能够在 Windows ARM64 原生运行。
项目五个原生二进制均为 ARM64 PE，并由 MSVC 14.51 链接。CUDA 内核的
编译记录和运行时校验确认使用真实 MSVC 19.51.36260、CUDA SDK 13.4.0。
这项结论不等于全部上游模型和后端均已移植；扩展插件测试仍有两项失败。

## 原生架构与编译证据

`IsWow64Process2` 返回进程架构 `0x0`、系统架构 `0xaa64`，说明 Python
进程直接运行于 ARM64，未经过 WOW64 仿真。Python 可执行文件的 PE 架构也
是 `0xaa64`。四个安装包中已检查的 Python 文件和原生文件均与 wheel 字节一致。

| 安装包 | 项目原生文件 | PE 架构 | 链接器版本 |
| --- | --- | --- | --- |
| tokenspeed | Python 层，`py3-none-any` | 不含原生文件 | 不适用 |
| tokenspeed-engine | Python 扩展、`ts-host.exe` | ARM64 | 14.51 |
| tokenspeed-kernel | RMSNorm、链式采样两个 CUDA 库 | ARM64 | 14.51 |
| tokenspeed-scheduler | Python 扩展 | ARM64 | 14.51 |

CUDA 库保留上游 `.so` 文件名，文件内容是 Windows PE DLL。
`native-artifacts-verification.json` 记录了各文件的架构、链接器版本、SHA256、
wheel 标签及安装内容比对结果；CUDA 库的构建身份记录中没有 14.44 辅助编译器。

本次发现旧 scheduler wheel 由 Visual Studio 生成器选择了安装目录中的
14.44 工具集。已使用 Ninja 和独立构建目录重新编译、安装。
`verify-scheduler-compiler.txt` 确认 C++ 编译器版本为 `19.51.36260.0`，
编译器和链接器均来自便携版 14.51 目录。运行脚本现在默认选择 Ninja，
构建文档也指定独立目录，避免复用旧生成器缓存。

CUDA 安装目录 `version.json` 的 SDK 版本为 **13.4.0**，其中 nvcc 组件
版本为 **13.4.46**。GPU 编译回归静态断言 `_MSC_VER == 1951` 和 ARM64
主机目标，并在 GPU 上逐个核对 4097 个向量加法结果。

## 本次测试结果

| 测试 | 结果 |
| --- | --- |
| MSVC 14.51 scheduler C++ | 661 通过 |
| 新安装 scheduler 的 Python 测试 | 280 通过 |
| 已安装 CUDA RMSNorm，FP16/BF16、PDL 开关 | 24 通过 |
| 已安装 CUDA 链式采样，与顺序参考结果比较 | 72 通过 |
| 已安装内核自动调优缓存 | 30 通过 |
| Python 宿主与扩展插件 | 51 通过、2 失败 |
| nvcc 主机探测、自动修复、GPU 向量加法 | 通过 |
| 已安装 wheel 的 Qwen3-0.6B BF16 推理 | 3 次顺序请求、2 次批量请求通过 |

合计 **1118 项测试通过，2 项扩展插件测试失败**；另外完成了架构审计、
工具链回归及五次真实推理。推理均生成 16 个 token，贪心 token ID 完全一致，
文本为 `1, 2, 3, 4, 5, 6`。配置为 eager、Triton attention、单 GPU。

两项失败均完整保存在 `verify-host.log`：

- `test_register_drafter_scopes_an_in_tree_algorithm_by_model_class`：
  Inkling/DeepSeek 导入路径需要当前环境缺少的 `flash_mla`。
- `test_mla_dcp_checks_backend_capability`：DSV4 导入路径无法解析
  `flash_mla_sparse_fwd`。

因此没有把 MLA、DSV4、相关推测解码插件纳入已验证功能范围。NVFP4 推理
此前未通过，HTTP/SMG、多 GPU、MoE、CUDA graph、卸载等也未在本次验证。
当前环境借用了相邻项目的 ARM64 PyTorch 等依赖，尚未验证全新机器上的
完整依赖安装。短推理验证证明功能与重复性，不构成压力测试或性能基准。

## 复现与产物

从仓库根目录执行，具体依赖和构建命令见
[Windows 构建指南](../../guides/windows-arm64.md)：

```powershell
$env:PYTHONPATH = ''
$py = '.venv/Scripts/python.exe'
& $py tools/windows/run.py $py tools/windows/check_native_artifacts.py
& $py tools/windows/run.py $py tools/windows/check_rmsnorm.py
& $py tools/windows/run.py $py tools/windows/check_sampling_chain.py
$env:HF_HUB_OFFLINE = '1'
& $py tools/windows/run.py $py tools/windows/validate_inference.py
```

四个 wheel 位于仓库 `dist/`，最新校验值位于 `dist/SHA256SUMS.txt`，
本目录的 `results.json` 保存当前测试结果和各 wheel 的完整审计记录。
