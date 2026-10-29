# 已解决问题与排查依据

| 现象 | 已确认原因 | 处理与验证 |
| --- | --- | --- |
| nvcc: Host compiler targets unsupported OS | raw MSVC VSIX 缺少 VS 布局标记，并非真实 19.51 不支持 | runner 从已安装 VS 复制官方 VC/Auxiliary/Build/vcvarsall.bat；check_cuda_toolchain.py 负/正探测及 GPU 测试 |
| cl 已是 14.51，但 scheduler wheel 链接器仍为 14.44 | CMake Visual Studio generator 选了已安装 toolset | Ninja + 新 build-dir；检查 CMakeCXXCompiler.cmake 和最终 PE linker 字段 |
| 普通 VS Code 终端 nvcc 失败 | 没有准备 ARM64 INCLUDE/LIB/PATH 或 VS 布局 | 用 tools/windows/nvcc.cmd，或 Activate-Toolchain.ps1；check_toolchain_activation.ps1 验证 |
| PowerShell 禁止执行 .ps1 | 本机执行策略 | nvcc.cmd 无需执行 .ps1；交互激活可用新进程的 -ExecutionPolicy Bypass |
| DLL load failed / 找不到 tvm_ffi.lib | x64/ARM64 混用或 DLL/import library 搜索路径缺失 | 原生 ARM64 Python 与依赖；runner 添加 base_prefix，核对 CUDA lib/arm64 与 TVM FFI 的 ARM64 lib |
| scheduler 依赖 OpenSSL DLL / C++ 类型转换失败 | 上游 POSIX/OpenSSL 假设或 MSVC 更严格转换 | Windows 使用 bcrypt/CNG；保留 framing；已修复 iterator 转换、includes、diagnostic 类型与扩展安装路径 |
| engine 构建找不到 scheduler/xgrammar | Windows checkout 的 symlink 可能为文本，或 wheel 无 C++ 头文件 | 保留本仓库 build.rs 修复；预解压 xgrammar 0.2.8，设置 FETCHCONTENT_SOURCE_DIR_XGRAMMAR |
| Rust host 启动/退出、指标文件在 Windows 失败 | POSIX FD、signal、文件 API 假设 | 原生 named-pipe 继承、进程 HANDLE 监控、console signal 与 Windows file API；跑 test_rust_host.py |
| 单 GPU 启动找不到 NCCL | Windows PyTorch 没有 NCCL | 此移植仅 world_size=1 以 Gloo 启动；多 GPU 显式拒绝，不绕过该限制 |
| 模型未使用的 Inkling/FlashMLA/CuTe/DeepGEMM 仍在导入时报错 | 过早导入全部模型与 vendor kernels | 按模型延迟导入，按真实依赖注册；强制调用不可用操作明确报错，不放置虚假后端 |
| 46 项 FP8/MLA 配置/dispatch 回归失败 | 缺大 batch portable FP8 注册及 Windows tensor FP8 GEMM、可选依赖被强制导入 | 增加 full-row token quantizer 与原生 cuBLASLt FP8，修正可用性判断；原 46 个 ID 均通过 |
| FlashInfer Linux JIT 编译 FP8 失败 | 该第三方编译路径未移植 | 用 SDK cuBLASLt + TVM FFI 实现 E4M3 tensor scale FP8；测试数值、bias、out stride、多 stream |
| autotune 临时文件不能重新打开 | Windows 下打开文件不允许其他库按原方式重开 | 写后先关闭，再交给 FlashInfer；验证持久化读写 |
| 用例通过但没有验证 wheel | PYTHONPATH 或 kernel conftest 把源码插入路径 | runtime 清空 PYTHONPATH；kernel 指向 test helpers 并用 --noconftest --import-mode=importlib |
| 精确 FP8 量化参考偶发差一个 ULP | FP32 GPU 除法可能使用 reciprocal 近似，或从舍入后 scale 逆推 multiplier | 独立 448/amax，参考先 FP64 再舍入 FP32；保留数值断言，不放宽到掩盖问题 |

## 不能据此扩张的结论

修复可选导入只证明类导入/配置路径；不代表完整 DeepSeek/MLA/MoE 服务。
SM100/103 的模拟平台元数据测试不证明有对应物理硬件测试。主包纯 Python
wheel 的 py3-none-any tag 合理，原生能力来自其他三个 win_arm64 wheel。
第三方 Python/依赖本身不要求全部由本项目 MSVC 14.51 编译；本轮 14.51
结论针对项目的六个原生二进制和实际 CUDA host 编译。

## 耗时、日志与缓存

冷启动常需 2–3 分钟，validate_inference.py 的 faulthandler 每 120 秒打印堆栈。
Timeout (0:02:00)! 快照不是自动宣告失败；继续看工作进程、加载日志、最终
退出码和 report.passed。不要为此反复并行启动相同模型。若进程确实失败，
保存完整日志后处理具体异常。NVML NUMA affinity Not Supported 在本机也不
等同于 CUDA 内核失败。

PowerShell 重定向可能生成 UTF-16 BOM；先检测 BOM 再转 UTF-8。中文文件用
Get-Content -Encoding utf8 读取。发布 JUnit 时 hostname 替换为 host，不把
未转义的 <host> 放进 XML attribute；替换绝对路径后还应重解析 XML/JSON。
build-identity.json 和日志可能包含工作区路径，应在发布副本中脱敏，保留
wheel 原始字节及其真实 SHA256，不修改 wheel 内的身份记录。
