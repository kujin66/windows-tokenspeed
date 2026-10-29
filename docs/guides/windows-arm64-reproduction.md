# Windows ARM64 复现与 WSL2 性能对比

本目录是从现有 Windows 移植迁移出的可复现工作区。构建技能见
[SKILL.md](../../skills/tokenspeed-windows-arm64-build/SKILL.md)。
最新的同次开机配对数据见 [重启后性能报告](../validation/windows-wsl2-perf-20261028-after-reboot/README.md)。
慢因排查、算子/同步实验与模型 graph 对照见
[性能诊断](../validation/windows-wsl2-diagnosis-20261028/README.md)。
Windows 构建和 202 项测试证据保存在 [重启前验证目录](../validation/windows-wsl2-perf-20261028/README.md)；
WSL2 安装、依赖和构建见 [Linux ARM64 复现说明](wsl2-arm64-build.md)。
历史测试通过数不重复累计到本轮性能测试。

## 源码版本

| 用途 | 来源 | 版本 |
| --- | --- | --- |
| Windows ARM64 | 原本机 windows-tokenspeed | 9a50ac82，Windows ARM64/CUDA 13.4 移植提交 |
| 共同上游基线 | https://github.com/lightseekorg/tokenspeed | 864bcdd9ef3c5f1512bbe293d970d73eb29d2c94 |
| 最初下载的主分支 | https://github.com/kujin66/windows-tokenspeed | main 分支原样保留 |

当前分支为 windows-arm64。Windows 源码、技能、测试和历史依赖来源记录已经完整包含在本分支。
最初下载的 main 已改用新的 launcher API，删除了旧 Python Engine 接口；
直接覆盖少量文件会把两代接口混在一起，因此本次使用经过验证的移植提交。
WSL2 官方源码已固定到共同上游基线；原始 clone 在 .port/tokenspeed-upstream，
Linux 实际构建目录为 Ubuntu-26.04 内的 /opt/tokenspeed-work/source。
这不是将 Windows 修改合并到最新官方 main 的工作。

## 目标目录内的文件

| 路径 | 内容 |
| --- | --- |
| tools/windows/run.py | 准备 ARM64 MSVC、Windows SDK、CUDA、Python DLL 搜索路径 |
| tools/windows/prepare_local.ps1 | 从现有本机环境复制工具链、依赖归档、模型，创建新的 venv |
| tools/windows/copy_python_dependencies.py | 复制原项目单独安装的十个构建/运行依赖，逐文件保存 SHA256 |
| tools/windows/build_wheels.ps1 | 依次编译、安装四个 wheel，并审计原生架构 |
| requirements/windows-arm64-tested.txt | 本次实际使用的关键依赖版本约束，不是完整 lockfile |
| .port/msvc | 便携 MSVC 14.51，包含 nvcc 识别必需的官方 vcvarsall.bat |
| .port/rust-toolchain | Rust 1.92.0，aarch64-pc-windows-msvc |
| .port/cargo/registry | Cargo 离线 registry 缓存，不复制账号凭据 |
| .port/xgrammar-0.2.8 | engine 静态编译所需 C++ 源码 |
| .port/downloads | 官方 MSVC/Rust/xgrammar 归档及 FlashInfer wheel |
| .port/wheelhouse | 原生 ARM64 PyTorch 和 TVM FFI wheel |
| .port/local-dependencies.json | 归档/ARM64 wheel SHA256 和来源；官方预期哈希与本地拷贝校验分开记录 |
| .port/python-dependencies.json | 额外 Python 依赖的版本及逐文件 SHA256 |
| .port/models/Qwen3-0.6B | BF16 性能测试检查点 |
| dist | 本工作区实际生成的四个项目 wheel |
| tools/perf | 跨 Windows/WSL2 的同一份基准脚本和结果比较器 |

.port、.venv、dist 是本机大文件目录，已被 Git 忽略，但文件实际保存在目标目录。
源码和说明可以纳入版本控制；这里没有把 MSVC/CUDA 安装器作为项目源码重新分发。
系统 Visual Studio、Windows SDK 10.0.26100.0、CUDA SDK 13.4.0 和驱动仍是系统前提。

## Python 依赖与限制

原生 CPython 为 3.13.16。目标 .venv 是新建环境，不是复制旧 venv。
local-runtime.pth 显式共享现有 vllm-bench/.venv/Lib/site-packages 中的依赖；
原项目单独安装的 scikit-build-core、nanobind、tokenspeed-spdlog、wheel、pathspec、
packaging、transformers、flashinfer-python、viztracer、objprint 则复制到目标 venv，
不通过旧 TokenSpeed 的 site-packages 导入项目自身。

关键版本：MSVC 编译器 19.51.36260、CUDA nvcc 13.4.46、PyTorch
2.14.0a0+git20b0f57、TVM FFI 0.1.13.post3、triton-windows 3.8.0.post29、
transformers 5.12.0、FlashInfer 0.7.2。完整来源和公共 URL 见
[技能依赖说明](../../skills/tokenspeed-windows-arm64-build/references/dependencies.md)。

这仍是“本机已准备依赖”流程。wheelhouse 只备份关键 ARM64 wheel，不是全部传递依赖的离线安装源。
主包仍声明部分未移植的服务依赖，构建使用 --no-build-isolation 和 --no-deps；
不声称完成了公共 PyPI 隔离构建、干净机器安装、Python grammar、多 GPU 或全量服务验证。

## 在本机准备和编译

在此仓库根目录的 PowerShell 执行。下面的相对路径已经按 official-repos 中的目标目录调整：


```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools/windows/prepare_local.ps1 -SourceRepo ../../windows-tokenspeed -RuntimeSitePackages ../../vllm-bench/.venv/Lib/site-packages -NativeWheelhouse ../../N1X_ARM_Package_Support/wheelhouse
powershell.exe -NoProfile -ExecutionPolicy Bypass -File tools/windows/build_wheels.ps1 -BuildId migration-20261028
```

第一次准备会复制数 GB 文件。再次执行不会刷新已经存在的工具链目录；更换工具链时应选择新的工作区或明确核对版本，不能假设目录存在就等于正确版本。
build_wheels.ps1 使用 Ninja、新 scheduler build-dir、Cargo --locked/offline、dense-triton profile 和 CUDA arch=121，frontend helper 为空。
依次产生 scheduler、engine、kernel、runtime 四个 wheel；dist 必须只包含本次四个候选 wheel，避免混装历史版本。
最终审计逐文件比较安装包与 wheel，并检查 Python 无 WOW64 仿真、六个 PE 为 ARM64、linker=14.51、CUDA build identity。

```powershell
$py = './.venv/Scripts/python.exe'
$env:PYTHONPATH = ''
& $py tools/windows/run.py $py tools/windows/check_cuda_toolchain.py
& $py tools/windows/run.py $py -m pytest test/runtime/test_compressed_tensors_fp8.py test/runtime/test_compressed_tensors_nvfp4.py test/runtime/test_runtime_plugins.py test/runtime/execution/test_dummy_drafter.py test/runtime/test_rust_host.py -q --basetemp=.port/pytest-runtime-tmp-new -o cache_dir=.port/pytest-runtime-cache-new --junitxml=.port/installed-runtime-new.xml
Push-Location tokenspeed-kernel
try {
    $env:PYTHONPATH = Join-Path $PWD 'test'
    & ../.venv/Scripts/python.exe ../tools/windows/run.py ../.venv/Scripts/python.exe -m pytest test/nvidia/ops/test_cublas_fp8_tensor.py test/nvidia/ops/test_nvfp4_linear.py --noconftest --import-mode=importlib -q --basetemp=../.port/pytest-gpu-tmp-new -o cache_dir=../.port/pytest-gpu-cache-new --junitxml=../.port/installed-gpu-new.xml
} finally { Pop-Location; $env:PYTHONPATH = '' }
```

本机系统共享 pytest 临时目录存在 ACL 冲突，测试应使用上述仓库内独立临时/缓存
目录。每轮为 tmp/cache/report 选择新名字，保留失败日志；不要调整系统 Temp 权限。

## 同条件性能测试

采用 Qwen3-0.6B BF16、单 GPU、Triton attention、greedy、eager，关闭 prefix caching、KV offload、prefill graph 和 retraction snapshot。
输入直接使用固定随机种子生成的 token IDs，避免两端 tokenizer 差异；输出强制固定长度并忽略 EOS。
输入和引擎均显式设置 seed=20261028，避免默认引擎种子随机变化。
模型上下文在最长输入加输出之外再预留一页 64 token，故本配置 max_model_len=704、
max_total_tokens=2816；仅设置为 640 会触发引擎保留窗口并截短最长请求。
每个 case 先预热 3 次，再测量 10 次；输入长度 128/512、输出长度 128、batch=1/4。
统计阻塞 Engine.generate 的端到端耗时，包含 prefill/decode 和 IPC；排除加载、JIT 预热、结果验证。
吞吐量为全部实测输出 token 数除以全部实测秒数，不是逐轮吞吐量的算术平均。
此测试不测 TTFT/ITL、HTTP 服务、CUDA graph 或长期并发负载。

```powershell
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONPATH = ''
& ./.venv/Scripts/python.exe tools/windows/run.py ./.venv/Scripts/python.exe tools/perf/benchmark_tokenspeed.py --platform windows-arm64 --attention-backend triton --consistency-policy record --model .port/models/Qwen3-0.6B --source-root . --upstream-base 864bcdd9ef3c5f1512bbe293d970d73eb29d2c94 --input-lengths 128 512 --batch-sizes 1 4 --output-length 128 --warmup 3 --rounds 10 --report .port/windows-bf16-perf.json
```

逐轮耗时、环境版本、GPU、模型 SHA256、benchmark 文件 SHA256、输入摘要、输出 IDs 和源码身份均写入 JSON。
任一请求输出长度不一致都会失败；不能把部分完成的 JSON 当成有效基线。
consistency-policy=record 保存每轮 greedy IDs 和一致性标志；require 则在 ID 差异时失败。
本机先前严格试跑在 batch=4 观察到重复 greedy IDs 差异，原因尚未定位。最终性能
测试采用 record：完成固定长度负载的吞吐测量，同时保留输出差异，不将其解释为
模型输出数值等价证明。

## WSL2 安装与官方源码

用户已授权安装 WSL2。首先尝试官方 wsl --install 下载入口；HTTP 403 后改用微软官方 GitHub ARM64 MSI。
本次下载 https://github.com/microsoft/WSL/releases/download/3.0.3/wsl.3.0.3.0.arm64.msi，
安装前 Authenticode 验证为 Valid，签名者 Microsoft Corporation；SHA256：
2937c7ce840bb35ec6e97f052189bdfff27f8c3463bfdd231df9a958212787a3。
安装命令使用 /qn /norestart；工具不自动重启主机。
管理员启用 VirtualMachinePlatform 的脚本为 tools/windows/enable_wsl2.ps1，使用 -NoRestart。
安装进度和错误保存在 .port/wsl2-install；本轮实际结果以验证目录为准。

重启前启用 VirtualMachinePlatform 后，系统返回 restart_needed=true，Ubuntu 注册曾
返回 HCS_E_SERVICE_NOT_AVAILABLE。用户重启后已完成 Ubuntu-26.04 注册和安装，
WSL2 GPU、Linux AArch64 原生构建和 BF16 模型推理均已验证。
Ubuntu 26.04.1 ARM64 镜像按微软 DistributionInfo.json 下载并校验，SHA256 为
c7d893d7ccb291009e6096c5a946937337b6fd4c426f11b6541f6cd2bf894f7d。

以下是已执行的发行版安装命令，现有环境无需再次注册：

```powershell
& 'C:/Program Files/WSL/wsl.exe' --install --from-file .port/wsl2-install/ubuntu-26.04.1-wsl-arm64.wsl --no-launch --location .port/wsl2-ubuntu
& 'C:/Program Files/WSL/wsl.exe' --list --verbose
```

Linux 实际工作目录为 /opt/tokenspeed-work，使用独立 CPython 3.13.16 venv。
完整依赖、CUDA SDK、Rust、四个 wheel 构建及修复步骤见
[WSL2 ARM64 构建](wsl2-arm64-build.md)。Linux 使用官方源码共同基线；
没有把 Windows win_arm64 wheel 安装到 Linux，也没有改写 Linux 推理源码。

## 配对性能结果

重启后先结束本任务的 Ubuntu 发行版，完成 Windows 测量，再启动 WSL2 运行
同一脚本和工作负载，避免两个基准争抢 GPU。每侧每个 case 预热 3 次、实测 10 次，
输入 128/512、输出固定 128、batch 1/4，共 100 个实测请求和 12,800 个实测输出 token。

结果、逐轮数据、输出 ID 差异、环境版本和构建审计见
[重启后配对报告](../validation/windows-wsl2-perf-20261028-after-reboot/README.md)。
重启前 Windows 报告保留为历史，没有混入本轮。

可从保留的原始报告重新运行比较器：

```powershell
& ./.venv/Scripts/python.exe -m tools.perf.compare_results --windows .port/windows-bf16-perf-after-reboot.json --wsl2 .port/wsl2-artifacts/perf.json --report .port/windows-vs-wsl2-bf16-after-reboot.json
```

比较器拒绝未完成报告、不同模型哈希、benchmark、上游基线、GPU 或 workload，
并单独报告两端重复请求及跨平台 greedy IDs 是否相同。比较器的百分比为
(Windows tok/s ÷ WSL2 tok/s − 1) × 100%，正值表示 Windows 更快；
报告表格另列 WSL2/Windows 和 WSL2 相对 Windows 的百分比，并明确方向。
两侧 PyTorch、Triton、CUDA 运行库及平台实现有差异，结果描述两套实际软件栈，
不能把全部差异归因于操作系统。
