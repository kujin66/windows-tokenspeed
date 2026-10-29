# 安装包测试与证据保存

## 先核对产物与编译器

在仓库根目录执行，确保本次四个 wheel 已安装：

```powershell
$py = '.venv/Scripts/python.exe'
$env:PYTHONPATH = ''
& $py tools/windows/run.py $py tools/windows/check_native_artifacts.py
if ($LASTEXITCODE -ne 0) { throw 'Installed artifact audit failed' }
& $py tools/windows/run.py $py tools/windows/check_cuda_toolchain.py
if ($LASTEXITCODE -ne 0) { throw 'CUDA toolchain regression failed' }
```

第一项用 IsWow64Process2 检查 native=0xaa64、process=0x0；检查 Python 的 PE、
四个 wheel 的 tag/哈希、六个 PE 的 machine=0xaa64 / linker=14.51，逐字节
比较已安装 .py/原生文件及 build identity；读取 CUDA/version.json 的 SDK
版本。第二项重新编译并执行 4097 元素向量加法。报告分别写到
.port/native-artifacts-verification.json、.port/nvcc1451-layout-regression.json
和 .port/nvcc1451-cuda-smoke-result.json。

## 160 项运行时回归

这些用例覆盖 FP8（80）、NVFP4（25）、插件（34）、dummy drafter（2）和
Rust host Python 逻辑（19）。清空 PYTHONPATH，避免导入源码覆盖 wheel：

```powershell
$env:PYTHONPATH = ''
& $py tools/windows/run.py $py -m pytest test/runtime/test_compressed_tensors_fp8.py test/runtime/test_compressed_tensors_nvfp4.py test/runtime/test_runtime_plugins.py test/runtime/execution/test_dummy_drafter.py test/runtime/test_rust_host.py -q --basetemp=.port/pytest-skill-runtime-new -o cache_dir=.port/pytest-skill-runtime-cache-new --junitxml=.port/installed-runtime.xml > .port/installed-runtime.log 2>&1
if ($LASTEXITCODE -ne 0) { throw 'Installed runtime regression failed; inspect .port/installed-runtime.log' }
```

源码测试可单独设置源码 PYTHONPATH 后执行，但不能与安装包测试重复累计。
SM100/103 参数化用例使用模拟平台元数据，应在报告中写明这一点。

## 26 项物理 GPU / 缓存回归

kernel test/conftest.py 会主动加入源码路径，安装包测试必须禁用它。只为
test/utils 等 helper 设置测试目录 PYTHONPATH，不能加 kernel/python：

```powershell
Push-Location tokenspeed-kernel
try {
    $env:PYTHONPATH = Join-Path $PWD 'test'
    & ../.venv/Scripts/python.exe ../tools/windows/run.py ../.venv/Scripts/python.exe -m pytest test/nvidia/ops/test_cublas_fp8_tensor.py test/nvidia/ops/test_nvfp4_linear.py --noconftest --import-mode=importlib -q --basetemp=../.port/pytest-skill-gpu-new -o cache_dir=../.port/pytest-skill-gpu-cache-new --junitxml=../.port/installed-gpu.xml > ../.port/installed-gpu.log 2>&1
    if ($LASTEXITCODE -ne 0) { throw 'Installed GPU regression failed' }
} finally {
    Pop-Location
    $env:PYTHONPATH = ''
}
```

使用独立 FP32 参考执行数值检查，并验证改变输入后的 CUDA Graph replay。

## BF16 与 NVFP4 真实推理

模型版本、文件 SHA256 与路径见
[总表的检查点记录](../../../docs/guides/windows-wsl2-engine-backend-retest.md)。
模型已在本机缓存时设置 HF_HUB_OFFLINE=1；缺模型时按固定 revision 下载，
校验后再执行，不把缺模型当成本机内核失败。

```powershell
$env:PYTHONPATH = ''
$env:HF_HUB_OFFLINE = '1'
& $py tools/windows/run.py $py tools/windows/validate_inference.py --model ../../vllm-bench/models/Qwen3-0.6B-NVFP4 --report .port/nvfp4-inference.json > .port/nvfp4-inference.log 2>&1
if ($LASTEXITCODE -ne 0) { throw 'NVFP4 inference failed' }
& $py tools/windows/run.py $py tools/windows/validate_inference.py --model .port/models/Qwen3-0.6B --report .port/bf16-inference.json > .port/bf16-inference.log 2>&1
if ($LASTEXITCODE -ne 0) { throw 'BF16 inference failed' }
```

固定设置为 world_size=1、dtype=bfloat16、Triton attention、greedy、eager、
context512、max_num_seqs=2、max_total_tokens=512、gpu_memory_utilization=0.25，
grammar_backend=none，禁用 prefill graph、KV offload 和 retraction snapshot。
NVFP4 检查点权重及矩阵乘保持 NVFP4，dtype=bfloat16 是其余模型计算的 dtype。

每模型 3 次顺序请求加 1 个包含 2 次请求的 batch，各输出 16 token；同一
检查点的 5 份 token ID 必须相同。若代码不应改变数值，还应与此前基线比对。
成功日志有 PASS: three repeat requests and two batched requests；JSON 的 passed
为 true，且退出码为 0。两个不同检查点之间不强求输出相同，不把短推理当
压力测试或模型质量评测。

## 组件改动后的扩大验证

仅当改动或失败需要时增加：

- scheduler C++：通过 runner 用 Ninja 配置独立目录，开启
  TOKENSPEED_SCHEDULER_BUILD_TESTS=ON 和 TOKENSPEED_SCHEDULER_BUILD_PYTHON=ON，可设置
  FETCHCONTENT_SOURCE_DIR_GOOGLETEST 指向 .port/googletest；构建后 ctest
  --test-dir 该构建目录 --output-on-failure。
- scheduler Python：对本机安装包运行 tokenspeed-scheduler/python/tests。
- engine：相同 Rust/MSVC/PYO3 环境下，进入 tokenspeed-engine 后运行
  cargo test --locked；按实际改动选择 crates 或 host 测试，保存退出码。
- 原生 CUDA：通过 runner 运行 tools/windows/check_rmsnorm.py 和
  check_sampling_chain.py。


## 文档与证据交付

为每次复验选择新的 docs/validation 子目录，保留原始失败与后来的修复记录。
复制并脱敏日志、JUnit、编译身份/架构 JSON、推理 JSON；发布副本转 UTF-8，
再重解析 XML/JSON。记录本地时区的时间、git commit 加未提交修改说明、
硬件、依赖位置、确切命令、wheel/文件 SHA256、通过/失败/跳过数。

主张“原生 ARM64”需要进程架构和项目 PE 两方面证据；“14.51 编译”还需要
CMake/compiler identity 与实际 CUDA 编译宏；“可用”需要安装包 GPU 数值和
模型推理。单凭 wheel 文件名、nvcc --version 或能 import 都不足以给出完整结论。
检查 docs/guides/windows-arm64-build-details.md 与 dist/SHA256SUMS.txt 是否仍
对应当前产物；有变化时更新表和快照，不照抄任何未经本轮验证的哈希。

## 创建提交时的完整检查

AGENTS.md 要求提交前执行原样的 pre-commit run --all-files，并提交 formatter
的调整。此步骤不是普通复验或构建的必需项。提交使用 sign-off，并避免在
受 no-commit-to-branch 保护的 main 上直接创建提交。

本机 pre-commit=4.7.0 安装于项目 venv。仓库 hooks 默认 Python3.12，固定的
clang-format18.1.8 没有本机 ARM64 Python 可用的 wheel；源码安装还会触发
旧 cmake.verbose 与新版 scikit-build-core 的冲突。为 hooks 单独准备 x64
Python3.12.15，可保留固定工具版本而不修改项目原生 ARM64 runtime：

```powershell
uv python install cpython-3.12.15-windows-x86_64-none --install-dir .port/precommit-python
if ($LASTEXITCODE -ne 0) { throw 'Pre-commit Python preparation failed' }
$env:PATH = (Join-Path $PWD '.venv/Scripts') + ';' + (Join-Path $PWD '.port/precommit-python/cpython-3.12.15-windows-x86_64-none') + ';' + $env:PATH
$env:PRE_COMMIT_HOME = Join-Path $PWD '.port/pre-commit-x64'
$env:PYTHONUTF8 = '1'
pre-commit run --all-files
```

首次安装 hooks 需要网络。x64 仅用于格式/提交检查，不作为模型进程架构的证据。
Windows checkout 若把 Git symlink 作为文本文件保存，end-of-file-fixer 会给
tokenspeed-engine/crates/ts-scheduler/scheduler 的链接目标追加换行。应在支持
符号链接的本机环境中恢复该路径为真实 symlink，目标保持 ../../../tokenspeed-scheduler；
不要把带换行的目标提交到 Git，也不要为此跳过完整 hook。已存在的目标目录无需删除。
创建本机 NT 相对 symlink 时使用 Windows 路径分隔符，并实际检查 exists/is_dir；
Git 会将链接内容规范为上游的正斜杠形式，单纯 resolve 路径并不能证明链接可访问。
日志换行或空白被规范化后，应更新公开副本的证据 SHA256；原始日志仍保留在
.port。若只有格式调整，不应把旧测试时间改为“重新执行”，或把原 wheel
哈希换成未经重新构建的值。
