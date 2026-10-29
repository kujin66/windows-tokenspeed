# 构建与安装

## 工作目录与环境入口

先读 [依赖前提](dependencies.md)。以下均在包含 Windows 移植修改的仓库根目录
执行。现有 .venv 可用时用它，不向系统 Python 安装包。PowerShell 限制激活脚本
时直接调用 .venv/Scripts/python.exe 即可；激活不是改变 Python 架构的方法。

```powershell
$py = '.venv/Scripts/python.exe'
$ErrorActionPreference = 'Stop'
function Invoke-TokenSpeedTool {
    & $py tools/windows/run.py @args
    if ($LASTEXITCODE -ne 0) { throw "TokenSpeed command failed with exit code $LASTEXITCODE" }
}
Invoke-TokenSpeedTool nvcc --version
```

cl 不带源码会返回非零；实际自动化使用下面的 CUDA 编译回归判断可用性，
不将 cl 的 usage 返回码当编译失败。

普通终端中的 nvcc 入口为 tools/windows/nvcc.cmd，它包装 run.py 并添加
--use-local-env。交互环境及其检验入口为：

```powershell
& ./tools/windows/nvcc.cmd --version
powershell.exe -NoProfile -ExecutionPolicy Bypass -NoExit -File ./tools/windows/Activate-Toolchain.ps1
# 以下在另一个普通终端执行，完成后退出：
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ./tools/windows/check_toolchain_activation.ps1
```

ExecutionPolicy Bypass 仅作用于新进程，不修改系统策略。不要把上述 -NoExit
交互命令当成自动构建步骤，否则会等待终端退出。

## runner 准备的内容

tools/windows/run.py 通过 setuptools 的 _get_vc_env("arm64") 获取已安装 VS/SDK
环境，然后将旧 VCTOOLSINSTALLDIR 在 PATH/INCLUDE/LIB/LIBPATH 中替换为便携
14.51 工具链；它不会安装 VS 或 Windows SDK。

| 设置 | 值或作用 |
| --- | --- |
| TOKENSPEED_MSVC_ROOT | 可选替换便携工具根目录；默认 .port/msvc/Contents/VC/Tools/MSVC/14.51.36231 |
| VCTOOLSINSTALLDIR / VCTOOLSVERSION | 指向该目录，版本目录名 14.51.36231 |
| CC / CXX | cl，PATH 前置 Hostarm64/arm64 |
| CUDA_HOME / CUDA_PATH | 固定为本机 CUDA/v13.4 目录 |
| CUDA_VISIBLE_DEVICES | 0 |
| CMAKE_GENERATOR | 未显式指定时 Ninja；构建前显式设 Ninja 防止继承其他值 |
| MAX_JOBS / CMAKE_BUILD_PARALLEL_LEVEL | 1 / 2，限制本机内存占用 |
| DISTUTILS_USE_SDK / MSSdk | 1，复用准备好的 SDK 环境 |
| VIRTUAL_ENV / PYTHONUTF8 | 项目 .venv / 1 |
| Python DLL / libs | PATH 加解释器目录和 base_prefix，LIB 加 base_prefix/libs |
| TRITON_CACHE_DIR | 默认 .port/triton |
| FLASHINFER_WORKSPACE_BASE / FLASHINFER_CACHE_DIR | 默认 .port/flashinfer |
| TOKENSPEED_FLASHINFER_AUTOTUNE_CACHE_DIR | 默认 .port/autotune |

缓存环境变量保留显式提供的值。run.py --print-env 只导出准备后改变的环境项，
可用于激活脚本，不将整份继承环境（可能含凭据）发布到文档。

## 编译探测与版本证据

```powershell
Invoke-TokenSpeedTool $py tools/windows/check_cuda_toolchain.py
```

该回归只移动仓库内复制的 vcvarsall.bat，先重现缺少布局时的
Host compiler targets unsupported OS，再让 runner 自动恢复官方文件，编译
tools/windows/cuda_smoke.cu 并执行。完成前不要同时运行其他 nvcc 构建。
预期 _MSC_VER=1951、_MSC_FULL_VER=195136260、runtime=13040、4097 元素正确。

## 四个 wheel 的构建顺序

上游源码基线为 864bcdd9ef3c5f1512bbe293d970d73eb29d2c94，Windows 移植
已在 9a50ac82 提交；本地复现脚本和文档扩展仍属于工作区修改。发布时同时保存
提交身份和未提交修改，不能只记录上游 SHA。
以下命令使用本机已准备的依赖，--no-deps 不会补全 runtime 或 build 依赖。

```powershell
$env:PATH = (Join-Path $PWD '.port/rust-toolchain/bin') + ';' + $env:PATH
$env:CARGO_HOME = Join-Path $PWD '.port/cargo'
$env:PYO3_PYTHON = Join-Path $PWD '.venv/Scripts/python.exe'
$env:FETCHCONTENT_SOURCE_DIR_XGRAMMAR = Join-Path $PWD '.port/xgrammar-0.2.8'
$env:CMAKE_GENERATOR = 'Ninja'
$env:CMAKE_ARGS = '-DCMAKE_TRY_COMPILE_CONFIGURATION=Release'
$env:PYTHONPATH = ''
New-Item -ItemType Directory -Force dist | Out-Null

# 1. C++ scheduler：独立目录避免继承 Visual Studio / 14.44 的缓存。
$schedulerBuild = Join-Path $PWD '.port/scheduler-wheel-msvc1451'
Invoke-TokenSpeedTool $py -m pip wheel ./tokenspeed-scheduler --no-build-isolation --no-deps --wheel-dir dist --config-settings=build-dir=$schedulerBuild

# 2. Rust engine：setup.py 先编译 Python 扩展，再编译嵌入 Python 的 host。
Invoke-TokenSpeedTool $py -m pip wheel ./tokenspeed-engine --no-build-isolation --no-deps --wheel-dir dist

# 3. CUDA kernel：显式选择本机 dense profile。
$env:TOKENSPEED_KERNEL_BACKEND = 'cuda'
$env:TOKENSPEED_KERNEL_WINDOWS_PROFILE = 'dense-triton'
$env:TOKENSPEED_CUDA_ARCH = '121'
$env:TOKENSPEED_KERNEL_CUDA_FRONTEND_CCBIN = ''
Invoke-TokenSpeedTool $py -m pip wheel ./tokenspeed-kernel/python --no-build-isolation --no-deps --wheel-dir dist

# 4. Python runtime。
Invoke-TokenSpeedTool $py -m pip wheel ./python --no-build-isolation --no-deps --wheel-dir dist
```

kernel 的 dev 版本由构建日期和 git SHA 派生。重现此轮文件名可显式设置
TOKENSPEED_KERNEL_VERSION_DATE=20261027 和 TOKENSPEED_KERNEL_GIT_SHA=864bcdd9；
新构建通常应使用实际日期/源码身份，而不是沿用旧名。相同文件名不保证二进制
逐字节相同，编译时间、绝对路径和 ZIP 时间戳会影响 SHA256。

engine 的 setup.py 使用 cargo build --release --locked，分别构建 tokenspeed-py
和 tokenspeed-host。只有扩展设置 PYO3_BUILD_EXTENSION_MODULE=1；host 需链接
当前解释器的 import library。构建脚本处理 Windows checkout 中 scheduler
symlink 变成文本的情况。engine CMake 部分使用 Ninja 和 xgrammar 静态 core。

更换编译器或生成器时选择新的 build-dir；Cargo 如需完全独立重建可将
CARGO_TARGET_DIR 指向新的仓库内目录。不要递归删除未检查路径，也不必为复验
未变的 wheel 删除整个 .port。

## CUDA 编译、链接与输出

dense-triton 构建 rmsnorm_fused_parallel、sampling_chain、cublas_fp8_tensor 三组。
当前实际身份记录包括 C++17、-O3、-DNDEBUG、-use_fast_math、
--expt-relaxed-constexpr、--use-local-env，以及 MSVC /MD,/EHsc,/utf-8,
/Zc:__cplusplus,/Zc:preprocessor。目标实际生成 compute_121a / sm_121a；
不要照抄上游默认的 SM100a/103a 作为本机产物。

链接使用 CUDA/lib/arm64 下的 cudart.lib、cuda.lib，TVM FFI 的 tvm_ffi.lib；
FP8 组还使用 -lcublasLt。每组输出目录内的 build-identity.json 保存编译器、
flags、include_dirs、link_flags 和空 frontend helper。最终 .so 扩展名沿用
上游约定，内容为 Windows ARM64 PE DLL，由 TVM FFI 加载，不是 Linux ELF。

NVFP4 的核心矩阵乘不依赖另一个 AOT .so，而由 Triton tl.dot_scaled 编译为
原生 SM12x FP4 Tensor Core 指令。压缩权重保持 packed，按 16 元素分组，
融合投影保留各自的 global scale；compressed-tensors 的 global scale 使用
divisor/inverse 约定，不能简单当成乘法 scale 替换。

tools/windows/build_kernels.py 可按组调试，但它不替代最终 wheel 打包与安装
审计。build_split_cuda.py 和旧 14.44 frontend 实验属于历史排查路径，不是
当前 14.51 直接编译方案。

## 安装与交付

只安装本次明确选出的四个 wheel，防止 dist 中历史版本一起被安装。可先查看：

```powershell
Get-ChildItem dist/*.whl
# 将以下值替换为本次构建的四个准确文件名；当前版本见产物总表。
$wheels = @(
    'dist/tokenspeed_scheduler-0.1.42-cp313-cp313-win_arm64.whl'
    'dist/tokenspeed_engine-0.1.0-cp313-cp313-win_arm64.whl'
    'dist/tokenspeed_kernel-0.1.4.dev20261028+git9a50ac82-cp313-cp313-win_arm64.whl'
    'dist/tokenspeed-0.1.1-py3-none-any.whl'
)
foreach ($wheelPath in $wheels) {
    & $py -m pip install --force-reinstall --no-deps --no-index $wheelPath
    if ($LASTEXITCODE -ne 0) { throw "Wheel installation failed: $wheelPath" }
}
Invoke-TokenSpeedTool $py tools/windows/check_native_artifacts.py
```

审计脚本当前期望 dist 中恰好四个项目 wheel、六个项目原生二进制；如果功能
范围或包结构改变，应在审阅后更新断言，而不是删掉不满足断言的真实产物。
正常流程最后执行 [安装包与推理验证](validation.md)，再更新 docs 中的当前表。
