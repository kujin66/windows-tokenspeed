import os
import sys
from pathlib import Path

os.environ['VIRTUAL_ENV'] = sys.prefix
_dll_dirs = []
site = Path(sys.prefix) / 'Lib/site-packages'
directories = [site / 'torch/lib', site / 'tvm_ffi/lib', Path(os.environ['CUDA_PATH']) / 'bin']
for directory in directories:
    if directory.is_dir():
        _dll_dirs.append(os.add_dll_directory(str(directory)))
os.environ['PATH'] = os.pathsep.join(str(path) for path in directories) + os.pathsep + os.environ['PATH']

if __name__ == '__main__':
    from tools.perf.benchmark_vllm import main
    main()
