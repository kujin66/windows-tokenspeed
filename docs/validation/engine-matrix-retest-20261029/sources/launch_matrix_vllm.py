import os
import runpy
import sys
from pathlib import Path

os.environ['VIRTUAL_ENV'] = sys.prefix
site = Path(sys.prefix) / 'Lib/site-packages'
dirs = [site/'torch/lib', site/'tvm_ffi/lib', Path(os.environ['CUDA_PATH'])/'bin']
handles = [os.add_dll_directory(str(d)) for d in dirs if d.is_dir()]
os.environ['PATH'] = os.pathsep.join(map(str,dirs))+os.pathsep+os.environ['PATH']
if __name__ == '__main__':
    script = sys.argv.pop(1)
    sys.argv[0] = script
    runpy.run_path(script,run_name='__main__')
