# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import os
import subprocess
import sys
from pathlib import Path

from tokenspeed.cli._proc import zmq_socket_dir


def test_lower_limit_target_preserves_process_limits():
    # Keep any POSIX limit adjustment inside an expendable child process.
    script = """
import json
import os
from tokenspeed.runtime.utils.common import set_ulimit

if os.name == 'nt':
    before = after = None
    result = set_ulimit(0)
else:
    import resource
    kinds = [resource.RLIMIT_NOFILE, resource.RLIMIT_STACK]
    before = [resource.getrlimit(kind) for kind in kinds]
    result = set_ulimit(0)
    after = [resource.getrlimit(kind) for kind in kinds]
print(json.dumps(dict(before=before, after=after, result=result)))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
        timeout=60,
    )
    limits = json.loads(result.stdout)
    assert limits["result"] is None
    assert limits["before"] == limits["after"]
    assert (limits["before"] is None) == (os.name == "nt")


def test_socket_directory_override_is_preserved(monkeypatch, tmp_path):
    directory = str(tmp_path / "sockets")
    monkeypatch.setenv("SMG_ZMQ_SOCKET_DIR", directory)
    assert zmq_socket_dir() == directory


def test_loaded_cuda_runtime_can_resolve_error_strings():
    from tokenspeed_kernel.thirdparty.cuda.cuda_ipc import (
        CudaRTLibrary,
        find_loaded_library,
    )

    path = find_loaded_library("libcudart")
    assert path is not None
    assert Path(path).is_file()
    runtime = CudaRTLibrary(path)
    assert runtime.lib.cudaGetErrorString(0) == b"no error"
