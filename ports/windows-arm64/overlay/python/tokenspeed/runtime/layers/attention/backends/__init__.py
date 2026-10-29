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

import sys

# ruff: noqa: E402,F401
# Import all backend modules to trigger register_backend() calls.
from tokenspeed_kernel.platform import current_platform

platform = current_platform()

if sys.platform != "win32":
    from tokenspeed.runtime.layers.attention.backends.specific import (  # noqa: F401
        deepseek_v41,
    )

if platform.is_hopper_plus and sys.platform != "win32":
    from tokenspeed.runtime.layers.attention.backends.paged import (  # noqa: F401
        flashmla,
    )

if platform.is_nvidia and sys.platform != "win32":
    from tokenspeed.runtime.layers.attention.backends.paged import trtllm  # noqa: F401
    from tokenspeed.runtime.layers.attention.backends.paged import (
        trtllm_mla,
    )  # noqa: F401
    from tokenspeed.runtime.layers.attention.backends.paged import (  # noqa: F401
        tokenspeed_mla,
    )

from tokenspeed.runtime.layers.attention.backends.paged import mha  # noqa: F401

if sys.platform != "win32":
    from tokenspeed.runtime.layers.attention.backends.paged import (  # noqa: F401
        ascend_dsa,
        ascend_mla,
        dsa,
        mla,
        msa,
        qsa,
    )
