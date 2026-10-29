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

"""Record actual one-operation CUDA graph kernel names for both backends."""

import argparse
import json
import re
from pathlib import Path

import torch
from windows_router_aot import RouterAOT


def inspect(directory):
    torch.cuda.set_device(0)
    summary = []
    for m, bs in ((1, 128), (2, 128), (4, 64)):
        a = torch.randn(m, 7168, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(896, 7168, dtype=torch.bfloat16, device="cuda")
        out = torch.empty(m, 896, dtype=torch.float32, device="cuda")
        loader = RouterAOT(
            directory,
            shape_mnk=(m, 896, 7168),
            block_size=bs,
            device=torch.device("cuda:0"),
        )
        try:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                loader(a, b, out=out)
                torch.mm(a, b.t(), out_dtype=torch.float32, out=out)
            torch.cuda.current_stream().wait_stream(stream)
            graphs = []
            row = {"m": m, "aot_block_size": bs}
            for name in ("aot", "torch"):
                graph = torch.cuda.CUDAGraph()
                graph.enable_debug_mode()
                with torch.cuda.graph(graph):
                    if name == "aot":
                        loader(a, b, out=out)
                    else:
                        torch.mm(a, b.t(), out_dtype=torch.float32, out=out)
                graph.replay()
                dot_path = directory / f"{name}-m{m}-graph.dot"
                graph.debug_dump(str(dot_path))
                dot = dot_path.read_text(encoding="utf-8")
                row[name] = {
                    "kernel_nodes": dot.count('label="{KERNEL'),
                    "kernel_labels": re.findall(r"topoId: \d+\) \| ([^\n]+)", dot),
                }
                graphs.append(graph)
            summary.append(row)
            torch.cuda.synchronize()
            del graphs, graph
        finally:
            loader.close()
    (directory / "graph-kernels.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    inspect(parser.parse_args().directory)
