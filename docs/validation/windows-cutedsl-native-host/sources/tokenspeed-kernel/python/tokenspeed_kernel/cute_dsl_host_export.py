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

"""Compiler-process-only export of a native Windows ARM64 host launcher.

Use the installed compiler's C export API; never import this in native Torch.
No installed compiler source is modified. Unsupported signatures fail explicitly.
"""

import functools
import hashlib
import re
from pathlib import Path

TARGET = "aarch64-pc-windows-msvc"
PREFIX = "tokenspeed_aot"
_compile_arguments = {}

C_TYPES = {
    "int8_t": "i8",
    "uint8_t": "u8",
    "int16_t": "i16",
    "uint16_t": "u16",
    "int32_t": "i32",
    "uint32_t": "u32",
    "int64_t": "i64",
    "uint64_t": "u64",
    "float": "f32",
    "double": "f64",
    "bool": "bool",
}


def install_compiler_hook():
    """Retarget cute.compile calls made by the supplied factory in this worker."""
    import cutlass.cute as cute

    original = cute.compile

    @functools.wraps(original)
    def cross_compile(*args, **kwargs):
        options = kwargs.get("options", "")
        if not isinstance(options, str):
            raise ValueError("Host export currently requires string compile options")
        if "--host-target" not in options:
            options += f" --host-target 'llvm -mtriple={TARGET}'"
        kwargs["options"] = options + " --keep-ptx --keep-cubin"
        compiled = original(*args, **kwargs)
        runtime_kwargs = {k: v for k, v in kwargs.items() if k != "options"}
        values = compiled.execution_args.get_rectified_args(args[1:], runtime_kwargs)
        _compile_arguments[id(compiled)] = dict(
            zip(compiled.execution_args.signature.parameters, values)
        )
        return compiled

    cute.compile = cross_compile


def _value_contract(value, symbols):
    from cutlass.cute.typing import SymInt

    if type(value) is int:
        return {"static": value}
    if isinstance(value, SymInt):
        symbol = symbols.setdefault(id(value), len(symbols))
        return {
            "symbol": symbol,
            "width": value.width,
            "divisibility": value.divisibility,
        }
    raise ValueError(
        "Automatic host tensor adaptation requires flat int/SymInt layouts"
    )


def export_host(compiled, directory: Path) -> dict:
    """Export generated host code and enough C ABI metadata for native callers."""
    from cutlass.cute.runtime import _FakeTensor, _Pointer

    if compiled.host_target.triple != TARGET:
        raise ValueError("The factory did not compile for Windows ARM64")
    compiled.export_to_c(str(directory), "host", PREFIX)
    header = (directory / "host.h").read_text(encoding="utf-8")
    wrapper = re.findall(r"static inline int32_t (\w+_wrapper)\(", header)
    if len(wrapper) != 1:
        raise ValueError("Expected the CUDA dialect's C-compatible host wrapper")
    captured = _compile_arguments[id(compiled)]
    info = compiled.c_header_arguments
    symbols = {}
    parameters = []
    for argument in info.arguments:
        argument = argument.replace(info.dummy_prefix_name, PREFIX)
        match = re.fullmatch(r"(.*?)([A-Za-z_]\w*)", argument.strip())
        if match is None:
            raise ValueError(f"Unsupported host C parameter: {argument}")
        ctype, name = match[1].strip(), match[2]
        sample = captured.get(name)
        parameter = {"name": name, "ctype": ctype}
        if ctype in ("cudaStream_t", "CUstream"):
            parameter["kind"] = "stream"
        elif isinstance(sample, _FakeTensor):
            parameter.update(
                kind="tensor",
                dtype=sample.element_type.__name__,
                alignment=sample._assumed_align,
                shape=[_value_contract(v, symbols) for v in sample.shape],
                stride=[_value_contract(v, symbols) for v in sample.stride],
            )
            typedefs = "\n".join(info.declarations).replace(
                info.dummy_prefix_name, PREFIX
            )
            struct = re.search(
                r"typedef struct\s*\{([^{}]*)\}\s*"
                + re.escape(ctype.rstrip("* ").strip())
                + r"\s*;",
                typedefs,
                re.S,
            )
            if struct is None:
                raise ValueError(f"Missing tensor descriptor declaration for {name}")
            fields = []
            for field in struct[1].split(";"):
                field = field.strip()
                if not field:
                    continue
                parsed = re.fullmatch(
                    r"(.*?)\s*(data|dynamic_shapes|dynamic_strides)(?:\[(\d+)\])?",
                    field,
                )
                if parsed is None:
                    raise ValueError(f"Unsupported tensor descriptor field: {field}")
                fields.append(
                    {
                        "ctype": parsed[1].strip(),
                        "name": parsed[2],
                        "count": int(parsed[3]) if parsed[3] else None,
                    }
                )
            parameter["fields"] = fields
        elif isinstance(sample, _Pointer):
            if ctype.replace(" ", "") != "void*":
                raise ValueError("Expected a C void pointer for the compiled pointer")
            parameter.update(
                kind="ptr", dtype=sample.dtype.__name__, alignment=sample._assumed_align
            )
        elif ctype in C_TYPES:
            parameter["kind"] = C_TYPES[ctype]
        else:
            # Aggregate signatures remain usable with an explicit by-value payload.
            parameter["kind"] = "opaque"
        parameters.append(parameter)
    locals_ = []
    sizes = []
    names = []
    for i, p in enumerate(parameters):
        ctype = p["ctype"]
        name = f"arg_{i}"
        locals_.append(
            f"{ctype} {name}; std::memcpy(&{name}, args[{i}], sizeof({name}));"
        )
        sizes.append(f"case {i}: return sizeof({ctype});")
        names.append(name)
    adapter = f"""
#include "host.h"
#include <cstring>
extern "C" int {PREFIX}_cuda_num_binaries();
extern "C" int {PREFIX}_cuda_init(void*);
extern "C" int {PREFIX}_cuda_load_to_device(void*, int);
static {PREFIX}_Kernel_Module_t module = {{}};
extern "C" __declspec(dllexport) int ts_prepare(int device) {{
    if ({PREFIX}_cuda_num_binaries() != 1) return int(cudaErrorNotSupported);
    int status = {PREFIX}_cuda_init(&module.module);
    if (status) return status;
    status = {PREFIX}_cuda_load_to_device(&module.module, device);
    if (status) {{ cudaLibraryUnload(module.module); module.module = nullptr; }}
    return status;
}}
extern "C" __declspec(dllexport) int ts_release() {{
    if (!module.module) return 0;
    int status = cudaLibraryUnload(module.module);
    if (!status) module.module = nullptr;
    return status;
}}
extern "C" __declspec(dllexport) size_t ts_parameter_size(int index) {{
    switch (index) {{ {" ".join(sizes)} default: return 0; }}
}}
extern "C" __declspec(dllexport) int ts_run(void** args) {{
    {" ".join(locals_)}
    return {wrapper[0]}(&module, {", ".join(names)});
}}
"""
    (directory / "adapter.cpp").write_text(adapter, encoding="utf-8")
    return {
        "target": TARGET,
        "parameters": parameters,
        "prefix": PREFIX,
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
