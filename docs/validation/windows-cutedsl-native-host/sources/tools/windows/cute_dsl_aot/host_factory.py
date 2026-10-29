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

"""Factories exercising preserved host control flow, tensors and launch attributes."""

import cutlass
import cutlass.cute as cute
import cutlass.utils
from cuda.bindings.driver import CUstream
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import make_fake_tensor, make_ptr
from elementwise_factory import affine


@cute.kernel
def tensor_copy(a: cute.Tensor, out: cute.Tensor):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    by, _, _ = cute.arch.block_dim()
    i = bx * by + tx
    if i < a.shape[0]:
        out[i] = a[i] * 2.0 + 1.0


def compile_tensor(parameters):
    @cute.jit
    def launch(a: cute.Tensor, out: cute.Tensor, stream: CUstream):
        tensor_copy(a, out).launch(
            grid=(cute.ceil_div(a.shape[0], 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    n = cute.sym_int64(divisibility=parameters["divisibility"])
    stride = cute.sym_int64()
    a = make_fake_tensor(cutlass.Float32, (n,), (stride,), assumed_align=16)
    out = make_fake_tensor(cutlass.Float32, (n,), (1,), assumed_align=16)
    return cute.compile(launch, a, out, CUstream(0))


def compile_sequence(parameters):
    @cute.jit
    def launch(
        a: cute.Pointer,
        temporary: cute.Pointer,
        out: cute.Pointer,
        n: cutlass.Int32,
        repetitions: cutlass.Int32,
        stream: CUstream,
    ):
        affine(a, temporary, n, cutlass.Int64(1), 2.0, 1.0).launch(
            grid=(cute.ceil_div(n, 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )
        for _ in range(repetitions):
            affine(temporary, temporary, n, cutlass.Int64(1), 2.0, 1.0).launch(
                grid=(cute.ceil_div(n, 128), 1, 1),
                block=(128, 1, 1),
                stream=stream,
            )
        affine(temporary, out, n, cutlass.Int64(1), 1.0, 0.0).launch(
            grid=(cute.ceil_div(n, 128), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    if parameters:
        raise ValueError("Sequence has no compile-time parameters")
    ptr = make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
    return cute.compile(
        launch, ptr, ptr, ptr, cutlass.Int32(1), cutlass.Int32(1), CUstream(0)
    )


@cute.kernel
def cluster_copy(
    a: cute.Pointer, out: cute.Pointer, n: cutlass.Int32, use_pdl: cutlass.Constexpr
):
    if cutlass.const_expr(use_pdl):
        cute.arch.griddepcontrol_wait()
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    rank = cute.arch.block_idx_in_cluster()
    i = bx * 128 + tx
    if i < n:
        source = cute.make_tensor(a, cute.make_layout(n))
        target = cute.make_tensor(out, cute.make_layout(n))
        target[i] = source[i] + cutlass.Float32(rank)
    if cutlass.const_expr(use_pdl):
        cute.arch.griddepcontrol_launch_dependents()


def compile_cluster(parameters):
    use_pdl = parameters["use_pdl"]

    @cute.jit
    def launch(a: cute.Pointer, out: cute.Pointer, n: cutlass.Int32, stream: CUstream):
        cluster_copy(a, out, n, use_pdl).launch(
            grid=(cute.ceil_div(n, 256) * 2, 1, 1),
            block=(128, 1, 1),
            cluster=(2, 1, 1),
            use_pdl=use_pdl,
            stream=stream,
        )

    ptr = make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
    return cute.compile(launch, ptr, ptr, cutlass.Int32(1), CUstream(0))


def compile_pdl_sequence(parameters):
    if parameters:
        raise ValueError("PDL sequence has no compile-time parameters")

    @cute.jit
    def launch(
        a: cute.Pointer,
        temporary: cute.Pointer,
        out: cute.Pointer,
        n: cutlass.Int32,
        stream: CUstream,
    ):
        cluster_copy(a, temporary, n, True).launch(
            grid=(cute.ceil_div(n, 256) * 2, 1, 1),
            block=(128, 1, 1),
            cluster=(2, 1, 1),
            use_pdl=True,
            stream=stream,
        )
        cluster_copy(temporary, out, n, True).launch(
            grid=(cute.ceil_div(n, 256) * 2, 1, 1),
            block=(128, 1, 1),
            cluster=(2, 1, 1),
            use_pdl=True,
            stream=stream,
        )

    ptr = make_ptr(cutlass.Float32, 0, cute.AddressSpace.gmem, assumed_align=16)
    return cute.compile(launch, ptr, ptr, ptr, cutlass.Int32(1), CUstream(0))


@cute.kernel
def tma_copy(
    atom: cute.CopyAtom,
    source: cute.Tensor,
    out: cute.Tensor,
    layout: cute.Layout,
):
    tx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    allocator = cutlass.utils.SmemAllocator()
    barrier = allocator.allocate(cutlass.Int64, byte_alignment=8)
    shared = allocator.allocate_tensor(cutlass.Float32, layout, byte_alignment=128)
    if tx == 0:
        cute.arch.mbarrier_init(barrier, 1)
        cute.arch.mbarrier_expect_tx(barrier, 1024)
    cute.arch.mbarrier_init_fence()
    cute.arch.barrier()
    tiled = cute.local_tile(source, (256,), (None,))
    s_view, g_view = cpasync.tma_partition(atom, 0, cute.make_layout(1), shared, tiled)
    if cute.arch.warp_idx() == 0:
        cute.copy(atom, g_view[(None, bx)], s_view, tma_bar_ptr=barrier)
    if tx == 0:
        cute.arch.mbarrier_arrive(barrier)
    # Reconverge the issuing lane before other lanes start polling its barrier.
    cute.arch.barrier()
    ready = cutlass.Boolean(False)
    attempts = cutlass.Int32(0)
    while not ready and attempts < 100000:
        ready = cute.arch.mbarrier_test_wait(barrier, 0)
        attempts += 1
    for offset in range(2):
        index = tx + offset * 128
        if ready:
            out[bx * 256 + index] = shared[index]
        else:
            out[bx * 256 + index] = cutlass.Float32(-987654.0)


def compile_tma(parameters):
    if parameters:
        raise ValueError("TMA copy has no compile-time parameters")

    @cute.jit
    def launch(a: cute.Tensor, out: cute.Tensor, stream: CUstream):
        layout = cute.make_layout(256)
        atom, tma_source = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            a,
            layout,
            (256,),
        )
        tma_copy(atom, tma_source, out, layout).launch(
            grid=(cute.ceil_div(a.shape[0], 256), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    n = cute.sym_int64(divisibility=256)
    tensor = make_fake_tensor(cutlass.Float32, (n,), (1,), assumed_align=16)
    return cute.compile(launch, tensor, tensor, CUstream(0))
