"""Experimental BF16 SM12.1 paged prefill through FlashInfer FA2."""
import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    'attention', 'mha_extend_with_kvcache',
    name='flashinfer_fa2_mha_extend_sm121', solution='flashinfer',
    capability=CapabilityRequirement(min_arch_version=ArchVersion(12, 1), max_arch_version=ArchVersion(12, 1), vendors=frozenset({'nvidia'})),
    signatures=format_signatures(('q', 'k_cache', 'v_cache'), 'dense', {torch.bfloat16}),
    priority=Priority.SPECIALIZED + 1,
    traits={'head_dim': frozenset({128}), 'is_causal': frozenset({True}), 'logit_cap': frozenset({False}), 'return_lse': frozenset({False}), 'sinks': frozenset({False}), 'sliding_window': frozenset({False})},
)
def flashinfer_fa2_mha_extend_sm121(
    *, q, k_cache, v_cache, page_table, cache_seqlens, cu_seqlens_q, cu_seqlens_kv,
    max_seqlen_q, max_seqlen_k, is_causal, window_left, logit_cap, sinks,
    return_lse, softmax_scale, q_scale, k_scale, v_scale, enable_pdl,
    fp8_kv_scales=None,
):
    if not is_causal or window_left != -1 or logit_cap != 0 or sinks is not None or return_lse:
        raise NotImplementedError('SM12.1 comparison supports causal full-context BF16 prefill without LSE/sinks/logit cap')
    if any(scale is not None for scale in (q_scale, k_scale, v_scale, fp8_kv_scales)):
        raise NotImplementedError('SM12.1 comparison does not support quantized attention')
    if q.dtype != torch.bfloat16 or k_cache.dtype != q.dtype or v_cache.dtype != q.dtype:
        raise ValueError('BF16 Q/K/V are required')
    q_indptr = cu_seqlens_q.to(device='cpu', dtype=torch.int32)
    kv_cumulative = cu_seqlens_kv.to(device='cpu', dtype=torch.int32)
    lengths = cache_seqlens.to(device='cpu', dtype=torch.int32)
    pages = page_table.to(device='cpu', dtype=torch.int32)
    if not torch.equal(kv_cumulative[1:] - kv_cumulative[:-1], lengths):
        raise ValueError('KV cumulative lengths disagree with cache lengths')
    if int(lengths.max()) > max_seqlen_k or int((q_indptr[1:] - q_indptr[:-1]).max()) > max_seqlen_q:
        raise ValueError('Actual sequence lengths exceed explicit upper bounds')
    page_size = k_cache.shape[1]
    counts = (lengths + page_size - 1) // page_size
    indptr = torch.cat((torch.zeros(1, dtype=torch.int32), counts.cumsum(0, dtype=torch.int32)))
    indices = torch.cat([pages[row, :int(count)] for row, count in enumerate(counts)])
    last = lengths - (counts - 1) * page_size
    import tokenspeed_kernel.ops.attention.mha.flashinfer as backend
    if backend._workspace_buffer is None:
        backend._workspace_buffer = torch.zeros(512 * 1024 * 1024, dtype=torch.uint8, device=q.device)
    wrapper = BatchPrefillWithPagedKVCacheWrapper(backend._workspace_buffer, kv_layout='NHD', backend='fa2')
    wrapper.plan(q_indptr, indptr, indices, last, q.shape[1], k_cache.shape[2], q.shape[2], page_size,
                 causal=is_causal, sm_scale=softmax_scale, q_data_type=q.dtype, kv_data_type=k_cache.dtype,
                 window_left=window_left, logits_soft_cap=logit_cap)
    return wrapper.run(q, (k_cache, v_cache), return_lse=return_lse, enable_pdl=enable_pdl)
