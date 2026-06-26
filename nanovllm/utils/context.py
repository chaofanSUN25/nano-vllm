from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    
    # Layer-level drop支持
    drop_mask: torch.Tensor | None = None  # 标记哪些sequence需要被drop
    num_seqs: int = 0  # batch中的sequence数量
    layer_drop_callback = None  # layer间drop回调函数

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, drop_mask=None, num_seqs=0, layer_drop_callback=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, drop_mask, num_seqs, layer_drop_callback)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
