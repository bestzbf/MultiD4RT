"""Attention with camera identity expressed as a learned same-camera relation.

Whole-query CUDA attention factors the relation into extra Q/K features, so
unmasked SDPA can use fused kernels without a dense learned mask. The reference
and chunked paths materialize their query range's bias, checkpointing chunks.
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _safe_key_valid(key_valid: Optional[torch.Tensor], num_keys: int):
    if key_valid is None:
        return None, None
    any_valid = key_valid.any(dim=-1)
    # All-masked attention can produce NaNs on some SDPA backends. Unmask one
    # dummy key for such batch items and zero their result afterwards.
    dummy_key = torch.arange(num_keys, device=key_valid.device) == 0
    safe_valid = key_valid | ((~any_valid)[:, None] & dummy_key[None, :])
    return safe_valid, any_valid


def _relation_chunk(q, k, v, weight, query_camera, key_camera, key_valid, dropout_p):
    same_camera = (
        (query_camera[:, :, None] == key_camera[:, None, :])
        & (query_camera[:, :, None] >= 0)
        & (key_camera[:, None, :] >= 0)
    )
    bias = same_camera[:, None].to(q.dtype) * weight.to(q.dtype)[None, :, None, None]
    safe_valid, any_valid = _safe_key_valid(key_valid, k.shape[-2])
    if safe_valid is not None and not safe_valid.all():
        bias = bias.masked_fill(~safe_valid[:, None, None, :], float("-inf"))
    result = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=dropout_p)
    if any_valid is not None:
        result = result * any_valid[:, None, None, None].to(result.dtype)
    return result


def _feature_relation_attention(q, k, v, weight, query_camera, key_camera, key_valid, dropout_p):
    """Factor alpha[h]*1[cq==ck] into Q/K, with the original SDPA scale.

    Q'=[Q, sqrt(d)*alpha[h]*one_hot(cq)], K'=[K, one_hot(ck)].
    Q'K'^T/sqrt(d)=QK^T/sqrt(d)+alpha[h]*1[cq==ck]. Negative
    camera IDs have all-zero features. Zero-padding V keeps its output exact.
    Parameter names/shapes and the relation-weight gradient are unchanged.
    """
    labels = torch.cat([query_camera, key_camera], dim=1)
    cameras, inverse = torch.unique(labels, return_inverse=True)
    hot = F.one_hot(inverse, num_classes=len(cameras)).to(q.dtype)
    hot = hot*(labels >= 0).unsqueeze(-1)
    query_hot, key_hot = hot.split([q.shape[-2], k.shape[-2]], dim=1)
    d, value_dim = q.shape[-1], v.shape[-1]
    width = ((max(d+len(cameras), value_dim)+7)//8)*8
    if q.is_cuda and width > 256:
        return None  # preserve the general reference path for very many views
    extra_q = query_hot[:, None]*weight.to(q.dtype)[None, :, None, None]*(d**.5)
    extra_k = key_hot[:, None].expand(-1, k.shape[1], -1, -1)
    extended_q = F.pad(torch.cat([q, extra_q], dim=-1), (0, width-d-len(cameras)))
    extended_k = F.pad(torch.cat([k, extra_k], dim=-1), (0, width-d-len(cameras)))
    extended_v = F.pad(v, (0, width-value_dim))
    safe_valid, any_valid = _safe_key_valid(key_valid, k.shape[-2])
    mask = None if safe_valid is None or safe_valid.all() else safe_valid[:, None, None, :]
    result = F.scaled_dot_product_attention(extended_q, extended_k, extended_v,
        attn_mask=mask, dropout_p=dropout_p, scale=d**-.5)[..., :value_dim]
    if any_valid is not None:
        result = result*any_valid[:, None, None, None].to(result.dtype)
    return result


def relation_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    query_camera: Optional[torch.Tensor] = None,
    key_camera: Optional[torch.Tensor] = None,
    relation_weight: Optional[torch.Tensor] = None,
    key_valid: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    query_chunk_size: Optional[int] = 128,
    checkpoint_chunks: bool = True,
) -> torch.Tensor:
    """SDPA plus a per-head same-camera logit bias.

    q is [B,h,Nq,d], k/v are [B,h,Nk,d]. Camera labels are [B,Nq]
    and [B,Nk]; negative query labels denote summary tokens with zero bias.
    relation_weight is [h] and key_valid is bool [B,Nk]. None disables query
    splitting and inner chunk checkpointing. On CUDA with unpadded fp16/bf16
    keys it uses an equivalent Q/K feature factorization. We never skip the
    relation branch merely because its initialized weights happen to be zero:
    its gradients must be retained. An all-invalid batch item returns zero.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [B,heads,tokens,head_dim]")
    batch, heads, num_queries, _ = q.shape
    num_keys = k.shape[-2]
    if (k.shape[:2] != q.shape[:2] or v.shape[:2] != q.shape[:2]
            or k.shape[-1] != q.shape[-1] or v.shape[-2] != num_keys):
        raise ValueError("q/k/v batch, head, key count, or q/k width mismatch")
    if query_chunk_size is not None and query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    if not 0 <= dropout_p < 1:
        raise ValueError("dropout_p must be in [0,1)")
    if key_valid is not None:
        if key_valid.shape != (batch, num_keys) or key_valid.dtype != torch.bool:
            raise ValueError("key_valid must be bool [B,Nk]")
        key_valid = key_valid.to(device=q.device)
    if num_queries == 0:
        return v.new_empty((batch, heads, 0, v.shape[-1]))
    if num_keys == 0:
        # Keep an autograd connection for empty-memory callers.
        return q.sum(dim=-1, keepdim=True).expand(batch, heads, num_queries, v.shape[-1]) * 0

    if relation_weight is None:
        safe_valid, any_valid = _safe_key_valid(key_valid, num_keys)
        mask = safe_valid[:, None, None, :] if safe_valid is not None else None
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
        if any_valid is not None:
            result = result * any_valid[:, None, None, None].to(result.dtype)
        return result

    if query_camera is None or key_camera is None:
        raise ValueError("relation_weight requires query_camera and key_camera")
    if query_camera.shape != (batch, num_queries) or key_camera.shape != (batch, num_keys):
        raise ValueError("camera labels must match [B,Nq] and [B,Nk]")
    if relation_weight.shape != (heads,):
        raise ValueError("relation_weight must have shape [heads]")
    query_camera = query_camera.to(device=q.device)
    key_camera = key_camera.to(device=q.device)
    # With one camera among the valid keys, every query's relation bias is
    # constant across those keys (including negative summary queries). Softmax
    # cancels this constant exactly. Preserve the zero weight gradient while
    # avoiding a heads*Nq*Nk allocation for native-resolution mono clips.
    valid_keys = key_valid if key_valid is not None else torch.ones_like(key_camera, dtype=torch.bool)
    reference = key_camera.gather(1, valid_keys.long().argmax(-1, keepdim=True))
    if ((key_camera == reference) | ~valid_keys).all():
        safe_valid, any_valid = _safe_key_valid(key_valid, num_keys)
        mask = None if safe_valid is None or safe_valid.all() else safe_valid[:, None, None, :]
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p)
        if any_valid is not None:
            result = result*any_valid[:, None, None, None].to(result.dtype)
        return result+relation_weight.to(result.dtype).sum()*0
    if query_chunk_size is None:
        low_precision = q.dtype in (torch.float16, torch.bfloat16)
        # CUDA autocast runs Q/K LayerNorm in fp32; SDPA casts it back to
        # the selected AMP dtype. Follow that same operator-level policy.
        amp_low_precision = (q.dtype == torch.float32 and torch.is_autocast_enabled("cuda")
                             and torch.get_autocast_dtype("cuda") in (torch.float16, torch.bfloat16))
        if (q.is_cuda and (low_precision or amp_low_precision)
                and (key_valid is None or key_valid.all())):
            result = _feature_relation_attention(q, k, v, relation_weight, query_camera,
                key_camera, key_valid, dropout_p)
            if result is not None:
                return result
        return _relation_chunk(q, k, v, relation_weight, query_camera,
                               key_camera, key_valid, dropout_p)
    use_checkpoint = (
        checkpoint_chunks and torch.is_grad_enabled()
        and any(t.requires_grad for t in (q, k, v, relation_weight))
    )
    chunks = []
    for start in range(0, num_queries, query_chunk_size):
        stop = min(start + query_chunk_size, num_queries)
        args = (
            q[:, :, start:stop], k, v, relation_weight,
            query_camera[:, start:stop], key_camera, key_valid, dropout_p,
        )
        if use_checkpoint:
            chunks.append(checkpoint(_relation_chunk, *args, use_reentrant=False))
        else:
            chunks.append(_relation_chunk(*args))
    return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=-2)
