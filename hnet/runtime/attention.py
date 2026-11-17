"""Pinned Hopper FA3 head64; native FlashAttention for JiT-H head80.

The custom-op boundary makes the upstream CUDA extension visible to AOTAutograd.
Attention-probability dropout remains zero. Projection dropout lives in the model.
No attention mask approximation, FP8, or sequence mixing is introduced.
"""
from typing import Optional, Tuple
import math
import torch
from flash_attn_interface import _flash_attn_forward, _flash_attn_backward


@torch.library.custom_op('hnet_fast_flash3::forward', mutates_args=())
def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            cumulative: Optional[torch.Tensor], maximum: int, scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    assert q.dtype == k.dtype == v.dtype == torch.bfloat16 and q.shape[-1] == 64
    result = _flash_attn_forward(q, k, v, None, None, None,
        cumulative, cumulative, None, None, None, maximum, maximum,
        None, None, None, None, None, None, None, None,
        scale, False, num_splits=1, pack_gqa=False)
    return result[0].contiguous(), result[1].contiguous()


@forward.register_fake
def _forward_fake(q, k, v, cumulative, maximum, scale):
    lse_shape = (q.shape[1], q.shape[0]) if cumulative is not None else (q.shape[0], q.shape[2], q.shape[1])
    return torch.empty_like(q, memory_format=torch.contiguous_format), q.new_empty(lse_shape, dtype=torch.float32)


@torch.library.custom_op('hnet_fast_flash3::backward', mutates_args=())
def backward(grad: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
             output: torch.Tensor, lse: torch.Tensor, cumulative: Optional[torch.Tensor],
             maximum: int, scale: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dq, dk, dv = [torch.empty_like(x, memory_format=torch.contiguous_format) for x in (q, k, v)]
    _flash_attn_backward(grad.contiguous(), q, k, v, output, lse,
        cumulative, cumulative, None, None, maximum, maximum,
        dq, dk, dv, scale, False, deterministic=False)
    return dq, dk, dv


@backward.register_fake
def _backward_fake(grad, q, k, v, output, lse, cumulative, maximum, scale):
    return tuple(torch.empty_like(x, memory_format=torch.contiguous_format) for x in (q, k, v))


def _setup(ctx, inputs, output):
    q, k, v, cumulative, maximum, scale = inputs
    ctx.save_for_backward(q, k, v, output[0], output[1], cumulative)
    ctx.maximum = maximum
    ctx.scale = scale
    ctx.mark_non_differentiable(output[1])


def _autograd(ctx, grad, _):
    q, k, v, output, lse, cumulative = ctx.saved_tensors
    return (*backward(grad, q, k, v, output, lse, cumulative, ctx.maximum, ctx.scale), None, None, None)


forward.register_autograd(_autograd, setup_context=_setup)


def packed_attention(q, k, v, cumulative, maximum, scale=None):
    if q.shape[-1] == 80:
        # The pinned FA3 binary was built with only head64 kernels. Keep the
        # JiT-H head80 on the existing native kernel.
        from .packed_attention import packed_attention as native_attention
        return native_attention(q, k, v, cumulative, maximum, scale=scale)
    return forward(q, k, v, cumulative, maximum, scale if scale is not None else 1 / math.sqrt(q.shape[-1]))[0]


def dense_attention(q, k, v, scale=None):
    # Input/output use native SDPA [B,H,N,D], FA3 uses [B,N,H,D].
    if q.shape[-1] == 80:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=scale)
    result = forward(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), None, q.shape[2],
                     scale if scale is not None else 1 / math.sqrt(q.shape[-1]))[0]
    return result.transpose(1, 2)
