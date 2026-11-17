"""JiT blocks with device-safe reference and optional packed execution."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from model_jit import SwiGLUFFN
from util.model_util import RMSNorm
from .runtime.packed_attention import packed_attention


class SpatialRoPE(nn.Module):
    """Original JiT frequency/layout arithmetic, with device-safe buffers."""
    def __init__(self, head_dim, side, prefix=0):
        super().__init__()
        half = head_dim // 2
        frequencies = 1 / (10000 ** (torch.arange(0, half, 2).float() / half))
        positions = torch.arange(side) / side * side
        phases = torch.einsum('n,f->nf', positions, frequencies).repeat_interleave(2, -1)
        grid = torch.cat((phases[:, None, :].expand(side, side, -1),
                          phases[None, :, :].expand(side, side, -1)), -1).reshape(side * side, head_dim)
        self.register_buffer('cos', torch.cat((torch.ones(prefix, head_dim), grid.cos())), persistent=False)
        self.register_buffer('sin', torch.cat((torch.zeros(prefix, head_dim), grid.sin())), persistent=False)

    def forward(self, x):
        paired = x.reshape(*x.shape[:-1], -1, 2)
        rotated = torch.stack((-paired[..., 1], paired[..., 0]), -1).flatten(-2)
        return x * self.cos + rotated * self.sin


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class Attention(nn.Module):
    def __init__(self, width, heads, backend='reference', proj_drop=0.0):
        super().__init__()
        self._fast_runtime = False
        self.heads, self.backend = heads, backend
        self.qkv = nn.Linear(width, 3 * width)
        self.q_norm = RMSNorm(width // heads)
        self.k_norm = RMSNorm(width // heads)
        self.proj = nn.Linear(width, width)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope=None, valid=None):
        b, n, d = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = rope(q), rope(k)
        mask = valid[:, None, None, :] if valid is not None else None
        if self.backend == 'reference':
            with torch.autocast(x.device.type, enabled=False):
                logits = q.float() @ k.float().transpose(-1, -2) / math.sqrt(d // self.heads)
                if mask is not None:
                    logits = logits.masked_fill(~mask, -torch.inf)
                weights = logits.softmax(-1)
            # Under BF16 autocast this matches upstream's second matrix product.
            output = weights @ v
        elif self.backend == 'sdpa':
            q, k = q.to(v.dtype), k.to(v.dtype)
            scale = 1 / math.sqrt(d // self.heads)
            if self._fast_runtime:
                from .runtime.attention import dense_attention
                if mask is not None:
                    raise ValueError('The Hopper profile uses unmasked dense or packed attention')
                output = dense_attention(q, k, v, scale=scale)
            else:
                output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
            output = output[..., :d // self.heads]
        else:
            raise ValueError(self.backend)
        return self.proj_drop(self.proj(output.transpose(1, 2).reshape(b, n, d)))

    def forward_packed(self, x, cumulative, maximum, rope=None):
        if self.backend != 'sdpa':
            raise ValueError('Packed execution requires the sdpa attention backend')
        tokens, width = x.shape
        q, k, v = self.qkv(x).reshape(tokens, 3, self.heads, width // self.heads).unbind(1)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            raise ValueError('Compact RoPE is not part of this reference')
        q, k = q.to(v.dtype), k.to(v.dtype)
        scale = 1 / math.sqrt(width // self.heads)
        if self._fast_runtime:
            from .runtime.attention import packed_attention as hopper_attention
            output = hopper_attention(q, k, v, cumulative, maximum, scale=scale)
        else:
            output = packed_attention(q, k, v, cumulative, maximum, scale=scale)
        return self.proj_drop(self.proj(output[..., :width // self.heads].reshape(tokens, width)))


class Block(nn.Module):
    def __init__(self, width, heads, backend, conditioning_width=None, proj_drop=0.0):
        super().__init__()
        self._fast_runtime = False
        self.norm1, self.norm2 = RMSNorm(width), RMSNorm(width)
        self.attn = Attention(width, heads, backend, proj_drop=proj_drop)
        self.mlp = SwiGLUFFN(width, width * 4, drop=proj_drop)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(conditioning_width or width, width * 6))
        self.adaln = True

    def forward(self, x, conditioning, rope=None, valid=None):
        if not self.adaln:
            x = x + self.attn(self.norm1(x), rope, valid)
            return x + self.mlp(self.norm2(x))
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN_modulation(conditioning).chunk(6, -1)
        x = x + gate_a[:, None] * self.attn(modulate(self.norm1(x), shift_a, scale_a), rope, valid)
        return x + gate_m[:, None] * self.mlp(modulate(self.norm2(x), shift_m, scale_m))

    def forward_packed(self, x, conditioning, sequence_ids, cumulative, maximum, rope=None):
        if not self.adaln:
            x = x + self.attn.forward_packed(self.norm1(x), cumulative, maximum, rope)
            return x + self.mlp(self.norm2(x))
        if self._fast_runtime:
            from .runtime.conditioning import expand
            modulation = expand(self.adaLN_modulation(conditioning), sequence_ids, cumulative, maximum)
        else:
            modulation = self.adaLN_modulation(conditioning).index_select(0, sequence_ids)
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = modulation.chunk(6, -1)
        value = self.norm1(x) * (1 + scale_a) + shift_a
        x = x + gate_a * self.attn.forward_packed(value, cumulative, maximum, rope)
        return x + gate_m * self.mlp(self.norm2(x) * (1 + scale_m) + shift_m)


class FinalLayer(nn.Module):
    def __init__(self, width, patch_size, conditioning_width=None):
        super().__init__()
        self.norm_final = RMSNorm(width)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(conditioning_width or width, 2 * width))
        self.linear = nn.Linear(width, patch_size * patch_size * 3)
        self.adaln = True

    def forward(self, x, conditioning):
        if not self.adaln:
            return self.linear(self.norm_final(x))
        shift, scale = self.adaLN_modulation(conditioning).chunk(2, -1)
        return self.linear(modulate(self.norm_final(x), shift, scale))
