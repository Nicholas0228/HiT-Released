"""Released dense MiniT2I and HiT with joint image/caption grouping.

Baseline: the pinned released PyTorch MiniT2I (B/16 here, 128-channel patch
bottleneck, 17 two-stream blocks, no time conditioning), constructed in its
exact RNG order. Its unused time/pooled modules stay in checkpoints.

HiT: linear patch input and an explicit time token, 3/3/10/3/3 blocks. Image
patches plus valid caption positions are grouped twice by exact average linkage
with global soft pooling/readback. The 16 compact blocks use one transformation
set (the image branch) for mixed image/text groups and no RoPE. The time token
joins every compact sequence but never enters a router, pooling or budget.
Caption padding stays in outer attention and receives only its skip path.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from third_party.minit2i_torch.model import (
    DoubleStreamBlock, MMJiTB32Text2, PlainTextBlock, RMSNorm, apply_1d_rope, apply_2d_rope_flat, sincos_2d)
from model_jit import TimestepEmbedder
from hnet.grouping import VarLenGrouping
from hnet.runtime.packed_attention import packed_attention
from .config import T2IConfig
from .tokens import pack_valid_tokens


def attend(q, k, v, valid=None, reference=False):
    mask = valid[:, None, None, :] if valid is not None else None
    if reference:
        logits = q.float() @ k.float().transpose(-1, -2) / math.sqrt(q.shape[-1])
        if mask is not None:
            logits = logits.masked_fill(~mask, -torch.inf)
        return logits.softmax(-1).to(v.dtype) @ v
    return F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v, attn_mask=mask)


class JointBlock(DoubleStreamBlock):
    """Released two-stream block: separate projections, one [text; image] attention.

    Text uses 1D RoPE and image patches 2D RoPE; a time-token prefix in the
    image stream keeps identity RoPE.
    """
    time_prefix = 0

    def forward(self, img, txt):
        b, li, _ = img.shape
        lt = txt.shape[1]
        qi, ki, vi = self.img_qkv(self.img_norm1(img)).view(b, li, 3, self.num_heads, self.head_dim).unbind(2)
        qt, kt, vt = self.txt_qkv(self.txt_norm1(txt)).view(b, lt, 3, self.num_heads, self.head_dim).unbind(2)
        qi, ki = self.q_norm(qi).transpose(1, 2), self.k_norm(ki).transpose(1, 2)
        qt, kt = self.q_norm(qt).transpose(1, 2), self.k_norm(kt).transpose(1, 2)
        qt, kt = apply_1d_rope(qt), apply_1d_rope(kt)
        p = self.time_prefix
        if p:
            qi = torch.cat((qi[:, :, :p], apply_2d_rope_flat(qi[:, :, p:], self.image_grid)), 2)
            ki = torch.cat((ki[:, :, :p], apply_2d_rope_flat(ki[:, :, p:], self.image_grid)), 2)
        else:
            qi, ki = apply_2d_rope_flat(qi, self.image_grid), apply_2d_rope_flat(ki, self.image_grid)
        q, k = torch.cat((qt, qi), 2), torch.cat((kt, ki), 2)
        v = torch.cat((vt, vi), 1).transpose(1, 2)
        out = attend(q, k, v, None, self.attention_impl == 'einsum').transpose(1, 2)
        txt = txt + self.txt_proj(out[:, :lt].reshape(b, lt, -1))
        img = img + self.img_proj(out[:, lt:].reshape(b, li, -1))
        return img + self.img_mlp(self.img_norm2(img)), txt + self.txt_mlp(self.txt_norm2(txt))


class SharedCompactBlock(nn.Module):
    """One transformation set for mixed groups, taken from a block's image branch.

    A group has no prescribed modality or grid position, so there is no RoPE.
    """
    def __init__(self, original):
        super().__init__()
        self.num_heads, self.head_dim = original.num_heads, original.head_dim
        self.attention_impl = original.attention_impl
        self.norm1, self.norm2 = original.img_norm1, original.img_norm2
        self.qkv, self.proj = original.img_qkv, original.img_proj
        self.q_norm, self.k_norm = original.q_norm, original.k_norm
        self.mlp = original.img_mlp

    def forward(self, value, valid=None):
        b, n, d = value.shape
        q, k, v = self.qkv(self.norm1(value)).view(b, n, 3, self.num_heads, self.head_dim).unbind(2)
        q, k = self.q_norm(q).transpose(1, 2), self.k_norm(k).transpose(1, 2)
        out = attend(q, k, v.transpose(1, 2), valid, self.attention_impl == 'einsum').transpose(1, 2).reshape(b, n, d)
        value = value + self.proj(out)
        return value + self.mlp(self.norm2(value))

    def packed(self, value, cumulative, maximum):
        q, k, v = self.qkv(self.norm1(value)).view(-1, 3, self.num_heads, self.head_dim).unbind(1)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)
        out = packed_attention(q, k, v, cumulative, maximum).reshape_as(value)
        value = value + self.proj(out)
        return value + self.mlp(self.norm2(value))


class LinearPatchEmbed(nn.Module):
    """One learned linear map per RGB patch; no low-rank intermediate."""
    def __init__(self, patch_size, width):
        super().__init__()
        self.proj = nn.Conv2d(3, width, patch_size, stride=patch_size)
        nn.init.xavier_uniform_(self.proj.weight.flatten(1))
        nn.init.zeros_(self.proj.bias)

    def forward(self, image):
        return self.proj(image).flatten(2).transpose(1, 2)


class TextToImage(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        c = self.config = config or T2IConfig(thresholds=(.5, .5))
        grid, head = c.grid, c.width // c.heads
        released = dict(image_size=c.image_size, patch_size=c.patch_size, hidden_size=c.width,
                        t5_hidden_size=c.text_width, text_preamble_depth=c.text_depth, num_heads=c.heads,
                        head_dim=head, mlp_ratio=c.mlp_ratio, rope_style='jax',
                        attention_impl='einsum' if c.attention_backend == 'reference' else 'sdpa')
        self.router = self.router2 = None
        self._compiled = False
        if c.variant == 'baseline':
            # Adopt the released model, including its RNG order and unused time/pooled modules.
            original = MMJiTB32Text2(depth_double=c.depth, pca_channels=c.patch_bottleneck, **released)
            for name, module in original.named_children():
                setattr(self, name, module)
            self.mask_token = original.mask_token
            self.register_buffer('pos_embed', original.pos_embed, persistent=False)
            for block in self.blocks:
                block.__class__ = JointBlock
            return
        with torch.random.fork_rng(devices=[]):
            # The same seed's released dense model; these draws do not advance HiT's RNG.
            reference = MMJiTB32Text2(depth_double=c.reference_depth, pca_channels=c.reference_bottleneck, **released)
        self.img_embed = LinearPatchEmbed(c.patch_size, c.width)
        self.txt_embed = nn.Linear(c.text_width, c.width, bias=False)
        self.mask_token = nn.Parameter(torch.randn(1, 1, c.text_width) * .02)
        self.register_buffer('pos_embed', sincos_2d(c.width, grid)[None], persistent=False)
        impl = released['attention_impl']
        self.txt_blocks = nn.ModuleList([PlainTextBlock(c.width, c.heads, head, c.mlp_ratio, qk_norm=True,
                                                        rms_affine=True, attention_impl=impl)
                                         for _ in range(c.text_depth)])
        self.blocks = nn.ModuleList([JointBlock(c.width, c.heads, head, c.mlp_ratio, grid, qk_norm=True,
                                                rms_affine=True, rope_style='jax', attention_impl=impl)
                                     for _ in range(c.depth)])
        self.final_norm = RMSNorm(c.width, elementwise_affine=True)
        self.final = nn.Linear(c.width, c.patch_size * c.patch_size * 3)
        nn.init.zeros_(self.final.weight)
        nn.init.zeros_(self.final.bias)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(c.time_token_init_seed)
            self.time_embed = TimestepEmbedder(c.width)
            for layer in (self.time_embed.mlp[0], self.time_embed.mlp[2]):
                nn.init.normal_(layer.weight, std=.02)
                nn.init.zeros_(layer.bias)
            self.time_identity = nn.Parameter(torch.randn(1, 1, c.width) * .02)
        thresholds = c.thresholds or (.5, .5)  # placeholders are replaced by calibration
        options = dict(reader_temperature=c.reader_temperature, budget_temperature=c.budget_temperature,
                       workers=c.routing_workers, pad_multiple=c.pad_multiple, transfer=c.routing_transfer,
                       validation=c.routing_validation, backend=c.routing_backend)
        self.router = VarLenGrouping(c.width, threshold=thresholds[0], init_seed=c.router_init_seed, **options)
        with torch.random.fork_rng(devices=[]):
            self.skip = nn.Linear(c.width, c.width, bias=False)
            with torch.no_grad():
                self.skip.weight.copy_(torch.eye(c.width))
        with torch.random.fork_rng(devices=[]):
            self.router2 = VarLenGrouping(c.width, threshold=thresholds[1], init_seed=c.second_router_init_seed,
                                          **options)
            self.skip2 = nn.Linear(c.width, c.width, bias=False)
            with torch.no_grad():
                self.skip2.weight.copy_(torch.eye(c.width))
        for name in ('txt_embed', 'txt_blocks', 'final_norm', 'final'):
            getattr(self, name).load_state_dict(getattr(reference, name).state_dict())
        with torch.no_grad():
            self.mask_token.copy_(reference.mask_token)
        # Dense 17 = outer 6 + core 5 + outer 6 -> HiT 0..10 and 16..21; the five
        # extra core blocks keep their fresh weights. No trained weights, no averaging.
        after = c.decoder_depth + c.intermediate_depth
        self.shared_blocks = list(range(c.reference_depth - after)) + list(range(c.depth - after, c.depth))
        for old, new in enumerate(self.shared_blocks):
            self.blocks[new].load_state_dict(reference.blocks[old].state_dict())
        if c.core_extension_base_depth:
            # Core blocks beyond the base layout come from their own seed, so every module of the base
            # layout keeps the initialization it has in the base model.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(c.core_extension_seed)
                for index in range(c.core_extension_base_depth - after, c.depth - after):
                    self.blocks[index].load_state_dict(JointBlock(
                        c.width, c.heads, head, c.mlp_ratio, grid, qk_norm=True, rms_affine=True, rope_style='jax',
                        attention_impl=impl).state_dict())
        for block in self.blocks:
            block.time_prefix = 1
        for index in range(c.encoder_depth, c.depth - c.decoder_depth):
            self.blocks[index] = SharedCompactBlock(self.blocks[index])
        for router in (self.router, self.router2):
            router.tree_precision = c.tree_precision

    def compile_blocks(self):
        """Compile MLPs and router projections/readers; joint attention stays eager."""
        if self._compiled:
            return
        for block in self.txt_blocks:
            block.mlp.forward = torch.compile(block.mlp.forward, dynamic=True)
        for block in self.blocks:
            if isinstance(block, SharedCompactBlock):
                block.mlp.forward = torch.compile(block.mlp.forward, dynamic=True)
            else:
                block.img_mlp.forward = torch.compile(block.img_mlp.forward, dynamic=True)
                block.txt_mlp.forward = torch.compile(block.txt_mlp.forward, dynamic=True)
        for router in (self.router, self.router2):
            if router is not None:
                router.project = torch.compile(router.project)
                router.read = torch.compile(router.read, dynamic=True)
        self._compiled = True

    def encode(self, image, context, mask, time=None):
        """Caption path, patch input, optional time token and the outer encoder blocks."""
        c = self.config
        if image.shape[1:] != (3, c.image_size, c.image_size) or context.shape[1:] != (c.text_length, c.text_width):
            raise ValueError('Unexpected image or caption-context shape')
        if mask.shape != context.shape[:2]:
            raise ValueError('The caption mask needs one entry per text position')
        context = torch.where(mask.bool()[..., None], context, self.mask_token.to(context.dtype))
        img = self.img_embed(image) + self.pos_embed
        txt = self.txt_embed(context)
        for block in self.txt_blocks:
            txt = block(txt)
        if c.variant == 'hnet':
            if time is None or time.shape != (image.shape[0],):
                raise ValueError('HiT needs the physical timestep for every image')
            # Raw physical t, as in the ImageNet HiT: no multiplication by 1,000.
            img = torch.cat((self.time_embed(time)[:, None] + self.time_identity, img), 1)
        for block in self.blocks[:c.encoder_depth]:
            img, txt = block(img, txt)
        return img, txt

    def grouping_input(self, img, txt, mask):
        """Split the time token; pack image patches plus valid caption positions."""
        n = self.config.image_tokens
        time_state, full = img[:, :1], torch.cat((img[:, 1:], txt), 1)
        eligible = torch.cat((torch.ones(full.shape[0], n, dtype=torch.bool, device=full.device), mask.bool()), 1)
        packed, layout = pack_valid_tokens(full, eligible)
        return time_state, full, packed, layout

    def compact(self, value, time_state, valid, lengths, start, stop):
        """Compact blocks over [time; groups], excluding padded group slots."""
        c = self.config
        value = torch.cat((time_state, value), 1)
        valid = torch.cat((torch.ones_like(valid[:, :1]), valid), 1)
        lengths = tuple(n + 1 for n in lengths)
        if c.compact_batching == 'packed':
            indices = valid.flatten().nonzero().flatten()
            counts = torch.tensor(lengths, device=value.device, dtype=torch.int32)
            cumulative = F.pad(counts.cumsum(0, dtype=torch.int32), (1, 0))
            packed = value.flatten(0, 1).index_select(0, indices)
            for block in self.blocks[start:stop]:
                packed = block.packed(packed, cumulative, max(lengths))
            value = value.new_zeros(value.shape[0] * value.shape[1], c.width).index_copy(
                0, indices, packed).reshape_as(value)
        else:
            for block in self.blocks[start:stop]:
                value = block(value, valid)
        return value[:, 1:], value[:, :1]

    def grouped(self, img, txt, mask, need_budget=True, fixed_structure=None):
        c = self.config
        e, m = c.encoder_depth, c.intermediate_depth
        inner_start, inner_stop, outer_stop = e + m, c.depth - c.decoder_depth - m, c.depth - c.decoder_depth
        time_state, full, packed, layout = self.grouping_input(img, txt, mask)
        fixed1, fixed2 = (None, None) if fixed_structure is None else fixed_structure
        one, a1, v1, s1, soft1 = self.router(packed, layout.valid, layout.counts, layout.lengths,
                                             need_budget, fixed1)
        one, time_state = self.compact(one, time_state, v1, s1.lengths, e, inner_start)
        two, a2, v2, s2, soft2 = self.router2(one, v1, s1.counts, s1.lengths, need_budget, fixed2)
        two, time_state = self.compact(two, time_state, v2, s2.lengths, inner_start, inner_stop)
        restored = (self.router2.reconstruct(a2, two) + self.skip2(one)) * v1[..., None]
        restored, time_state = self.compact(restored, time_state, v1, s1.lengths, inner_stop, outer_stop)
        restored = layout.restore(self.router.reconstruct(a1, restored)) + self.skip(full)
        n = c.image_tokens
        aux = {'hard_counts': s1.counts, 'soft_counts': soft1, 'population1': layout.counts,
               'hard_counts2': s2.counts, 'soft_counts2': soft2, 'population2': s1.counts,
               'stats': self.router.last_statistics, 'stats2': self.router2.last_statistics,
               'structure': s1, 'structure2': s2, 'assignment1': a1.detach(), 'assignment2': a2.detach(),
               'input_valid': layout.valid, 'input_positions': layout.positions}
        return torch.cat((time_state, restored[:, :n]), 1), restored[:, n:], aux

    def forward(self, image, context, mask, time=None, need_budget=True, fixed_structure=None):
        """Predict the clean image (x-prediction) for noisy images and T5 caption features."""
        c = self.config
        img, txt = self.encode(image, context, mask, time)
        aux = {}
        if c.variant == 'hnet':
            img, txt, aux = self.grouped(img, txt, mask, need_budget, fixed_structure)
            outer = self.blocks[c.depth - c.decoder_depth:]
        else:
            outer = self.blocks
        for block in outer:
            img, txt = block(img, txt)
        if c.variant == 'hnet':
            img = img[:, 1:]
        patches = self.final(self.final_norm(img))
        b, g, p = patches.shape[0], c.grid, c.patch_size
        output = patches.reshape(b, g, g, p, p, 3).permute(0, 5, 1, 3, 2, 4).reshape(b, 3, c.image_size, c.image_size)
        return output.float(), aux
