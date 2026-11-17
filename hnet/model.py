"""One implementation for the released dense JiT and the five-stage HNet.

D18 is 3/3/6/3/3. L and H only change dimensions/depths in their model configs.
Conditioning tokens bypass both image routers and both image skip connections.
"""
from dataclasses import asdict
import math
import numpy as np
import torch
from torch import nn
from model_jit import JiT, LabelEmbedder, TimestepEmbedder
from util.model_util import get_2d_sincos_pos_embed
from .config import ModelConfig
from .grouping import SoftGrouping, VarLenGrouping
from .layers import Block, FinalLayer, SpatialRoPE


class CleanJiT(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        initialization_rng = torch.get_rng_state()
        self.config = c = config or ModelConfig()
        self._fast_runtime = False
        side = c.image_size // c.patch_size
        self.t_embedder = TimestepEmbedder(c.width)
        self.y_embedder = LabelEmbedder(c.classes, c.width)
        self.x_embedder = nn.Conv2d(3, c.width, c.patch_size, stride=c.patch_size)
        self.register_buffer('pos_embed', torch.from_numpy(get_2d_sincos_pos_embed(c.width, side)).float()[None].to(self.x_embedder.weight.device))
        self.rope = SpatialRoPE(c.width // c.heads, side)
        self.blocks = nn.ModuleList([
            Block(c.width, c.heads, c.attention_backend, c.width,
                  proj_drop=c.proj_dropout if c.depth // 4 <= i < (c.depth // 4) * 3 else 0.)
            for i in range(c.depth)])
        self.final_layer = FinalLayer(c.width, c.patch_size, c.width)
        self.initialize_common()
        self.router = self.router2 = None
        if c.variant == 'hnet':
            options = dict(reader_temperature=c.reader_temperature,
                           budget_temperature=c.budget_temperature, workers=c.routing_workers,
                           pad_multiple=c.pad_multiple, transfer=c.routing_transfer,
                           validation=c.routing_validation, backend=c.routing_backend)
            self.router = SoftGrouping(c.width, threshold=c.threshold,
                                       init_seed=c.router_init_seed, **options)
            self.skip = nn.Linear(c.width, c.width, bias=False)
            nn.init.eye_(self.skip.weight)
            self.router2 = VarLenGrouping(c.width, threshold=c.second_threshold,
                                         init_seed=c.second_router_init_seed, **options)
            self.skip2 = nn.Linear(c.width, c.width, bias=False)
            nn.init.eye_(self.skip2.weight)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(c.context_init_seed)
            slots = torch.empty(1, max(32, c.in_context_len), c.width)
            nn.init.normal_(slots, std=.02)
        self.in_context_posemb = nn.Parameter(slots[:, :c.in_context_len].clone(memory_format=torch.contiguous_format))
        self.context_count = c.in_context_len
        if c.variant == 'hnet':
            # Legacy initialization drew these modules before deleting them.
            # Retaining that order preserves every common weight and the RNG.
            for block in self.blocks:
                del block.adaLN_modulation
                block.adaln = False
            del self.final_layer.adaLN_modulation
            self.final_layer.adaln = False
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(c.time_token_init_seed)
                vector = torch.empty(1, 1, c.width)
                nn.init.normal_(vector, std=.02)
            self.time_token_pos = nn.Parameter(vector)
            self.context_count += 1
        self.rope_context = SpatialRoPE(c.width // c.heads, side, self.context_count)
        self.last_aux = {}
        if c.variant == 'baseline':
            self.initialize_released_dense(initialization_rng)

    def initialize_common(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.x_embedder.weight.flatten(1))
        nn.init.zeros_(self.x_embedder.bias)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=.02)
        for block in self.blocks:
            nn.init.zeros_(block.adaLN_modulation[-1].weight)
            nn.init.zeros_(block.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def initialize_released_dense(self, initialization_rng):
        """Use upstream initialization, including its RNG consumption, for any size."""
        c = self.config
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(initialization_rng)
            reference = JiT(input_size=c.image_size, patch_size=c.patch_size,
                            hidden_size=c.width, depth=c.depth, num_heads=c.heads,
                            num_classes=c.classes, bottleneck_dim=c.patch_bottleneck,
                            in_context_len=c.in_context_len, in_context_start=c.in_context_start,
                            proj_drop=c.proj_dropout)
            reference_rng = torch.get_rng_state()
            # Reuse the patch layers directly; all other parameter names already match.
            self.x_embedder = nn.Sequential(reference.x_embedder.proj1, reference.x_embedder.proj2)
            parameters = dict(reference.named_parameters())
            copied = set()
            with torch.no_grad():
                for key, value in self.named_parameters():
                    source = key.replace('x_embedder.0.', 'x_embedder.proj1.').replace('x_embedder.1.', 'x_embedder.proj2.')
                    value.copy_(parameters[source]); copied.add(source)
                assert copied == {k for k, v in parameters.items() if v.requires_grad}
                self.pos_embed.copy_(reference.pos_embed)
                for target, source in ((self.rope, reference.feat_rope), (self.rope_context, reference.feat_rope_incontext)):
                    target.cos.copy_(source.freqs_cos)
                    target.sin.copy_(source.freqs_sin)
        torch.set_rng_state(reference_rng)

    def _compact_stage(self, compact, conditioning, valid, lengths, start, stop):
        if self.config.compact_batching == 'padded':
            for block in self.blocks[start:stop]:
                compact = block(compact, conditioning, None, valid)
            return compact
        rows = np.repeat(np.arange(len(lengths), dtype=np.int64), lengths)
        cumulative_np = np.concatenate(([0], np.cumsum(lengths))).astype(np.int32)
        sequence_ids = torch.as_tensor(rows, device=compact.device)
        cumulative = torch.as_tensor(cumulative_np, device=compact.device)
        if self._fast_runtime:
            value = self._fast_pack(compact, sequence_ids, cumulative)
        else:
            positions = np.arange(len(rows)) - np.repeat(cumulative_np[:-1], lengths)
            ids = torch.as_tensor(rows * compact.shape[1] + positions, device=compact.device)
            value = compact.flatten(0, 1).index_select(0, ids)
        maximum = compact.shape[1] if self._fast_runtime else max(lengths)
        for block in self.blocks[start:stop]:
            value = block.forward_packed(value, conditioning, sequence_ids, cumulative, maximum)
        if self._fast_runtime:
            return self._fast_unpack(value, sequence_ids, cumulative, compact.shape[1])
        return torch.zeros_like(compact).flatten(0, 1).index_copy(0, ids, value).reshape_as(compact)

    def _with_context(self, value, valid, lengths, context):
        ones = torch.ones(context.shape[:2], device=valid.device, dtype=torch.bool)
        return (torch.cat((context, value), dim=1), torch.cat((ones, valid), dim=1),
                tuple(length + context.shape[1] for length in lengths))

    def _two_stage(self, x, conditioning, context, return_aux):
        c, ctx = self.config, self.context_count
        e, m, d = c.encoder_depth, c.intermediate_depth, c.decoder_depth
        inner_start, inner_stop = e + m, c.depth - d - m
        x = torch.cat((context, x), dim=1)
        for block in self.blocks[:e]:
            x = block(x, conditioning, self.rope_context)
        context, x = x[:, :ctx], x[:, ctx:]

        residual1 = self.skip(x)
        compact1, assignment1, valid1, structure1, soft1 = self.router(x, need_budget=return_aux)
        stats1 = dict(self.router.last_statistics)
        value, mask, lengths = self._with_context(compact1, valid1, structure1.lengths, context)
        value = self._compact_stage(value, conditioning, mask, lengths, e, inner_start)
        context, h1 = value[:, :ctx], value[:, ctx:]

        residual2 = self.skip2(h1)
        compact2, assignment2, valid2, structure2, soft2 = self.router2(
            h1, valid1, structure1.counts, structure1.lengths, need_budget=return_aux)
        value, mask, lengths = self._with_context(compact2, valid2, structure2.lengths, context)
        value = self._compact_stage(value, conditioning, mask, lengths, inner_start, inner_stop)
        inner, context = value[:, ctx:], value[:, :ctx]

        reconstructed2 = self.router2.reconstruct(assignment2, inner)
        h1 = residual2 + reconstructed2
        value, mask, lengths = self._with_context(h1, valid1, structure1.lengths, context)
        value = self._compact_stage(value, conditioning, mask, lengths, inner_stop, c.depth - d)
        intermediate, context = value[:, ctx:], value[:, :ctx]
        reconstructed1 = self.router.reconstruct(assignment1, intermediate)
        x = torch.cat((context, residual1 + reconstructed1), dim=1)
        for block in self.blocks[c.depth - d:]:
            x = block(x, conditioning, self.rope_context)
        aux = dict(stats1)
        aux.update(hard_counts=structure1.counts, soft_counts=soft1, token_count=assignment1.shape[1],
                   hard_counts2=structure2.counts, soft_counts2=soft2, stage2=dict(self.router2.last_statistics),
                   residual_rms=_diagnostic_rms(residual1, self._fast_runtime),
                   reconstructed_rms=_diagnostic_rms(reconstructed1, self._fast_runtime),
                   residual2_rms=_diagnostic_rms(residual2, self._fast_runtime),
                   reconstructed2_rms=_diagnostic_rms(reconstructed2, self._fast_runtime))
        return x[:, ctx:], aux

    def forward(self, image, time, labels, return_aux=False):
        c = self.config
        class_embedding, time_embedding = self.y_embedder(labels), self.t_embedder(time)
        conditioning = time_embedding + class_embedding
        context = class_embedding[:, None] + self.in_context_posemb
        x = self.x_embedder(image).flatten(2).transpose(1, 2)
        aux = {}
        if c.variant == 'hnet':
            context = torch.cat((time_embedding[:, None] + self.time_token_pos, context), dim=1)
            x = x + self.pos_embed
            x, aux = self._two_stage(x, conditioning, context, return_aux)
        else:
            # In-place addition preserves upstream BF16 patch-embedding arithmetic.
            x += self.pos_embed
            for index, block in enumerate(self.blocks):
                if index == c.in_context_start:
                    x = torch.cat((context, x), dim=1)
                x = block(x, conditioning, self.rope_context if index >= c.in_context_start else self.rope)
            x = x[:, self.context_count:]
        patches = self.final_layer(x, conditioning)
        b, n, _ = patches.shape
        side, p = math.isqrt(n), c.patch_size
        output = patches.reshape(b, side, side, p, p, 3).permute(0, 5, 1, 3, 2, 4).reshape(b, 3, c.image_size, c.image_size)
        self.last_aux = aux
        return (output, aux) if return_aux else output

    def compile_blocks(self, *, runtime='native', activation_memory_budget=None):
        if runtime == 'hopper':
            from .runtime.hopper import enable
            enable(self, activation_memory_budget)
        elif runtime != 'native' or activation_memory_budget is not None:
            raise ValueError('Unknown runtime or activation budget without Hopper')
        for block in self.blocks:
            block.forward = torch.compile(block.forward, dynamic=True)
        if self.config.compact_batching == 'packed' and self.router is not None:
            for block in self.blocks[self.config.encoder_depth:self.config.depth - self.config.decoder_depth]:
                block.forward_packed = torch.compile(block.forward_packed, dynamic=True)
        self.final_layer.forward = torch.compile(self.final_layer.forward, dynamic=True)
        for router in (self.router, self.router2):
            if router is not None:
                for name in ('project', 'read', 'reconstruct'):
                    setattr(router, name, torch.compile(getattr(router, name), dynamic=True))
        if runtime == 'hopper':
            from .runtime.sampling import enable_sampling
            enable_sampling(self)

    def configuration(self):
        return asdict(self.config)


@torch.compile(dynamic=True)
def _compiled_diagnostic_rms(value):
    return value.detach().float().square().mean().sqrt()


def _diagnostic_rms(value, fast):
    return _compiled_diagnostic_rms(value) if fast else value.detach().float().square().mean().sqrt()
