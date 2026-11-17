"""Stable selection of grouping-eligible tokens and restoration of their slots."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class TokenLayout:
    positions: torch.Tensor
    valid: torch.Tensor
    counts: torch.Tensor
    lengths: tuple
    original_length: int

    def restore(self, reconstruction):
        # Selected positions are unique. Invalid packing rows contribute exactly
        # zero; the caller adds the unchanged full-resolution skip.
        source = reconstruction * self.valid[..., None]
        indices = self.positions[..., None].expand_as(source)
        return source.new_zeros(source.shape[0], self.original_length, source.shape[-1]).scatter(1, indices, source)


def pack_valid_tokens(features, eligible):
    """Move eligible tokens to a prefix, keeping image patches first and text order."""
    if eligible.shape != features.shape[:2]:
        raise ValueError('Eligibility must have one entry per original token')
    eligible = eligible.bool()
    counts = eligible.sum(1)
    lengths = tuple(counts.detach().cpu().tolist())
    if not lengths or min(lengths) < 1:
        raise ValueError('Each example must retain at least one eligible token')
    maximum = max(lengths)
    positions = torch.argsort(eligible.to(torch.int8), dim=1, descending=True, stable=True)[:, :maximum]
    valid = torch.arange(maximum, device=features.device)[None, :] < counts[:, None]
    packed = features.gather(1, positions[..., None].expand(-1, -1, features.shape[-1])) * valid[..., None]
    return packed, TokenLayout(positions, valid, counts, lengths, features.shape[1])
