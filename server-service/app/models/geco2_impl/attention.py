import math
from typing import Tuple

import torch
from torch import Tensor, nn


class Attention(nn.Module):
    """Multi-head attention with optional projection downsampling."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert (
            self.internal_dim % num_heads == 0
        ), "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out


class PrototypeAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int) -> None:
        super().__init__()
        self.cross_attention = Attention(embedding_dim, num_heads)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(
        self, image_f: Tensor, prototypes: Tensor
    ) -> Tuple[Tensor, Tensor]:
        image_f = image_f + self.cross_attention(
            q=image_f, k=prototypes, v=prototypes
        )
        image_f = self.norm(image_f)
        return image_f
