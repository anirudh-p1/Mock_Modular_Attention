"""
Mini Transformer using MockModularAttention blocks.

The Transformer follows a standard pre-norm design:

    x = x + Dropout(Attention(LayerNorm(x)))
    x = x + Dropout(FeedForward(LayerNorm(x)))

MockModularAttention is used for every self-attention layer.
The module is designed as a language-model head that predicts next tokens,
but the blocks are generic and can be composed freely.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MockModularAttention
from .qseries import KernelName


class MockModularTransformerBlock(nn.Module):
    """Single Transformer block using MockModularAttention.

    Parameters
    ----------
    embed_dim : int
        Model embedding dimension.
    num_heads : int
        Number of attention heads.
    ff_dim : int
        Hidden dimension of the feed-forward sublayer.
    kernel : KernelName
        Mock theta kernel passed through to ``MockModularAttention``.
    init_temperature : float
        Initial learnable temperature for the q-series kernel.
    dropout : float
        Dropout probability (attention weights and residual connections).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        kernel: KernelName = "mock_theta_f",
        init_temperature: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Self-attention sublayer (batch_first=True for convenience)
        self.attn = MockModularAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            kernel=kernel,
            init_temperature=init_temperature,
        )

        # Feed-forward sublayer
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pre-norm Transformer block forward pass.

        Parameters
        ----------
        x : Tensor
            Shape ``(batch, seq_len, embed_dim)``.
        attn_mask : Tensor, optional
            Causal or arbitrary mask, shape ``(seq_len, seq_len)`` (bool or
            float), forwarded to ``MockModularAttention``.
        key_padding_mask : Tensor, optional
            Padding mask of shape ``(batch, seq_len)`` (bool).

        Returns
        -------
        Tensor
            Shape ``(batch, seq_len, embed_dim)``.
        """
        # Self-attention with pre-normalisation
        normed = self.norm1(x)
        attn_out, _ = self.attn(
            normed, normed, normed,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop(attn_out)

        # Feed-forward with pre-normalisation
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class MockModularTransformer(nn.Module):
    """Small causal language model built on Mock Modular Attention.

    Architecture
    ------------
    token_emb + pos_emb  →  N × MockModularTransformerBlock
    →  LayerNorm  →  Linear(vocab_size)

    Parameters
    ----------
    vocab_size : int
        Number of token types.
    seq_len : int
        Maximum sequence length (sets the positional embedding table size).
    embed_dim : int
        Model width.
    num_heads : int
        Attention heads per block.
    num_layers : int
        Number of Transformer blocks.
    ff_dim : int
        Hidden dimension of each feed-forward sublayer.
    kernel : KernelName
        Mock theta kernel used in every attention layer.
    init_temperature : float
        Initial temperature for every q-series kernel.
    dropout : float
        Dropout probability throughout the model.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 256,
        kernel: KernelName = "mock_theta_f",
        init_temperature: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.seq_len = seq_len

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(seq_len, embed_dim)
        self.emb_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                MockModularTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    kernel=kernel,
                    init_temperature=init_temperature,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(
        self,
        token_ids: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        token_ids : LongTensor
            Shape ``(batch, seq_len)``.
        key_padding_mask : BoolTensor, optional
            Shape ``(batch, seq_len)``.  ``True`` marks padding positions.

        Returns
        -------
        Tensor
            Logits of shape ``(batch, seq_len, vocab_size)``.
        """
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0)
        x = self.emb_drop(self.token_emb(token_ids) + self.pos_emb(pos))

        # Causal mask: shape (L, L), True = future position (mask out)
        causal = torch.triu(
            torch.ones(L, L, device=token_ids.device, dtype=torch.bool),
            diagonal=1,
        )

        for block in self.blocks:
            x = block(x, attn_mask=causal, key_padding_mask=key_padding_mask)

        return self.head(self.norm(x))

    @staticmethod
    def causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Boolean upper-triangular causal mask of shape ``(seq_len, seq_len)``."""
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
