"""
Tests for MockModularTransformer and MockModularTransformerBlock.

Validates:
- Output shapes for all components
- Causal mask prevents future look-ahead
- Gradient flow through all layers
- Loss decreases during a short training loop
- All four kernels can be used in a full Transformer
"""

import pytest
import torch
import torch.nn.functional as F
from mock_modular_attention.transformer import (
    MockModularTransformerBlock,
    MockModularTransformer,
)


# ---------------------------------------------------------------------------
# MockModularTransformerBlock
# ---------------------------------------------------------------------------

class TestTransformerBlock:
    def test_output_shape(self):
        block = MockModularTransformerBlock(embed_dim=32, num_heads=4, ff_dim=64)
        x = torch.randn(2, 8, 32)
        out = block(x)
        assert out.shape == x.shape

    def test_with_causal_mask(self):
        L = 6
        block = MockModularTransformerBlock(embed_dim=16, num_heads=2, ff_dim=32)
        x = torch.randn(2, L, 16)
        mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
        out = block(x, attn_mask=mask)
        assert out.shape == x.shape

    def test_with_key_padding_mask(self):
        block = MockModularTransformerBlock(embed_dim=16, num_heads=2, ff_dim=32)
        x = torch.randn(2, 6, 16)
        pad = torch.zeros(2, 6, dtype=torch.bool)
        pad[0, 4:] = True
        out = block(x, key_padding_mask=pad)
        assert out.shape == x.shape

    def test_gradients_flow(self):
        block = MockModularTransformerBlock(embed_dim=16, num_heads=2, ff_dim=32)
        x = torch.randn(2, 5, 16, requires_grad=True)
        out = block(x)
        out.sum().backward()
        assert x.grad is not None


# ---------------------------------------------------------------------------
# MockModularTransformer
# ---------------------------------------------------------------------------

class TestTransformer:
    def test_output_shape(self):
        model = MockModularTransformer(vocab_size=20, seq_len=8,
                                       embed_dim=32, num_heads=2)
        tokens = torch.randint(0, 20, (2, 8))
        logits = model(tokens)
        assert logits.shape == (2, 8, 20)

    def test_causal_mask_static(self):
        L = 6
        mask = MockModularTransformer.causal_mask(L, device=torch.device("cpu"))
        assert mask.shape == (L, L)
        assert mask.dtype == torch.bool
        # Upper triangle (above diagonal) must be True (masked)
        assert mask[0, 1]
        # Diagonal and below must be False
        assert not mask[0, 0]
        assert not mask[2, 1]

    def test_with_padding_mask(self):
        model = MockModularTransformer(vocab_size=10, seq_len=6,
                                       embed_dim=16, num_heads=2)
        tokens = torch.randint(0, 10, (2, 6))
        pad = torch.zeros(2, 6, dtype=torch.bool)
        pad[1, 4:] = True
        logits = model(tokens, key_padding_mask=pad)
        assert logits.shape == (2, 6, 10)

    def test_gradients_flow(self):
        model = MockModularTransformer(vocab_size=10, seq_len=5,
                                       embed_dim=16, num_heads=2)
        tokens = torch.randint(0, 10, (2, 5))
        logits = model(tokens)
        loss = F.cross_entropy(logits.view(-1, 10), tokens.view(-1))
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No grad for {name}"

    def test_loss_decreases_over_training(self):
        torch.manual_seed(0)
        model = MockModularTransformer(
            vocab_size=10, seq_len=6, embed_dim=32,
            num_heads=2, num_layers=1, ff_dim=64,
        )
        optim = torch.optim.Adam(model.parameters(), lr=5e-3)

        first, last = [], []
        for step in range(150):
            tokens = torch.randint(0, 10, (8, 6))
            logits = model(tokens)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, 10),
                tokens[:, 1:].reshape(-1),
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            if step < 5:
                first.append(loss.item())
            if step >= 145:
                last.append(loss.item())

        assert sum(last) / len(last) < sum(first) / len(first), (
            "Training loss did not decrease"
        )


# ---------------------------------------------------------------------------
# All kernels work in a full Transformer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kernel",
    ["mock_theta_f", "mock_theta_omega", "mock_theta_phi", "q_series"],
)
def test_transformer_all_kernels(kernel):
    model = MockModularTransformer(
        vocab_size=10, seq_len=6, embed_dim=16, num_heads=2,
        num_layers=1, ff_dim=32, kernel=kernel,
    )
    tokens = torch.randint(0, 10, (2, 6))
    logits = model(tokens)
    assert logits.shape == (2, 6, 10)
    loss = F.cross_entropy(logits.view(-1, 10), tokens.view(-1))
    loss.backward()   # must not raise
