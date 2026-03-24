"""
Tests for MockModularAttention.

Validates the drop-in compatibility with nn.MultiheadAttention:
- Identical __init__ signature (all standard params accepted)
- Identical forward signature and return type (tensor, optional tensor)
- Correct output shapes for batch_first / non-batch_first
- Masking: key_padding_mask, attn_mask (bool & float), is_causal
- Gradient flow through the q-series kernel
- Temperature parameter is learnable and positive
- All four kernel variants produce valid attention distributions
- add_bias_kv and add_zero_attn extensions
"""

import math
import pytest
import torch
import torch.nn as nn
from mock_modular_attention.attention import MockModularAttention


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mma(embed_dim=32, num_heads=4, **kw) -> MockModularAttention:
    return MockModularAttention(embed_dim=embed_dim, num_heads=num_heads, **kw)


def seq_batch_inputs(L=6, B=2, D=32):
    """(L, B, D) layout — matches nn.MultiheadAttention default."""
    q = torch.randn(L, B, D)
    return q, q.clone(), q.clone()


def batch_seq_inputs(L=6, B=2, D=32):
    """(B, L, D) layout — matches batch_first=True."""
    q = torch.randn(B, L, D)
    return q, q.clone(), q.clone()


# ---------------------------------------------------------------------------
# Drop-in API: __init__ accepts all nn.MultiheadAttention parameters
# ---------------------------------------------------------------------------

class TestDropInInit:
    def test_standard_params_accepted(self):
        # Should not raise
        MockModularAttention(
            embed_dim=64,
            num_heads=8,
            dropout=0.1,
            bias=True,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=None,
            vdim=None,
            batch_first=False,
            device=None,
            dtype=None,
        )

    def test_mma_kwargs_appended(self):
        MockModularAttention(
            embed_dim=32,
            num_heads=4,
            kernel="mock_theta_omega",
            init_temperature=0.5,
        )

    def test_invalid_embed_heads_raises(self):
        with pytest.raises(ValueError):
            MockModularAttention(embed_dim=15, num_heads=4)

    def test_invalid_kernel_raises(self):
        with pytest.raises(ValueError):
            MockModularAttention(embed_dim=16, num_heads=2, kernel="bad")

    def test_invalid_temperature_raises(self):
        with pytest.raises(ValueError):
            MockModularAttention(embed_dim=16, num_heads=2, init_temperature=0.0)


# ---------------------------------------------------------------------------
# Drop-in API: forward returns (Tensor, Optional[Tensor])
# ---------------------------------------------------------------------------

class TestDropInForward:
    def test_return_type_is_tuple(self):
        attn = make_mma()
        q, k, v = seq_batch_inputs()
        out = attn(q, k, v)
        assert isinstance(out, tuple) and len(out) == 2

    def test_output_tensor_shape_seq_batch(self):
        L, B, D = 6, 2, 32
        attn = make_mma(embed_dim=D, num_heads=4)
        q, k, v = torch.randn(L, B, D), torch.randn(L, B, D), torch.randn(L, B, D)
        out, _ = attn(q, k, v)
        assert out.shape == (L, B, D)

    def test_output_tensor_shape_batch_first(self):
        B, L, D = 2, 6, 32
        attn = make_mma(embed_dim=D, num_heads=4, batch_first=True)
        q, k, v = torch.randn(B, L, D), torch.randn(B, L, D), torch.randn(B, L, D)
        out, _ = attn(q, k, v)
        assert out.shape == (B, L, D)

    def test_need_weights_false_returns_none(self):
        attn = make_mma()
        q, k, v = seq_batch_inputs()
        _, weights = attn(q, k, v, need_weights=False)
        assert weights is None

    def test_need_weights_true_returns_tensor(self):
        attn = make_mma()
        q, k, v = seq_batch_inputs(L=5, B=2, D=32)
        _, weights = attn(q, k, v, need_weights=True, average_attn_weights=True)
        # average_attn_weights=True → (B, L, S)
        assert weights is not None
        assert weights.shape == (2, 5, 5)

    def test_need_weights_no_average(self):
        L, B, D, H = 5, 2, 32, 4
        attn = make_mma(embed_dim=D, num_heads=H)
        q, k, v = seq_batch_inputs(L=L, B=B, D=D)
        _, weights = attn(q, k, v, need_weights=True, average_attn_weights=False)
        assert weights.shape == (B, H, L, L)

    def test_weights_sum_to_one(self):
        attn = make_mma()
        q, k, v = seq_batch_inputs(L=8, B=2, D=32)
        _, weights = attn(q, k, v, need_weights=True, average_attn_weights=False)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_weights_nonnegative(self):
        attn = make_mma()
        q, k, v = seq_batch_inputs()
        _, weights = attn(q, k, v, need_weights=True)
        assert (weights >= 0).all()


# ---------------------------------------------------------------------------
# Cross-attention (kdim / vdim)
# ---------------------------------------------------------------------------

class TestCrossAttention:
    def test_cross_attention_different_kdim(self):
        attn = MockModularAttention(embed_dim=32, num_heads=4, kdim=16, vdim=16)
        q = torch.randn(5, 2, 32)   # query
        k = torch.randn(8, 2, 16)   # key (different dim and length)
        v = torch.randn(8, 2, 16)
        out, _ = attn(q, k, v)
        assert out.shape == (5, 2, 32)


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------

class TestMasking:
    def test_key_padding_mask_zeros_out_padded(self):
        """Padded positions should receive near-zero attention weight."""
        attn = make_mma(embed_dim=16, num_heads=2)
        L, B, D = 6, 2, 16
        q = torch.randn(L, B, D)
        # Mask out all but the first position for batch item 1
        pad_mask = torch.zeros(B, L, dtype=torch.bool)
        pad_mask[1, 2:] = True      # mask positions 2..5 for item 1
        _, weights = attn(q, q, q, key_padding_mask=pad_mask,
                          need_weights=True, average_attn_weights=False)
        # weights: (B=2, H=2, L=6, S=6)
        # For batch item 1, columns 2..5 should be zero
        masked_weights = weights[1, :, :, 2:]   # (H, L, 4)
        assert masked_weights.abs().max().item() < 1e-5

    def test_bool_attn_mask_causal_upper_triangle(self):
        """Boolean upper-triangular attn_mask should prevent future look-ahead."""
        L, B, D = 6, 1, 16
        attn = make_mma(embed_dim=D, num_heads=2)
        q = torch.randn(L, B, D)
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
        _, weights = attn(q, q, q, attn_mask=causal,
                          need_weights=True, average_attn_weights=False)
        # weights: (1, 2, L, L) — upper triangle must be 0
        w = weights[0, 0]       # (L, L)
        upper = torch.triu(w, diagonal=1)
        assert upper.abs().max().item() < 1e-5

    def test_is_causal_equivalent_to_explicit_mask(self):
        L, B, D = 5, 1, 16
        attn = make_mma(embed_dim=D, num_heads=2)
        q = torch.randn(L, B, D)
        causal_mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)

        with torch.no_grad():
            _, w_flag = attn(q, q, q, is_causal=True,
                             need_weights=True, average_attn_weights=False)
            _, w_mask = attn(q, q, q, attn_mask=causal_mask,
                             need_weights=True, average_attn_weights=False)

        assert torch.allclose(w_flag, w_mask, atol=1e-6)

    def test_float_attn_mask_large_negative_masks(self):
        L, B, D = 4, 1, 16
        attn = make_mma(embed_dim=D, num_heads=2)
        q = torch.randn(L, B, D)
        # Float mask: -inf on upper triangle
        float_mask = torch.full((L, L), 0.0)
        float_mask = float_mask.masked_fill(
            torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1), float("-inf")
        )
        _, weights = attn(q, q, q, attn_mask=float_mask,
                          need_weights=True, average_attn_weights=False)
        w = weights[0, 0]
        upper = torch.triu(w, diagonal=1)
        assert upper.abs().max().item() < 1e-5


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------

class TestGradients:
    def test_gradient_flows_through_kernel(self):
        attn = make_mma(embed_dim=16, num_heads=2)
        q = torch.randn(4, 2, 16, requires_grad=True)
        out, _ = attn(q, q, q)
        out.sum().backward()
        assert q.grad is not None

    def test_temperature_gradient(self):
        attn = make_mma(embed_dim=16, num_heads=2)
        q = torch.randn(4, 2, 16)
        out, _ = attn(q, q, q)
        out.sum().backward()
        assert attn.log_temperature.grad is not None


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------

class TestTemperature:
    def test_temperature_always_positive(self):
        attn = make_mma(init_temperature=0.1)
        assert attn.temperature.item() > 0

    def test_init_temperature_respected(self):
        attn = make_mma(init_temperature=2.5)
        assert abs(attn.temperature.item() - 2.5) < 1e-4


# ---------------------------------------------------------------------------
# All four kernels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "kernel",
    ["mock_theta_f", "mock_theta_omega", "mock_theta_phi", "q_series"],
)
class TestAllKernels:
    def test_output_shape(self, kernel):
        attn = MockModularAttention(embed_dim=16, num_heads=2, kernel=kernel)
        q, k, v = seq_batch_inputs(L=5, B=2, D=16)
        out, _ = attn(q, k, v)
        assert out.shape == (5, 2, 16)

    def test_weights_valid_distribution(self, kernel):
        attn = MockModularAttention(embed_dim=16, num_heads=2, kernel=kernel)
        q, k, v = seq_batch_inputs(L=5, B=2, D=16)
        _, weights = attn(q, k, v, need_weights=True, average_attn_weights=False)
        assert (weights >= 0).all()
        sums = weights.sum(-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# ---------------------------------------------------------------------------
# add_bias_kv / add_zero_attn
# ---------------------------------------------------------------------------

class TestOptionalFeatures:
    def test_add_bias_kv_output_shape(self):
        attn = make_mma(embed_dim=16, num_heads=2, add_bias_kv=True)
        q, k, v = seq_batch_inputs(L=5, B=2, D=16)
        out, _ = attn(q, k, v)
        assert out.shape == (5, 2, 16)

    def test_add_zero_attn_output_shape(self):
        attn = make_mma(embed_dim=16, num_heads=2, add_zero_attn=True)
        q, k, v = seq_batch_inputs(L=5, B=2, D=16)
        out, _ = attn(q, k, v)
        assert out.shape == (5, 2, 16)


# ---------------------------------------------------------------------------
# Comparison with nn.MultiheadAttention (API compatibility check)
# ---------------------------------------------------------------------------

class TestNNCompatibility:
    """Verify MockModularAttention can be swapped for nn.MultiheadAttention
    without changing any call-site code."""

    def _run_with(self, module, L=6, B=2, D=32):
        q = torch.randn(L, B, D)
        return module(q, q, q)

    def test_same_call_site_works(self):
        std = nn.MultiheadAttention(32, 4, dropout=0.0, batch_first=False)
        mma = MockModularAttention(32, 4, dropout=0.0, batch_first=False)
        # Both should accept identical arguments without error
        out_std, w_std = self._run_with(std)
        out_mma, w_mma = self._run_with(mma)
        assert out_std.shape == out_mma.shape
        assert w_std.shape == w_mma.shape
