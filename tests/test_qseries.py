"""
Tests for mock_modular_attention.qseries

Validates:
- Shapes are preserved for all functions
- q_series_kernel, mock_theta_f, mock_theta_omega, mock_theta_phi are positive
- Convergence (more terms → larger partial sum for positive-term series)
- scores_to_q maps any real → (0,1)
- mock_modular_weights: non-negative, sums to 1, correct kernels dispatched
"""

import pytest
import torch
from mock_modular_attention.qseries import (
    q_series_kernel,
    mock_theta_f,
    mock_theta_omega,
    mock_theta_phi,
    scores_to_q,
    mock_modular_weights,
)


# ---------------------------------------------------------------------------
# q_series_kernel  (Jacobi theta-3)
# ---------------------------------------------------------------------------

class TestQSeriesKernel:
    def test_shape_preserved(self):
        q = torch.rand(3, 4, 5)
        out = q_series_kernel(q, num_terms=8)
        assert out.shape == q.shape

    def test_positive_for_positive_q(self):
        q = torch.rand(20) * 0.98 + 0.01
        assert (q_series_kernel(q, num_terms=8) > 0).all()

    def test_n0_term_is_one(self):
        # When num_terms=1, only n=0 contributes: q^0 = 1
        q = torch.rand(10) * 0.9 + 0.05
        out = q_series_kernel(q, num_terms=1)
        assert torch.allclose(out, torch.ones(10), atol=1e-6)

    def test_monotone_in_num_terms(self):
        # All terms q^{n^2} >= 0, so adding more must be non-decreasing
        q = torch.tensor([0.5])
        val4 = q_series_kernel(q, num_terms=4).item()
        val8 = q_series_kernel(q, num_terms=8).item()
        assert val8 >= val4


# ---------------------------------------------------------------------------
# mock_theta_f
# ---------------------------------------------------------------------------

class TestMockThetaF:
    def test_shape_preserved(self):
        q = torch.rand(2, 3)
        assert mock_theta_f(q, num_terms=6).shape == q.shape

    def test_positive(self):
        q = torch.rand(30) * 0.95 + 0.01
        assert (mock_theta_f(q, num_terms=8) > 0).all()

    def test_n0_term_contributes_one(self):
        # With num_terms=1, only q^0/1 = 1 contributes
        q = torch.rand(10) * 0.9 + 0.05
        out = mock_theta_f(q, num_terms=1)
        assert torch.allclose(out, torch.ones(10), atol=1e-6)

    def test_convergence_increases_with_terms(self):
        q = torch.tensor([0.5])
        val4 = mock_theta_f(q, num_terms=4).item()
        val8 = mock_theta_f(q, num_terms=8).item()
        assert val8 >= val4

    def test_gradient_flows(self):
        q = (torch.rand(4) * 0.9 + 0.05).detach().requires_grad_(True)
        out = mock_theta_f(q, num_terms=4).sum()
        out.backward()
        assert q.grad is not None


# ---------------------------------------------------------------------------
# mock_theta_omega
# ---------------------------------------------------------------------------

class TestMockThetaOmega:
    def test_shape_preserved(self):
        q = torch.rand(4)
        assert mock_theta_omega(q, num_terms=5).shape == q.shape

    def test_positive(self):
        # omega is positive for q well away from 1
        q = torch.rand(20) * 0.8 + 0.01
        assert (mock_theta_omega(q, num_terms=6) > 0).all()

    def test_n0_dominates_small_q(self):
        # For very small q, only n=0 term matters:
        # omega ~ q^0 / (1-q)^2 ~ 1 for q→0
        q = torch.tensor([1e-4])
        val = mock_theta_omega(q, num_terms=6).item()
        assert abs(val - 1.0) < 0.01  # close to 1 for tiny q


# ---------------------------------------------------------------------------
# mock_theta_phi
# ---------------------------------------------------------------------------

class TestMockThetaPhi:
    def test_shape_preserved(self):
        q = torch.rand(5, 3)
        assert mock_theta_phi(q, num_terms=6).shape == q.shape

    def test_nonnegative(self):
        # phi returns absolute value → always >= 0
        q = torch.rand(30) * 0.95 + 0.01
        assert (mock_theta_phi(q, num_terms=8) >= 0).all()

    def test_n0_term_one(self):
        q = torch.rand(8) * 0.9 + 0.05
        val = mock_theta_phi(q, num_terms=1)
        assert torch.allclose(val, torch.ones(8), atol=1e-6)


# ---------------------------------------------------------------------------
# scores_to_q
# ---------------------------------------------------------------------------

class TestScoresToQ:
    def test_output_in_open_unit_interval(self):
        scores = torch.randn(100)
        q = scores_to_q(scores)
        assert (q > 0).all()
        assert (q < 1).all()

    def test_temperature_scaling(self):
        scores = torch.tensor([1.0])
        q_t1 = scores_to_q(scores, temperature=1.0).item()
        q_t2 = scores_to_q(scores, temperature=2.0).item()
        # Higher temperature → sigmoid argument halved → q closer to 0.5
        assert abs(q_t2 - 0.5) < abs(q_t1 - 0.5)

    def test_monotone_in_score(self):
        scores = torch.linspace(-5, 5, 20)
        q = scores_to_q(scores)
        assert (q.diff() > 0).all()


# ---------------------------------------------------------------------------
# mock_modular_weights
# ---------------------------------------------------------------------------

class TestMockModularWeights:
    @pytest.mark.parametrize(
        "kernel",
        ["mock_theta_f", "mock_theta_omega", "mock_theta_phi", "q_series"],
    )
    def test_sums_to_one(self, kernel):
        scores = torch.randn(2, 4, 8, 8)
        w = mock_modular_weights(scores, num_terms=6, kernel=kernel)
        sums = w.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    @pytest.mark.parametrize(
        "kernel",
        ["mock_theta_f", "mock_theta_omega", "mock_theta_phi", "q_series"],
    )
    def test_nonnegative(self, kernel):
        scores = torch.randn(2, 6)
        w = mock_modular_weights(scores, num_terms=4, kernel=kernel)
        assert (w >= 0).all()

    def test_invalid_kernel_raises(self):
        with pytest.raises(ValueError, match="Unknown kernel"):
            mock_modular_weights(torch.randn(1, 4), kernel="bad_kernel")

    def test_temperature_affects_distribution(self):
        scores = torch.tensor([[2.0, 0.0, -2.0]])
        w_hot = mock_modular_weights(scores, temperature=0.5)
        w_cold = mock_modular_weights(scores, temperature=2.0)
        # Hot temperature → more peaked (lower entropy)
        entropy = lambda w: -(w * (w + 1e-9).log()).sum(-1)
        assert entropy(w_hot).item() < entropy(w_cold).item()
