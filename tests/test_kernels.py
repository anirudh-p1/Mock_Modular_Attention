"""
Tests for mock_modular_attention.kernels (nn.Module wrappers).

Validates:
- All kernel modules produce valid (non-negative, sum-to-1) attention weights
- Learnable temperature is positive and receives gradients
- build_kernel factory dispatches correctly
- extra_repr contains useful debug info
"""

import math
import pytest
import torch
from mock_modular_attention.kernels import (
    MockThetaFKernel,
    MockThetaOmegaKernel,
    MockThetaPhiKernel,
    JacobiThetaKernel,
    build_kernel,
)


ALL_KERNEL_CLASSES = [
    MockThetaFKernel,
    MockThetaOmegaKernel,
    MockThetaPhiKernel,
    JacobiThetaKernel,
]

ALL_KERNEL_NAMES = [
    "mock_theta_f",
    "mock_theta_omega",
    "mock_theta_phi",
    "q_series",
]


@pytest.mark.parametrize("KernelClass", ALL_KERNEL_CLASSES)
class TestKernelModules:
    def test_output_is_valid_distribution(self, KernelClass):
        kernel = KernelClass(num_terms=6)
        scores = torch.randn(2, 4, 8, 8)
        weights = kernel(scores)
        assert (weights >= 0).all()
        sums = weights.sum(-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_output_shape_preserved(self, KernelClass):
        kernel = KernelClass(num_terms=6)
        scores = torch.randn(3, 5, 7)
        weights = kernel(scores)
        assert weights.shape == scores.shape

    def test_temperature_is_positive(self, KernelClass):
        kernel = KernelClass(num_terms=4, init_temperature=1.5)
        assert kernel.temperature.item() > 0

    def test_init_temperature_respected(self, KernelClass):
        kernel = KernelClass(num_terms=4, init_temperature=2.0)
        assert abs(kernel.temperature.item() - 2.0) < 1e-4

    def test_gradient_flows_through_temperature(self, KernelClass):
        kernel = KernelClass(num_terms=4)
        scores = torch.randn(2, 6, requires_grad=True)
        weights = kernel(scores)
        weights.sum().backward()
        assert kernel.tau_raw.grad is not None

    def test_boolean_mask_zeros_out(self, KernelClass):
        kernel = KernelClass(num_terms=4)
        scores = torch.randn(1, 6)
        # Mask last 3 positions
        mask = torch.zeros(1, 6, dtype=torch.bool)
        mask[0, 3:] = True
        weights = kernel(scores, mask=mask)
        assert weights[0, 3:].abs().max().item() < 1e-6

    def test_extra_repr_contains_num_terms(self, KernelClass):
        kernel = KernelClass(num_terms=8)
        assert "8" in kernel.extra_repr()


class TestBuildKernelFactory:
    @pytest.mark.parametrize("name", ALL_KERNEL_NAMES)
    def test_factory_dispatches_correctly(self, name):
        kernel = build_kernel(name, num_terms=6)
        scores = torch.randn(2, 4, 8)
        weights = kernel(scores)
        sums = weights.sum(-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Unknown kernel"):
            build_kernel("fake_kernel")

    def test_init_temperature_forwarded(self):
        kernel = build_kernel("mock_theta_f", init_temperature=3.0)
        assert abs(kernel.temperature.item() - 3.0) < 1e-4
