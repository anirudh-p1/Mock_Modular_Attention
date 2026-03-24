"""
Mock Modular Attention
======================
A prototype Transformer attention mechanism that replaces standard softmax
weighting with kernels derived from q-series and Ramanujan's mock theta
functions, embedding an approximate modular-symmetry inductive bias into
sequential learning.

Modules
-------
qseries    : Pure-functional truncated q-series and mock theta functions.
kernels    : Learnable PyTorch nn.Module wrappers for each kernel.
attention  : MockModularAttention — drop-in for nn.MultiheadAttention.
transformer: Mini Transformer using MockModularAttention blocks.
"""

from .attention import MockModularAttention
from .kernels import (
    MockThetaKernel,
    MockThetaFKernel,
    MockThetaOmegaKernel,
    MockThetaPhiKernel,
    JacobiThetaKernel,
    build_kernel,
)
from .transformer import MockModularTransformer, MockModularTransformerBlock
from .qseries import (
    q_series_kernel,
    mock_theta_f,
    mock_theta_omega,
    mock_theta_phi,
    scores_to_q,
    mock_modular_weights,
)

__all__ = [
    # Attention (drop-in for nn.MultiheadAttention)
    "MockModularAttention",
    # Transformer blocks
    "MockModularTransformer",
    "MockModularTransformerBlock",
    # Kernel modules
    "MockThetaKernel",
    "MockThetaFKernel",
    "MockThetaOmegaKernel",
    "MockThetaPhiKernel",
    "JacobiThetaKernel",
    "build_kernel",
    # Functional q-series
    "q_series_kernel",
    "mock_theta_f",
    "mock_theta_omega",
    "mock_theta_phi",
    "scores_to_q",
    "mock_modular_weights",
]
