"""
Mock Theta Kernel modules — learnable PyTorch wrappers around each q-series.

Design rationale
----------------
Each :class:`MockThetaKernel` subclass wraps one Ramanujan mock theta
function as an ``nn.Module``.  The only learnable parameter is the
**temperature** τ > 0, which scales the raw attention scores before the
sigmoid mapping that converts them to the nome q ∈ (0, 1).

Temperature has a direct mathematical interpretation: it determines the
"operating region" of the mock theta function on the unit disk.  A small τ
drives q toward the boundary {0, 1}, where the series converges very
quickly or very slowly respectively.  Letting the model learn τ via
backpropagation is therefore equivalent to letting it search for the
optimal modular operating point — a natural analogue of learning the shape
of the attention distribution.

Unlike a simple softmax temperature, which merely sharpens or flattens the
same exponential, the temperature here interacts with the *structure* of the
mock theta expansion — different terms q^{n²} / denom_n are affected
non-uniformly, creating a richer learned bias landscape.

Class hierarchy
---------------
MockThetaKernel           (abstract base)
├── MockThetaFKernel      – Ramanujan f(q)
├── MockThetaOmegaKernel  – Ramanujan ω(q)
├── MockThetaPhiKernel    – Ramanujan φ(q)  (oscillatory; abs is taken)
└── JacobiThetaKernel     – Classical θ₃(q)  (fully modular reference)

All kernels share the same forward signature and can be swapped in/out of
:class:`~mock_modular_attention.attention.MockModularAttention` without
changing any other part of the model.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from .qseries import (
    mock_theta_f,
    mock_theta_omega,
    mock_theta_phi,
    q_series_kernel,
    scores_to_q,
)


class MockThetaKernel(nn.Module, ABC):
    """Abstract base class for all mock theta attention kernels.

    Subclasses implement :meth:`kernel_values`, which maps a tensor of nome
    values q ∈ (0, 1) to non-negative kernel values.  The base class
    handles temperature-scaled sigmoid mapping and L1 normalisation.

    Parameters
    ----------
    num_terms:
        Truncation depth of the q-series.  More terms → more accurate but
        more compute.  Typical range: 4–16.
    init_temperature:
        Initial value of the learnable temperature parameter τ.  The actual
        temperature is kept positive via ``softplus``.
    eps:
        Numerical stability constant for normalisation.
    """

    def __init__(
        self,
        num_terms: int = 8,
        init_temperature: float = 1.0,
        eps: float = 1e-9,
    ) -> None:
        super().__init__()
        self.num_terms = num_terms
        self.eps = eps
        # Store τ in unconstrained space; actual temperature = softplus(τ_raw)
        # initialised so that softplus(τ_raw) ≈ init_temperature.
        tau_raw_init = math.log(math.expm1(init_temperature))
        self.tau_raw = nn.Parameter(torch.tensor(tau_raw_init))

    @property
    def temperature(self) -> torch.Tensor:
        """Positive-constrained temperature τ = softplus(τ_raw)."""
        return nn.functional.softplus(self.tau_raw)

    @abstractmethod
    def kernel_values(self, q: torch.Tensor) -> torch.Tensor:
        """Evaluate the mock theta series at nome values q ∈ (0, 1).

        Parameters
        ----------
        q:
            Tensor with values strictly in (0, 1), any shape.

        Returns
        -------
        torch.Tensor
            Non-negative kernel values, same shape as *q*.
        """

    def forward(
        self,
        scores: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert raw attention scores to normalised Mock Modular weights.

        Parameters
        ----------
        scores:
            Raw attention logits, shape ``(batch, heads, query_len, key_len)``
            or any ``(..., key_len)`` shape.
        mask:
            Boolean mask broadcastable to *scores*; ``True`` marks positions
            to exclude (set to −1e9 before sigmoid so q ≈ 0 and the kernel
            value ≈ 1, which washes out during normalisation).

        Returns
        -------
        torch.Tensor
            Attention weights: same shape as *scores*, non-negative, summing
            to 1 along the last dimension.
        """
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        q = scores_to_q(scores, temperature=self.temperature)
        values = self.kernel_values(q)

        # Hard-zero masked positions: additive score masking drives q→0 but
        # kernel(q=0) = a₀ ≠ 0, so we must explicitly zero the kernel values.
        if mask is not None:
            values = values.masked_fill(mask, 0.0)

        weights = values / (values.sum(dim=-1, keepdim=True) + self.eps)
        return weights

    def extra_repr(self) -> str:
        return (
            f"num_terms={self.num_terms}, "
            f"temperature={self.temperature.item():.4f}"
        )


# ---------------------------------------------------------------------------
# Concrete kernels
# ---------------------------------------------------------------------------

class MockThetaFKernel(MockThetaKernel):
    """Attention kernel derived from Ramanujan's mock theta function f(q).

    Series
    ~~~~~~
    ``f(q) = Σ_{n=0}^{N-1} q^{n²} / ((-q; q)_n)²``

    where ``(-q; q)_n = (1+q)(1+q²)···(1+q^n)``.

    Inductive bias
    ~~~~~~~~~~~~~~
    The denominator squares the rising q-Pochhammer product, which grows
    rapidly with n, causing high-order terms to be strongly damped.  This
    gives f(q) a *heavy-tailed-but-concentrated* weight profile: the
    leading term contributes most, but subsequent terms add meaningful
    structure governed by partition-theoretic identities of the q-Pochhammer
    symbol.  The result is attention weights that emphasise the top-scoring
    token while preserving smooth context from the rest of the sequence.

    This is the closest Ramanujan mock theta analogue to the standard
    softmax — it is purely positive and concentrating — but constrained by
    the modular geometry of f(q) rather than a raw exponential.
    """

    def kernel_values(self, q: torch.Tensor) -> torch.Tensor:
        return mock_theta_f(q, num_terms=self.num_terms)


class MockThetaOmegaKernel(MockThetaKernel):
    """Attention kernel derived from Ramanujan's mock theta function ω(q).

    Series
    ~~~~~~
    ``ω(q) = Σ_{n=0}^{N-1} q^{2n(n+1)} / ((q; q²)_{n+1})²``

    where ``(q; q²)_{n+1} = (1-q)(1-q³)···(1-q^{2n+1})``.

    Inductive bias
    ~~~~~~~~~~~~~~
    The exponent 2n(n+1) grows faster than n² (used in f and θ₃), so ω(q)
    attenuates higher-order terms more aggressively.  Combined with the
    diverging denominator (q; q²)_{n+1} as q → 1, this produces
    *sparser* attention distributions: the kernel focuses even more sharply
    on the top-scoring token.

    ω(q) is the mock theta function that arises naturally in the study of
    unimodal sequences, lending the kernel a discrete-geometric inductive
    bias oriented toward dominant, unimodal attention peaks.
    """

    def kernel_values(self, q: torch.Tensor) -> torch.Tensor:
        return mock_theta_omega(q, num_terms=self.num_terms)


class MockThetaPhiKernel(MockThetaKernel):
    """Attention kernel derived from Ramanujan's mock theta function φ(q).

    Series
    ~~~~~~
    ``φ(q) = Σ_{n=0}^{N-1} (-1)^n q^{n²} / (-q²; q²)_n``

    where ``(-q²; q²)_n = (1+q²)(1+q⁴)···(1+q^{2n})``.

    The alternating sign (−1)^n means φ(q) can be negative.  The
    **absolute value** |φ(q)| is used as the kernel value.

    Inductive bias
    ~~~~~~~~~~~~~~
    |φ(q)| has a markedly *flatter* profile than f(q) or ω(q), because the
    alternating positive and negative terms partially cancel.  The result is
    an attention distribution that is more **uniform**, acting as a soft
    regulariser that prevents the model from over-committing to a single
    token.

    This is particularly useful in tasks where distributing attention broadly
    improves generalisation (e.g., long-range dependency modelling), while
    the oscillatory q-series structure still provides the modular-symmetry
    inductive bias.
    """

    def kernel_values(self, q: torch.Tensor) -> torch.Tensor:
        return mock_theta_phi(q, num_terms=self.num_terms)


class JacobiThetaKernel(MockThetaKernel):
    """Attention kernel derived from the classical Jacobi theta-3 series.

    Series
    ~~~~~~
    ``θ₃(q) = Σ_{n=0}^{N-1} q^{n²}``

    This is the simplest *fully modular* q-series (not mock).  Its modular
    transformation is exact:

        θ₃(0 | -1/τ) = (-iτ)^{1/2} · θ₃(0 | τ)

    Inductive bias
    ~~~~~~~~~~~~~~
    Because no Pochhammer denominator is present, all terms are purely
    monomial q^{n²}.  The weight profile is intermediate between f(q) (which
    damps higher terms via the denominator) and a flat distribution.  This
    makes JacobiThetaKernel a useful **baseline** when comparing Mock
    Modular Attention to softmax: it inherits the positive q-series
    structure but without the partition-theoretic depth of the mock theta
    functions.
    """

    def kernel_values(self, q: torch.Tensor) -> torch.Tensor:
        return q_series_kernel(q, num_terms=self.num_terms)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_KERNEL_REGISTRY: dict[str, type[MockThetaKernel]] = {
    "mock_theta_f": MockThetaFKernel,
    "mock_theta_omega": MockThetaOmegaKernel,
    "mock_theta_phi": MockThetaPhiKernel,
    "q_series": JacobiThetaKernel,
}


def build_kernel(
    name: str,
    num_terms: int = 8,
    init_temperature: float = 1.0,
) -> MockThetaKernel:
    """Instantiate a :class:`MockThetaKernel` by name.

    Parameters
    ----------
    name:
        One of ``"mock_theta_f"``, ``"mock_theta_omega"``,
        ``"mock_theta_phi"``, ``"q_series"``.
    num_terms:
        Truncation depth.
    init_temperature:
        Initial temperature value.

    Returns
    -------
    MockThetaKernel
        The instantiated kernel module.

    Raises
    ------
    ValueError
        If *name* is not registered.
    """
    if name not in _KERNEL_REGISTRY:
        raise ValueError(
            f"Unknown kernel '{name}'. "
            f"Available: {sorted(_KERNEL_REGISTRY.keys())}"
        )
    return _KERNEL_REGISTRY[name](
        num_terms=num_terms,
        init_temperature=init_temperature,
    )
