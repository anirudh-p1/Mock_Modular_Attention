"""
Q-Series and Mock Theta Function kernels (truncated power series).

Mathematical background
-----------------------
Ramanujan's mock theta functions are formal q-series (power series in the
nome q, |q| < 1) that transform under the modular group SL(2,Z) almost
like modular forms, but carry a non-holomorphic correction term — the
"shadow" — that accounts for the symmetry defect.

Specifically, each mock theta function M(q) possesses a shadow S(τ) such
that the *completion*  M̂(τ) = M(q) + S(τ)  transforms exactly as a
modular form of weight 1/2, where q = e^{2πiτ} with Im(τ) > 0.

Embedding these functions as attention kernels instils an *approximate
modular-symmetry inductive bias*: the model is structurally encouraged to
weight tokens in ways that respect the near-symmetric geometry encoded by
Ramanujan's q-series, rather than relying on unconstrained softmax logits.

Series implemented
------------------
The three Ramanujan third-order mock theta functions implemented here are:

  f(q)  = Σ_{n≥0}  q^{n²}             / ((-q; q)_n)²
  ω(q)  = Σ_{n≥0}  q^{2n(n+1)}        / ((q; q²)_{n+1})²
  φ(q)  = Σ_{n≥0}  (-1)^n q^{n²}      / (-q²; q²)_n

plus the classical Jacobi theta-3 series (the simplest positive q-series):

  θ₃(q) = Σ_{n≥0}  q^{n²}

All series are truncated to N terms for computational tractability.
Truncation errors are exponentially small in N for |q| < 1:

  |f(q) - f_N(q)|  = O(q^{N²})
  |ω(q) - ω_N(q)|  = O(q^{2N(N+1)})
  |θ₃(q) - θ_N(q)| = O(q^{N²})

q-Pochhammer notation
---------------------
  (a; q)_n = Π_{k=0}^{n-1} (1 - a·qᵏ)   [(a; q)_0 = 1]
  (-q; q)_n = Π_{k=1}^{n}  (1 + q^k)
  (q; q²)_n = Π_{k=0}^{n-1} (1 - q^{2k+1})
  (-q²; q²)_n = Π_{k=0}^{n-1} (1 + q^{2k+2})

Public API
----------
q_series_kernel        – Jacobi θ₃:  Σ q^{n²}
mock_theta_f           – Ramanujan f: Σ q^{n²} / (-q; q)_n²
mock_theta_omega       – Ramanujan ω: Σ q^{2n(n+1)} / (q; q²)_{n+1}²
mock_theta_phi         – Ramanujan φ: Σ (-1)^n q^{n²} / (-q²; q²)_n
scores_to_q            – Map real-valued scores → q ∈ (0, 1) via sigmoid
mock_modular_weights   – Full pipeline: scores → normalised attention weights
"""

from __future__ import annotations

import torch
from typing import Literal

KernelName = Literal["mock_theta_f", "mock_theta_omega", "mock_theta_phi", "q_series"]


# ---------------------------------------------------------------------------
# Core q-series
# ---------------------------------------------------------------------------

def q_series_kernel(q: torch.Tensor, num_terms: int = 8) -> torch.Tensor:
    """Truncated Jacobi theta-3 series: ``θ_N(q) = Σ_{n=0}^{N-1} q^{n²}``.

    This is the canonical positive q-series, equal to the classical modular
    form θ₃(0 | τ) on the upper half-plane (with q = e^{2πiτ}).  Its
    modular transformation is exact:

        θ₃(0 | -1/τ) = (-iτ)^{1/2} · θ₃(0 | τ)

    All terms are non-negative, so the partial sum is always ≥ 1.

    Parameters
    ----------
    q:
        Tensor whose values lie strictly in (0, 1).  Any shape is accepted.
    num_terms:
        Number of terms N retained in the truncated series.
        Truncation error: O(q^{N²}).

    Returns
    -------
    torch.Tensor
        Same shape as *q*.  Every element ≥ 1.
    """
    result = torch.zeros_like(q)
    for n in range(num_terms):
        result = result + q.pow(n * n)
    return result


def mock_theta_f(q: torch.Tensor, num_terms: int = 8) -> torch.Tensor:
    """Ramanujan's third-order mock theta function f(q), truncated.

    Exact definition
    ~~~~~~~~~~~~~~~~
    ``f(q) = Σ_{n=0}^{∞} q^{n²} / ((-q; q)_n)²``

    where  ``(-q; q)_n = Π_{k=1}^{n} (1 + q^k)``  (with ``(-q; q)_0 = 1``).

    All terms are strictly positive, so the partial sum is always ≥ 1.

    Inductive bias
    ~~~~~~~~~~~~~~
    The denominator ``((-q; q)_n)²`` is the square of a *q-Pochhammer* rising
    product, which encodes the partition-theoretic structure of the q-series.
    Attention weights derived from f(q) are biased toward tokens whose
    relative score places them in the principal growth regime of Ramanujan's
    modular structure — geometrically, they respect the torus symmetry of
    the associated modular curve.

    Parameters
    ----------
    q:
        Tensor with values in (0, 1).
    num_terms:
        Truncation depth N.  Truncation error: O(q^{N²}).

    Returns
    -------
    torch.Tensor
        Same shape as *q*.  Every element ≥ 1.
    """
    result = torch.zeros_like(q)
    denom = torch.ones_like(q)          # (-q; q)_0 = 1

    for n in range(num_terms):
        # n-th term:  q^{n²} / ((-q; q)_n)²
        result = result + q.pow(n * n) / denom.pow(2)
        # Recurrence: (-q; q)_{n+1} = (-q; q)_n · (1 + q^{n+1})
        denom = denom * (1.0 + q.pow(n + 1))

    return result


def mock_theta_omega(q: torch.Tensor, num_terms: int = 8) -> torch.Tensor:
    """Ramanujan's third-order mock theta function ω(q), truncated.

    Exact definition
    ~~~~~~~~~~~~~~~~
    ``ω(q) = Σ_{n=0}^{∞} q^{2n(n+1)} / ((q; q²)_{n+1})²``

    where  ``(q; q²)_{n+1} = Π_{k=0}^{n} (1 - q^{2k+1})``.

    All terms are positive for q ∈ (0, 1).

    Inductive bias
    ~~~~~~~~~~~~~~
    The exponent 2n(n+1) = 2·triangular_number grows faster than n² (used
    in f(q)), so ω(q) attenuates high-order terms more aggressively.  This
    produces *sparser* attention distributions than f(q) — the model is
    biased toward focusing on fewer, more salient tokens.

    The denominator is the *odd part* of the q-Pochhammer symbol, encoding
    the symmetry of Ramanujan's ω-function under the modular group.

    Numerical stability
    ~~~~~~~~~~~~~~~~~~~
    The denominator product (q; q²)_{n+1} → 0 as q → 1.  Each factor is
    clamped to ≥ 1e-8 to prevent division by zero while preserving the
    series structure.

    Parameters
    ----------
    q:
        Tensor with values in (0, 1).
    num_terms:
        Truncation depth N.  Truncation error: O(q^{2N(N+1)}).

    Returns
    -------
    torch.Tensor
        Same shape as *q*.  Every element > 0.
    """
    result = torch.zeros_like(q)
    denom = torch.ones_like(q)          # (q; q²)_0 = 1  (empty product)

    for n in range(num_terms):
        # Accumulate (1 − q^{2n+1}) to build (q; q²)_{n+1}
        factor = torch.clamp(1.0 - q.pow(2 * n + 1), min=1e-8)
        denom = denom * factor
        # n-th term:  q^{2n(n+1)} / ((q; q²)_{n+1})²
        result = result + q.pow(2 * n * (n + 1)) / denom.pow(2)

    return result


def mock_theta_phi(q: torch.Tensor, num_terms: int = 8) -> torch.Tensor:
    """Ramanujan's third-order mock theta function φ(q), truncated.

    Exact definition
    ~~~~~~~~~~~~~~~~
    ``φ(q) = Σ_{n=0}^{∞} (-1)^n q^{n²} / (-q²; q²)_n``

    where  ``(-q²; q²)_n = Π_{k=0}^{n-1} (1 + q^{2k+2})``
    (even q-Pochhammer, ``(-q²; q²)_0 = 1``).

    φ(q) is *real-valued* but can be **negative** for some q, because of the
    alternating signs (−1)^n.  The absolute value |φ(q)| is used as the
    attention kernel so that weights remain non-negative.

    Inductive bias
    ~~~~~~~~~~~~~~
    The alternating-sign structure makes φ(q) an *oscillatory* q-series.
    Using |φ(q)| as a kernel biases the model toward attention patterns that
    respect the alternating-parity symmetry of the mock theta function — a
    finer-grained constraint than the purely positive f(q) or ω(q).  In
    practice this produces more *uniform* weight distributions than f(q),
    acting as a soft regulariser on spiky attention patterns.

    Parameters
    ----------
    q:
        Tensor with values in (0, 1).
    num_terms:
        Truncation depth N.

    Returns
    -------
    torch.Tensor
        Same shape as *q*.  |φ(q)|, always ≥ 0.
    """
    result = torch.zeros_like(q)
    denom = torch.ones_like(q)          # (-q²; q²)_0 = 1

    for n in range(num_terms):
        sign = (-1.0) ** n
        # n-th term:  (-1)^n q^{n²} / (-q²; q²)_n
        result = result + sign * q.pow(n * n) / denom
        # Recurrence: (-q²; q²)_{n+1} = (-q²; q²)_n · (1 + q^{2n+2})
        denom = denom * (1.0 + q.pow(2 * (n + 1)))

    # Take absolute value: kernel values must be non-negative for attention
    return result.abs()


# ---------------------------------------------------------------------------
# Attention-weight pipeline
# ---------------------------------------------------------------------------

def scores_to_q(scores: torch.Tensor, temperature=1.0) -> torch.Tensor:
    """Map real-valued attention logits → q ∈ (0, 1) via sigmoid.

    In modular forms theory the nome q = e^{2πiτ} where τ lies in the upper
    half-plane.  Here τ is implicitly purely imaginary and the mapping

        q = σ(s / T) = 1 / (1 + e^{-s/T})

    sends s ∈ ℝ to q ∈ (0, 1), corresponding to |q| < 1 on the unit disk.

    The temperature T scales the input before sigmoid, controlling how
    sharply the mock theta kernel discriminates between high and low scores:
    small T → near-binary q; large T → q clustered near 0.5.

    Parameters
    ----------
    scores:
        Raw attention score tensor (any shape, any real values).
    temperature:
        Positive scaling factor applied before sigmoid.  Default 1.0.

    Returns
    -------
    torch.Tensor
        Values strictly in (0, 1), same shape as *scores*.
    """
    return torch.sigmoid(scores / temperature)


def mock_modular_weights(
    scores: torch.Tensor,
    num_terms: int = 8,
    kernel: KernelName = "mock_theta_f",
    temperature=1.0,
    eps: float = 1e-9,
) -> torch.Tensor:
    """Convert raw attention scores to Mock Modular Attention weights.

    Full pipeline
    ~~~~~~~~~~~~~
    1. ``q = sigmoid(scores / temperature)``   — map to nome ∈ (0, 1)
    2. ``values = kernel(q, num_terms)``        — evaluate mock theta series
    3. ``weights = values / Σ values``          — normalise to probability simplex

    This replaces the standard softmax pipeline
    ``weights = exp(scores) / Σ exp(scores)``
    with one whose weighting function is governed by Ramanujan's mock theta
    mathematics rather than the exponential function.

    Parameters
    ----------
    scores:
        Raw attention logits, shape ``(..., seq_len)``.
    num_terms:
        Truncation depth for the q-series.
    kernel:
        Weighting kernel: ``"mock_theta_f"``, ``"mock_theta_omega"``,
        ``"mock_theta_phi"``, or ``"q_series"``.
    temperature:
        Sigmoid temperature (see :func:`scores_to_q`).
    eps:
        Added to the normalisation denominator for numerical stability.

    Returns
    -------
    torch.Tensor
        Attention weights, same shape as *scores*, non-negative, summing to
        1 along the last dimension.

    Raises
    ------
    ValueError
        If *kernel* is not a recognised name.
    """
    q = scores_to_q(scores, temperature=temperature)

    if kernel == "mock_theta_f":
        values = mock_theta_f(q, num_terms=num_terms)
    elif kernel == "mock_theta_omega":
        values = mock_theta_omega(q, num_terms=num_terms)
    elif kernel == "mock_theta_phi":
        values = mock_theta_phi(q, num_terms=num_terms)
    elif kernel == "q_series":
        values = q_series_kernel(q, num_terms=num_terms)
    else:
        raise ValueError(
            f"Unknown kernel '{kernel}'. "
            "Choose from 'mock_theta_f', 'mock_theta_omega', "
            "'mock_theta_phi', 'q_series'."
        )

    # Normalise over last dimension → probability simplex
    weights = values / (values.sum(dim=-1, keepdim=True) + eps)
    return weights
