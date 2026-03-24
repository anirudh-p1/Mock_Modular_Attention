"""
MockModularAttention — drop-in replacement for ``nn.MultiheadAttention``.

The kernel K(Q, K) replaces standard ``softmax(QKᵀ/√d)`` with a
mock-theta / q-series weighting derived from Ramanujan's mathematics.

Mathematical kernel
-------------------
For each pair of positions (i, j):

  1. Compute raw dot-product score  s_ij = (Qᵢ · Kⱼ) / √d
  2. Map to nome  q_ij = sigmoid(s_ij / τ) = 1/(1 + e^{-s_ij/τ}) ∈ (0,1)
     This is equivalent to q = e^{iπτ̃} with τ̃ purely imaginary in the
     upper half-plane (Im(τ̃) = log(1/q_ij)/π > 0).
  3. Evaluate the 3rd-order Taylor kernel:
       K(q) = a₀ + a₁q + a₂q² + a₃q³   (Horner's method, O(d) per element)
  4. Normalise over key positions → attention weights w_ij

Taylor coefficients aₙ
-----------------------
The coefficients come from expanding each Ramanujan series in powers of q:

  mock_theta_f(q)     = 1 + q − 2q² + 3q³ + O(q⁴)    [a = 1, 1, −2, 3]
  mock_theta_omega(q) = 1 + 2q + 3q² + 4q³ + O(q⁴)   [a = 1, 2,  3, 4]
  mock_theta_phi(q)   = 1 − q + 0q² + q³ + O(q⁴)     [a = 1,−1,  0, 1]
  q_series/theta3(q)  = 1 + q + 0q² + 0q³ + O(q⁴)    [a = 1, 1,  0, 0]

Each kernel is non-negative on q ∈ [0,1] (verified analytically; see
module docstring of qseries.py for proofs).

Drop-in compatibility
---------------------
``MockModularAttention`` matches the ``__init__`` and ``forward`` signatures
of ``nn.MultiheadAttention`` exactly, including ``batch_first``, ``kdim``,
``vdim``, ``add_bias_kv``, ``add_zero_attn``, ``key_padding_mask``,
``attn_mask``, ``is_causal``, ``need_weights``, and ``average_attn_weights``.
Two MMA-specific keyword arguments (``kernel``, ``init_temperature``) are
appended at the end of ``__init__`` so existing call-sites require zero changes.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .qseries import KernelName

# ---------------------------------------------------------------------------
# Taylor-coefficient table
# ---------------------------------------------------------------------------
# 3rd-order Taylor expansion of each mock theta function around q = 0:
#   K(q) = a₀ + a₁q + a₂q² + a₃q³
#
# Derivation sketch (mock_theta_f):
#   n=0 term: 1
#   n=1 term: q/(1+q)² = q − 2q² + 3q³ − …
#   n≥2 terms start at q⁴ → irrelevant at 3rd order
#   ⟹  f(q) = 1 + q − 2q² + 3q³ + O(q⁴)
#
# All kernels satisfy K(q) ≥ 0 for q ∈ [0,1] (monotone/bounded proofs
# in module docstring of qseries.py).

_TAYLOR_COEFFS: dict[str, list[float]] = {
    #                    a₀    a₁    a₂    a₃
    "mock_theta_f":     [1.0,  1.0, -2.0,  3.0],
    "mock_theta_omega": [1.0,  2.0,  3.0,  4.0],
    "mock_theta_phi":   [1.0, -1.0,  0.0,  1.0],
    "q_series":         [1.0,  1.0,  0.0,  0.0],
}


def _horner_eval(q: torch.Tensor, kernel: KernelName) -> torch.Tensor:
    """Evaluate K(q) = a₀ + a₁q + a₂q² + a₃q³ using Horner's method.

    Horner factorisation: a₀ + q(a₁ + q(a₂ + q·a₃))
    Cost: 3 multiply-adds per element — O(1) in the degree for fixed order.

    The result is clamped to ≥ 0 as a numerical safety net; analytically
    every implemented kernel is non-negative on [0, 1].
    """
    a = _TAYLOR_COEFFS[kernel]
    val = torch.full_like(q, a[3])   # start from innermost coefficient
    val = q * val + a[2]
    val = q * val + a[1]
    val = q * val + a[0]
    return val.clamp(min=0.0)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MockModularAttention(nn.Module):
    """Multi-head attention with a Ramanujan mock-theta kernel.

    This class is a **drop-in replacement** for ``torch.nn.MultiheadAttention``.
    The ``__init__`` and ``forward`` signatures are identical; two optional
    keyword-only arguments (``kernel`` and ``init_temperature``) are appended
    after the standard ``dtype`` parameter.

    How it differs from softmax attention
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Standard attention:  w = softmax(QKᵀ / √d)
    Mock Modular:        w = normalise(K_τ(sigmoid(QKᵀ / (√d·τ))))

    where K_τ(q) = 1 + q − 2q² + 3q³  (default: mock_theta_f, 3rd order).

    The temperature τ is a single learnable scalar initialised to
    *init_temperature*.  It controls the "operating region" of the mock
    theta function on the unit disk — a mathematically motivated analogue
    of the softmax temperature.

    Parameters
    ----------
    embed_dim : int
        Total embedding dimension (must be divisible by ``num_heads``).
    num_heads : int
        Number of parallel attention heads.
    dropout : float
        Dropout probability applied to attention weights.
    bias : bool
        Add bias to projection layers.
    add_bias_kv : bool
        Append learned bias vectors to projected K and V sequences.
    add_zero_attn : bool
        Append a zero-padded position to K/V (after projection).
    kdim : int, optional
        Feature dimension of the key input (cross-attention).  Defaults
        to ``embed_dim``.
    vdim : int, optional
        Feature dimension of the value input (cross-attention).  Defaults
        to ``embed_dim``.
    batch_first : bool
        If ``True``, inputs/outputs are ``(batch, seq, dim)``; otherwise
        ``(seq, batch, dim)``.
    device, dtype
        Forwarded to all parameter tensors.
    kernel : KernelName
        Mock theta kernel to use.  One of ``"mock_theta_f"`` (default),
        ``"mock_theta_omega"``, ``"mock_theta_phi"``, ``"q_series"``.
    init_temperature : float
        Initial value of the learnable temperature τ > 0.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        batch_first: bool = False,
        device=None,
        dtype=None,
        # ---- MMA-specific (appended after standard args) ----
        kernel: KernelName = "mock_theta_f",
        init_temperature: float = 1.0,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        if kernel not in _TAYLOR_COEFFS:
            raise ValueError(
                f"Unknown kernel '{kernel}'. "
                f"Available: {sorted(_TAYLOR_COEFFS.keys())}"
            )
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )
        if init_temperature <= 0:
            raise ValueError("init_temperature must be positive.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout
        self.batch_first = batch_first
        self.add_bias_kv = add_bias_kv
        self.add_zero_attn = add_zero_attn
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.kernel = kernel
        self._scale = math.sqrt(self.head_dim)

        # Learnable temperature stored in log-space so τ = exp(log_τ) > 0 always.
        self.log_temperature = nn.Parameter(
            torch.tensor(math.log(init_temperature), **factory_kwargs)
        )

        # Q / K / V input projections (separate for clarity and kdim/vdim support)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        # Optional bias vectors appended to projected K and V sequences
        if add_bias_kv:
            self.bias_k = nn.Parameter(
                torch.empty(1, 1, embed_dim, **factory_kwargs)
            )
            self.bias_v = nn.Parameter(
                torch.empty(1, 1, embed_dim, **factory_kwargs)
            )
        else:
            self.bias_k = None
            self.bias_v = None

        self._dropout = nn.Dropout(dropout)
        self._reset_parameters()

    # ------------------------------------------------------------------
    # Parameter initialisation
    # ------------------------------------------------------------------

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        for proj in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            if proj.bias is not None:
                nn.init.constant_(proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
            nn.init.xavier_normal_(self.bias_v)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def temperature(self) -> torch.Tensor:
        """Positive temperature τ = exp(log_τ)."""
        return self.log_temperature.exp()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute Mock Modular Attention.

        Parameters match ``nn.MultiheadAttention.forward`` exactly.

        Parameters
        ----------
        query : Tensor
            Shape ``(L, B, E)`` or ``(B, L, E)`` if ``batch_first=True``.
        key : Tensor
            Shape ``(S, B, E)`` or ``(B, S, E)`` if ``batch_first=True``.
        value : Tensor
            Shape ``(S, B, E)`` or ``(B, S, E)`` if ``batch_first=True``.
        key_padding_mask : BoolTensor, optional
            Shape ``(B, S)``.  ``True`` marks positions to exclude.
        need_weights : bool
            If ``True``, also return attention weights.
        attn_mask : Tensor, optional
            Shape ``(L, S)`` or ``(B·H, L, S)``.
            Boolean: ``True`` = mask out.
            Float: added to raw scores before nome mapping (-inf = mask out).
        average_attn_weights : bool
            Average weights over heads before returning (only relevant when
            ``need_weights=True``).
        is_causal : bool
            Apply an upper-triangular causal mask.

        Returns
        -------
        attn_output : Tensor
            Shape matching the *query* input.
        attn_output_weights : Tensor or None
            Attention weights, or ``None`` if ``need_weights=False``.
        """
        # ---- 0. Normalise layout to (B, L/S, D) -------------------------
        if self.batch_first:
            query, key, value = (
                query.transpose(0, 1),
                key.transpose(0, 1),
                value.transpose(0, 1),
            )
        # Now: query (L, B, D), key (S, B, D), value (S, B, D)
        # Transpose to (B, L/S, D)
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        B, L, _ = query.shape
        S = key.shape[1]
        H, Hd = self.num_heads, self.head_dim

        # ---- 1. Linear projections ----------------------------------------
        Q = self.q_proj(query)   # (B, L, D)
        K = self.k_proj(key)     # (B, S, D)
        V = self.v_proj(value)   # (B, S, D)

        # ---- 2. Optional bias_k / bias_v (add_bias_kv) --------------------
        if self.add_bias_kv:
            K = torch.cat([K, self.bias_k.expand(B, -1, -1)], dim=1)
            V = torch.cat([V, self.bias_v.expand(B, -1, -1)], dim=1)
            S = K.shape[1]
            if key_padding_mask is not None:
                # Extend mask with a non-masked column for the bias token
                key_padding_mask = F.pad(key_padding_mask, (0, 1), value=False)

        # ---- 3. Optional zero-attention row (add_zero_attn) ---------------
        if self.add_zero_attn:
            zeros = query.new_zeros(B, 1, self.embed_dim)
            K = torch.cat([K, zeros], dim=1)
            V = torch.cat([V, zeros], dim=1)
            S = K.shape[1]
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1), value=False)

        # ---- 4. Reshape to (B, H, L/S, Hd) --------------------------------
        Q = Q.view(B, L, H, Hd).transpose(1, 2)   # (B, H, L, Hd)
        K = K.view(B, S, H, Hd).transpose(1, 2)   # (B, H, S, Hd)
        V = V.view(B, S, H, Hd).transpose(1, 2)   # (B, H, S, Hd)

        # ---- 5. Raw dot-product scores -------------------------------------
        # scores shape: (B, H, L, S)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self._scale

        # ---- 6. Additive float attn_mask → also build boolean mask ---------
        # bool_mask: True = position is masked out (kernel value → 0)
        bool_mask = query.new_zeros(B, H, L, S, dtype=torch.bool)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                # Boolean mask: broadcast to (B, H, L, S)
                bool_mask = bool_mask | _broadcast_attn_mask(attn_mask, B, H, L, S)
            else:
                # Float mask: add to raw scores AND mark extreme negatives
                fm = _broadcast_attn_mask(attn_mask, B, H, L, S)
                scores = scores + fm
                bool_mask = bool_mask | (fm < -1e8)

        # ---- 7. Causal mask -----------------------------------------------
        if is_causal:
            causal = torch.triu(
                query.new_ones(L, S, dtype=torch.bool), diagonal=1
            )
            bool_mask = bool_mask | causal.unsqueeze(0).unsqueeze(0)

        # ---- 8. Key-padding mask -------------------------------------------
        if key_padding_mask is not None:
            # (B, S) → (B, 1, 1, S)
            bool_mask = bool_mask | key_padding_mask.view(B, 1, 1, S)

        # ---- 9. Map scores → nome q ∈ (0, 1) ------------------------------
        # q = sigmoid(s / τ)  ≡  e^{iπτ̃}  with τ̃ = i·log(1/q)/π ∈ iℝ₊
        # scores are already normalised by √d; τ controls the operating region.
        tau = self.temperature          # positive scalar tensor (learnable)
        q = torch.sigmoid(scores / tau)

        # ---- 10. Evaluate 3rd-order Taylor kernel K(q) --------------------
        # K(q) = a₀ + a₁q + a₂q² + a₃q³  via Horner's method
        kernel_values = _horner_eval(q, self.kernel)   # (B, H, L, S), ≥ 0

        # ---- 11. Hard-zero masked positions --------------------------------
        # (Additive masking doesn't zero out q-series kernels; explicit
        # zeroing is the correct approach for this family of kernels.)
        if bool_mask.any():
            kernel_values = kernel_values.masked_fill(bool_mask, 0.0)

        # ---- 12. Normalise → attention weights ----------------------------
        weights = kernel_values / (kernel_values.sum(dim=-1, keepdim=True) + 1e-9)

        # ---- 13. Dropout --------------------------------------------------
        if self.training and self.dropout_p > 0.0:
            weights = self._dropout(weights)

        # ---- 14. Aggregate values -----------------------------------------
        out = torch.matmul(weights, V)                   # (B, H, L, Hd)
        out = out.transpose(1, 2).contiguous().view(B, L, self.embed_dim)
        out = self.out_proj(out)                         # (B, L, D)

        # ---- 15. Restore layout -------------------------------------------
        # Back to (L, B, D) and then handle batch_first
        out = out.transpose(0, 1)   # (L, B, D)
        if self.batch_first:
            out = out.transpose(0, 1)   # (B, L, D)

        # ---- 16. Return weights if requested ------------------------------
        if need_weights:
            # weights: (B, H, L, S)
            if average_attn_weights:
                ret_weights = weights.mean(dim=1)   # (B, L, S)
            else:
                ret_weights = weights               # (B, H, L, S)
            return out, ret_weights

        return out, None

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"kernel='{self.kernel}', "
            f"temperature={self.temperature.item():.4f}, "
            f"batch_first={self.batch_first}"
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _broadcast_attn_mask(
    mask: torch.Tensor,
    B: int,
    H: int,
    L: int,
    S: int,
) -> torch.Tensor:
    """Broadcast an attn_mask of shape (L, S) or (B*H, L, S) to (B, H, L, S)."""
    if mask.dim() == 2:
        # (L, S) → (1, 1, L, S) → broadcast
        return mask.unsqueeze(0).unsqueeze(0).expand(B, H, L, S)
    if mask.dim() == 3:
        # (B*H, L, S) → (B, H, L, S)
        return mask.view(B, H, L, S)
    raise ValueError(
        f"attn_mask must be 2-D (L, S) or 3-D (B*H, L, S), got shape {tuple(mask.shape)}"
    )
