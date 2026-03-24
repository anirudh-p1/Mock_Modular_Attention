#!/usr/bin/env python3
"""
Mock Modular Attention — prototype demonstration
=================================================

Shows three things:

1. Q-series / mock theta mathematics (convergence, function shapes)
2. MockModularAttention as a drop-in for nn.MultiheadAttention
3. Mini Transformer training on a synthetic copy task
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mock_modular_attention import (
    MockModularAttention,
    MockModularTransformer,
    mock_theta_f,
    mock_theta_omega,
    mock_theta_phi,
    q_series_kernel,
    mock_modular_weights,
    build_kernel,
)


# ---------------------------------------------------------------------------
# Section 1 – Q-series mathematics
# ---------------------------------------------------------------------------

def demo_q_series() -> None:
    print("=" * 64)
    print("Section 1 — Q-Series / Mock Theta Mathematics")
    print("=" * 64)

    q_vals = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
    fmt = lambda t: [f"{x:.4f}" for x in t.tolist()]

    print(f"\nq values         : {fmt(q_vals)}")
    print(f"θ₃(q)  [q_series]: {fmt(q_series_kernel(q_vals, num_terms=8))}")
    print(f"f(q)   [mock_θ_f]: {fmt(mock_theta_f(q_vals, num_terms=8))}")
    print(f"ω(q)   [mock_θ_ω]: {fmt(mock_theta_omega(q_vals, num_terms=8))}")
    print(f"|φ(q)| [mock_θ_φ]: {fmt(mock_theta_phi(q_vals, num_terms=8))}")

    # Taylor-kernel values (3rd-order, used in MockModularAttention)
    print("\n3rd-order Taylor kernel K(q) = a₀ + a₁q + a₂q² + a₃q³:")
    for name in ["mock_theta_f", "mock_theta_omega", "mock_theta_phi", "q_series"]:
        w = mock_modular_weights(q_vals.unsqueeze(0), kernel=name)[0]
        print(f"  {name:<22}: weights = {fmt(w)}")

    # Convergence of mock_theta_f at q = 0.7
    print("\nConvergence of mock_theta_f(q=0.7) as N increases:")
    q = torch.tensor([0.7])
    for N in [1, 2, 4, 6, 8, 12]:
        print(f"  N={N:2d}: {mock_theta_f(q, num_terms=N).item():.8f}")


# ---------------------------------------------------------------------------
# Section 2 – Drop-in replacement for nn.MultiheadAttention
# ---------------------------------------------------------------------------

def demo_drop_in() -> None:
    print("\n" + "=" * 64)
    print("Section 2 — Drop-in Replacement for nn.MultiheadAttention")
    print("=" * 64)

    torch.manual_seed(42)
    B, L, D, H = 2, 10, 64, 4

    # Identical call sites — only the class name changes
    mha = nn.MultiheadAttention(D, H, batch_first=True)
    mma = MockModularAttention(D, H, batch_first=True, kernel="mock_theta_f")

    q = torch.randn(B, L, D)
    causal = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)

    with torch.no_grad():
        out_mha, w_mha = mha(q, q, q, attn_mask=causal)
        out_mma, w_mma = mma(q, q, q, attn_mask=causal)

    print(f"\nInput  : (B={B}, L={L}, D={D})")
    print(f"nn.MultiheadAttention  output: {tuple(out_mha.shape)}")
    print(f"MockModularAttention   output: {tuple(out_mma.shape)}")
    print(f"Weights match shape?  {w_mha.shape == w_mma.shape}")

    # Show attention weight distributions differ (MMA has different inductive bias)
    row = w_mha[0, 5]          # MHA weights for query position 5
    row_mma = w_mma[0, 5]      # MMA weights for query position 5
    fmt = lambda t: [f"{x:.3f}" for x in t.tolist()]
    print(f"\nAttention weights at query pos 5 (first 6 keys):")
    print(f"  nn.MHA  : {fmt(row[:6])}")
    print(f"  MMA(f)  : {fmt(row_mma[:6])}")

    # Compare all four MMA kernels
    print("\nMMA kernel comparison (average attention entropy per head):")
    for kernel in ["mock_theta_f", "mock_theta_omega", "mock_theta_phi", "q_series"]:
        m = MockModularAttention(D, H, batch_first=True, kernel=kernel)
        with torch.no_grad():
            _, w = m(q, q, q, attn_mask=causal,
                     need_weights=True, average_attn_weights=False)
        entropy = -(w * (w + 1e-9).log()).sum(-1).mean().item()
        print(f"  {kernel:<22}: H = {entropy:.4f} nats")


# ---------------------------------------------------------------------------
# Section 3 – Learnable kernel modules
# ---------------------------------------------------------------------------

def demo_kernel_modules() -> None:
    print("\n" + "=" * 64)
    print("Section 3 — Learnable Mock Theta Kernel Modules")
    print("=" * 64)

    kernel = build_kernel("mock_theta_f", num_terms=8, init_temperature=1.0)
    print(f"\nKernel: {kernel}")
    print(f"Learnable parameters: {sum(p.numel() for p in kernel.parameters())}")
    print(f"Initial temperature τ = {kernel.temperature.item():.4f}")

    # Forward + backward
    scores = torch.randn(2, 4, 8, 8)
    weights = kernel(scores)
    loss = weights.sum()
    loss.backward()
    print(f"After backward — τ grad: {kernel.tau_raw.grad.item():.6f}")


# ---------------------------------------------------------------------------
# Section 4 – Mini Transformer training
# ---------------------------------------------------------------------------

def demo_training() -> None:
    print("\n" + "=" * 64)
    print("Section 4 — Mini Transformer Training (Copy Task)")
    print("=" * 64)

    torch.manual_seed(0)
    VOCAB, SEQ = 16, 10
    model = MockModularTransformer(
        vocab_size=VOCAB, seq_len=SEQ,
        embed_dim=64, num_heads=4, num_layers=2, ff_dim=128,
        kernel="mock_theta_f", dropout=0.0,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(f"Kernel:           mock_theta_f (3rd-order Taylor)")
    print(f"Vocab / SeqLen:   {VOCAB} / {SEQ}\n")

    optim = torch.optim.Adam(model.parameters(), lr=2e-3)
    first_loss = None

    for step in range(300):
        tokens = torch.randint(0, VOCAB, (32, SEQ))
        logits = model(tokens)
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB),
            tokens[:, 1:].reshape(-1),
        )
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if first_loss is None:
            first_loss = loss.item()
        if (step + 1) % 60 == 0:
            print(f"  step {step+1:3d}: loss = {loss.item():.4f}")

    print(f"\n  Start loss: {first_loss:.4f}")
    print(f"  End   loss: {loss.item():.4f}")
    print(f"  Reduction:  {100*(first_loss - loss.item())/first_loss:.1f}%")
    print("\nTraining complete ✓")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_q_series()
    demo_drop_in()
    demo_kernel_modules()
    demo_training()
    print("\n✓  Mock Modular Attention prototype demo complete.")
