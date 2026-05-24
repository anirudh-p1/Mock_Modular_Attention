# Mock Modular Attention (MMA)

A mathematically grounded, production-grade alternative to standard Softmax attention. This repository replaces stochastic, unconstrained heuristic logits with deterministic kernel weightings derived from **Ramanujan's third-order mock theta functions** and classical $q$-series. 

MMA introduces an approximate modular-symmetry inductive bias directly into the forward pass of Transformer architectures.

---

## Mathematical Architecture & Core Mechanics

Standard Multihead Attention computes token alignment using the unconstrained exponential function:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V = \frac{\exp(s_{ij})}{\sum_j \exp(s_{ij})}V$$

This unconstrained formulation permits fluid weight fluctuations, allowing models to maximize optimization tracking via low-rank polysemantic superposition. 

MMA enforces hard geometric bounds on the attention function space across four distinct implemented series. For each pair of sequence positions $(i, j)$:

### 1. Mapping Logits to the Complex Unit Disk
Raw dot-product scores are scaled by a learnable temperature $\tau$ and mapped into the complex nome $q \in (0, 1)$ via a sigmoid activation:

$$q_{ij} = \sigma\left(\frac{s_{ij}}{\tau}\right) = \frac{1}{1 + e^{-s_{ij}/\tau}}$$

This mapping is mathematically equivalent to the complex coordinate $q = e^{i\pi\tilde{\tau}}$, where $\tilde{\tau}$ is purely imaginary and resides strictly within the upper half-plane:

$$\text{Im}(\tilde{\tau}) = \frac{\log(1/q)}{\pi} > 0$$

### 2. Truncated Taylor Expansions ($\mathcal{O}(1)$ Complexity)
To preserve standard Transformer scaling metrics, infinite $q$-series expansions are truncated to a 3rd-order Taylor series around $q = 0$. Evaluation is optimized via **Horner's Method**, requiring only 3 multiply-add operations per element ($\mathcal{O}(1)$ relative to polynomial degree):

$$K(q) = a_0 + q(a_1 + q(a_2 + q \cdot a_3))$$

---

## Implemented Kernels & Taylor Coefficients

Every implemented kernel is analytically verified to be non-negative across the bounded region $q \in [0, 1]$, satisfying probability simplex constraints upon normalization:

| Kernel Identifier | Mathematical Representation | Truncated Taylor Polynomial ($O(q^4)$) | Coefficient Invariants $[a_0, a_1, a_2, a_3]$ | Structural Inductive Bias |
| :--- | :--- | :--- | :--- | :--- |
| `mock_theta_f` | $\sum_{n \ge 0} \frac{q^{n^2}}{((-q; q)_n)^2}$ | $1 + q - 2q^2 + 3q^3$ | `[1.0, 1.0, -2.0, 3.0]` | **Principal Growth Regime:** Aligns weights along the torus geometry of the modular curve. |
| `mock_theta_omega` | $\sum_{n \ge 0} \frac{q^{2n(n+1)}}{((q; q^2)_{n+1})^2}$ | $1 + 2q + 3q^2 + 4q^3$ | `[1.0, 2.0, 3.0, 4.0]` | **Aggressive Attenuation:** Faster exponent growth forces high-order sparsity. |
| `mock_theta_phi` | $\sum_{n \ge 0} \frac{(-1)^n q^{n^2}}{(-q^2; q^2)_n}$ | $\|1 - q + 0q^2 + q^3\|$ | `[1.0, -1.0, 0.0, 1.0]` | **Alternating Parity:** Soft oscillatory regularizer favoring uniform distributions. |
| `q_series` | $\sum_{n \ge 0} q^{n^2}$ | $1 + q + 0q^2 + 0q^3$ | `[1.0, 1.0, 0.0, 0.0]` | **Canonical Positive form:** Exact classical Jacobi $\theta_3$ modular transformation. |

---

## AI Safety & Mechanistic Interpretability Implications

This research addresses the AI alignment problem at the **architectural level** by replacing uninterpretable empirical black boxes with mathematically constrained white boxes:

* **Collapsing Superposition Hiding Spaces:** By locking activation scaling into rigorous near-symmetric geometries, features remain isolated and mapped, preventing networks from forming fluid, untraceable circuit configurations.
* **Proactive Alignment & Circuit Auditing:** The predictable internal layouts governed by fixed algebraic invariants enable automated tools to reliably spot anomalous or adversarial logic transitions during runtime.
* **Out-of-Distribution (OOD) Generalization:** Grounding the attention kernel in bounded modular forms anchors the model’s reasoning loops, mitigating tail-end representation drift when handling OOD evaluations.

---

## Codebase Layout & Architecture

The repository is modularized for rapid integration and benchmarking:
* `qseries.py`: Pure-functional definitions of truncated $q$-Pochhammer calculations, series mappings, and analytical non-negativity parameters.
* `attention.py`: `MockModularAttention` — a robust, feature-complete **drop-in replacement** for `torch.nn.MultiheadAttention`.
* `transformer.py`: Multi-layer pre-norm causal sequence architecture (`MockModularTransformer`) built directly over MMA layers.

### Parameter Invariance
The scaling temperature $\tau$ is stored internally in log-space (`self.log_temperature`) to guarantee strict parameter positivity throughout backpropagation:

$$\tau = \exp(\text{log\_temperature}) > 0$$

---

## Quickstart Usage Example

Because `MockModularAttention` mimics the exact initialization and forward execution signatures of PyTorch's native multi-head layer, migrating an existing model requires altering exactly one line of code:

```python
import torch
from mock_modular_attention import MockModularAttention

# Fully compatible with standard PyTorch tensors and masks
batch_size, seq_len, embed_dim, num_heads = 16, 128, 64, 4
X = torch.rand(batch_size, seq_len, embed_dim)

# Instantiate MMA layer with specialized Ramanujan kernel selection
mma_layer = MockModularAttention(
    embed_dim=embed_dim,
    num_heads=num_heads,
    batch_first=True,
    kernel="mock_theta_f",      # Options: mock_theta_f, mock_theta_omega, mock_theta_phi, q_series
    init_temperature=1.0        # Learnable scaling initialization
)

# Forward pass natively supports causal masks and padding masks
output, attn_weights = mma_layer(query=X, key=X, value=X, need_weights=True)
print(output.shape)  # Torch.Size([16, 128, 64])
print(attn_weights.shape)  # Torch.Size([16, 128, 128])
