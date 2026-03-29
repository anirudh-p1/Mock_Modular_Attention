# Mock-Modular-Attention

Mock Modular Attention: Q-Series Weighted Attention as an Approximate Symmetry Inductive Bias for Sequential Learning

Research using a novel transformer attention mechanism that replaces standard heuristic weighting with an inductive bias derived from q-series and non-holomorphic mock theta functions. Whilst contemporary architectures rely on stochastic data-driven correlations, this approach embeds approximate modular symmetries into the attention layer. This research aims to transition AI architectures from uninterpretable black boxes to mathematically grounded white boxes.

This project focuses on managing and mitigating the existential risks associated with misaligned AGI, by ensuring that sequential learning is governed by mathematical constraints rather than arbitrary weight fluctuations. Beyond safety, the implementation leverages truncated Taylor series approximations to maintain computational efficiency, leading towards robust, interpretable and high-performance sequential modelling, which is especially useful in sensitive domains like predictive healthcare and automated reasoning.

Built in 6-hours (Post EV rejection)

Replaced Softmax with O(1) Mock Modular Kernels (f,ω,ϕ) per kernel operation via Horner’s Method to enable structural interpretability in Transformers.

Status: It works! (Functional Prototype)

# Other:

Horner’s Method Optimization: Reducing polynomial evaluation to O(1) constant-time complexity per kernel operation, ensuring MMA remains competitive with standard attention.

Functional PyTorch Drop-in: A prototype that replaces MultiHeadAttention with modular kernels (f,ω,ϕ).

AI Safety & Interpretability
This research addresses the AI alignment problem by:
Reducing Model Opacity: Moving away from arbitrary weight fluctuations toward structured symmetries that facilitate mechanistic interpretability.
Proactive Alignment: Enabling automated circuit analysis to detect potential deceptive alignment by providing more "auditable" internal logic.
OOD Robustness: Utilizing the inductive biases of modular forms to improve performance on Out-of-Distribution data, critical for safety-critical domains like predictive healthcare and automated reasoning.

# Next Milestone: 
Large-scale benchmarking against standard Transformers on OOD datasets and a formal mechanistic interpretability audit using Sparse Autoencoders (SAEs).
