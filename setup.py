from setuptools import setup, find_packages

setup(
    name="mock-modular-attention",
    version="0.1.0",
    description=(
        "Mock Modular Attention: Q-Series Weighted Attention as an "
        "Approximate Symmetry Inductive Bias for Sequential Learning"
    ),
    author="Anirudh Prabhu",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=["torch>=1.13.0"],
    extras_require={"dev": ["pytest>=7.0.0"]},
)
