"""
experiments/exp1_pure_manifold.py
=================================
Experiment 1: Dimensionality and Redundancy of Pure Two-Qubit States.

Research Question:
Does PCA applied to standard quantum measures on pure states reveal a
1-dimensional manifold reflecting the single Schmidt entanglement parameter?

Theoretical Expectation:
For any pure state |psi> in H_A \otimes H_B, Schmidt decomposition gives:
|psi> = sqrt(lambda_0) |00>_S + sqrt(lambda_1) |11>_S, with lambda_0 + lambda_1 = 1.
Hence all measures:
- C = 2 * sqrt(lambda_0 * lambda_1)
- N = C / 2
- S_A = S_B = -lambda_0 log2 lambda_0 - lambda_1 log2 lambda_1
- I(A:B) = 2 S_A
- S_AB = 0
- Purity = 1
are 1D functions of a single scalar lambda_0 in [0, 0.5].
Therefore, PCA on standardized Set A will display massive PC1 dominance (>90-95% variance).
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.measures import (
    feature_vector_set_a,
    feature_vector_set_b,
    FEATURE_SET_A_NAMES,
    PAULI_15_NAMES,
    compute_all_measures,
)
from src.generators import (
    parametrized_pure_state,
    haar_random_pure_state,
)
from src.pca import ManualPCA


def run_experiment_1(n_haar: int = 300, n_param: int = 100, seed: int = 42):
    rng = np.random.default_rng(seed)

    states = []
    labels = []
    concurrences = []
    thetas = []

    # Parametrized Schmidt sweep theta in [0, pi/4]
    for th in np.linspace(0.0, np.pi / 4.0, n_param):
        rho = parametrized_pure_state(th, phi=0.0)
        states.append(rho)
        labels.append("Parametric")
        thetas.append(float(th))

    # Haar random pure states
    for _ in range(n_haar):
        rho = haar_random_pure_state(rng)
        states.append(rho)
        labels.append("Haar_Pure")
        thetas.append(None)

    # Feature matrices
    X_A = np.array([feature_vector_set_a(r) for r in states])
    X_B = np.array([feature_vector_set_b(r) for r in states])

    for r in states:
        m = compute_all_measures(r)
        concurrences.append(m["concurrence"])
    concurrences = np.array(concurrences)

    # Fit PCA on Feature Set A (standardized)
    pca_A = ManualPCA(standardize=True)
    Z_A = pca_A.fit_transform(X_A)

    evr_A = pca_A.explained_variance_ratio_
    cum_evr_A = pca_A.cumulative_variance_ratio_

    # Plot 1: Publication figure
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # Panel 1: Scree Plot
    comps = np.arange(1, len(evr_A) + 1)
    axes[0].bar(comps, evr_A * 100, color="#1A73E8", alpha=0.8, label="Individual %")
    axes[0].plot(comps, cum_evr_A * 100, marker="o", color="#0D9488", lw=2.2, label="Cumulative %")
    axes[0].set_xlabel("Principal Component Index", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Explained Variance (%)", fontsize=11, fontweight="bold")
    axes[0].set_title("Experiment 1: Scree Plot (Pure States, Set A)", fontsize=12, fontweight="bold")
    axes[0].set_xticks(comps)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="best")

    # Panel 2: PC1 vs PC2 scores colored by Concurrence
    scatter = axes[1].scatter(
        Z_A[:, 0], Z_A[:, 1],
        c=concurrences, cmap="plasma",
        s=30, alpha=0.85, edgecolors="none"
    )
    cbar = plt.colorbar(scatter, ax=axes[1])
    cbar.set_label("Wootters Concurrence $C(\\rho)$", fontsize=10, fontweight="bold")
    axes[1].set_xlabel(f"PC1 ({evr_A[0]*100:.1f}%)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel(f"PC2 ({evr_A[1]*100:.1f}%)", fontsize=11, fontweight="bold")
    axes[1].set_title("Pure State 1D Manifold Collapse in PC Space", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs("reports/figures", exist_ok=True)
    fig_path = "reports/figures/exp1_pure_manifold.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    results = {
        "n_samples": len(states),
        "pc1_explained_variance_ratio": float(evr_A[0]),
        "pc2_explained_variance_ratio": float(evr_A[1]),
        "explained_variance_ratios": evr_A.tolist(),
        "cumulative_variance_ratios": cum_evr_A.tolist(),
        "loadings_pc1": {name: float(pca_A.components_[0, i]) for i, name in enumerate(FEATURE_SET_A_NAMES)},
        "loadings_pc2": {name: float(pca_A.components_[1, i]) for i, name in enumerate(FEATURE_SET_A_NAMES)},
        "figure_saved": fig_path,
        "pc_scores_sample": Z_A[:50, :2].tolist(),
        "concurrences_sample": concurrences[:50].tolist(),
    }

    print(f"[Exp 1] Finished. PC1 captures {evr_A[0]*100:.2f}% of total variance on pure states.")
    return results


if __name__ == "__main__":
    res = run_experiment_1()
    with open("reports/exp1_results.json", "w") as f:
        json.dump(res, f, indent=2)
