"""
experiments/exp2_mixed_redundancy.py
====================================
Experiment 2: Mixed States and Entanglement Redundancy Breaking.

Research Question:
How does mixedness (state rank r in {1, 2, 3, 4} and purity gamma in [0.25, 1.0])
alter the low-dimensional structure found in pure states?

Theoretical Expectation:
In pure states (rank 1), S_AB = 0, Purity = 1, and S_A = S_B are strictly coupled,
collapsing measures to 1 effective dimension.
When mixed states are admitted:
1. Joint entropy S_AB > 0 provides an orthogonal degree of freedom.
2. Purity drops toward 0.25.
3. Subsystem entropies decouple: S_A != S_B in generic mixed states.
4. Concurrence no longer uniquely determines E_F or Negativity (bounds widen).
5. Effective dimensionality increases from ~1 to 3+, breaking measure redundancy.
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
    FEATURE_SET_A_NAMES,
    compute_all_measures,
)
from src.generators import (
    haar_random_pure_state,
    werner_state,
    random_mixed_state,
)
from src.pca import ManualPCA


def compute_effective_dimension(evals: np.ndarray) -> float:
    """
    Computes participation ratio / effective dimension of an eigenvalue spectrum:
    D_eff = (sum lambda_i)^2 / sum(lambda_i^2).
    """
    pos = evals[evals > 1e-12]
    if len(pos) == 0:
        return 0.0
    return float((np.sum(pos) ** 2) / np.sum(pos ** 2))


def run_experiment_2(
    n_pure: int = 150,
    n_werner: int = 60,
    n_r2: int = 100,
    n_r3: int = 100,
    n_r4: int = 150,
    seed: int = 42
):
    rng = np.random.default_rng(seed)

    states = []
    classes = []
    purity_vals = []
    concurrence_vals = []
    ranks = []

    # 1. Pure (Rank 1)
    for _ in range(n_pure):
        rho = haar_random_pure_state(rng)
        states.append(rho)
        classes.append("Pure (Rank 1)")
        ranks.append(1)

    # 2. Werner states (variable purity and concurrence)
    for p in np.linspace(0.0, 1.0, n_werner):
        rho = werner_state(p)
        states.append(rho)
        classes.append("Werner")
        ranks.append(4)

    # 3. Random mixed Rank 2
    for _ in range(n_r2):
        rho = random_mixed_state(rank=2, rng=rng)
        states.append(rho)
        classes.append("Mixed (Rank 2)")
        ranks.append(2)

    # 4. Random mixed Rank 3
    for _ in range(n_r3):
        rho = random_mixed_state(rank=3, rng=rng)
        states.append(rho)
        classes.append("Mixed (Rank 3)")
        ranks.append(3)

    # 5. Random mixed Rank 4
    for _ in range(n_r4):
        rho = random_mixed_state(rank=4, rng=rng)
        states.append(rho)
        classes.append("Mixed (Rank 4)")
        ranks.append(4)

    for r in states:
        m = compute_all_measures(r)
        purity_vals.append(m["purity"])
        concurrence_vals.append(m["concurrence"])

    purity_vals = np.array(purity_vals)
    concurrence_vals = np.array(concurrence_vals)
    ranks = np.array(ranks)

    # Feature matrix Set A
    X_A = np.array([feature_vector_set_a(r) for r in states])

    # Fit PCA across all states
    pca_all = ManualPCA(standardize=True)
    Z_all = pca_all.fit_transform(X_A)
    evr_all = pca_all.explained_variance_ratio_

    # Fit PCA on Pure states only for comparison
    X_pure = X_A[np.array(classes) == "Pure (Rank 1)"]
    pca_pure = ManualPCA(standardize=True).fit(X_pure)
    evr_pure = pca_pure.explained_variance_ratio_

    # Fit PCA on Rank 4 states only
    X_r4 = X_A[np.array(classes) == "Mixed (Rank 4)"]
    pca_r4 = ManualPCA(standardize=True).fit(X_r4)
    evr_r4 = pca_r4.explained_variance_ratio_

    # Effective dimension comparison
    d_eff_pure = compute_effective_dimension(pca_pure.eigenvalues_)
    d_eff_all = compute_effective_dimension(pca_all.eigenvalues_)
    d_eff_r4 = compute_effective_dimension(pca_r4.eigenvalues_)

    # Figure generation
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=150)

    # Panel 1: Scree Spectrum Comparison (Pure vs Rank 4 vs Full Ensemble)
    k_range = np.arange(1, len(evr_all) + 1)
    axes[0].plot(k_range, evr_pure * 100, marker="o", color="#1A73E8", lw=2.2, label=f"Pure (D_eff={d_eff_pure:.2f})")
    axes[0].plot(k_range, evr_r4 * 100, marker="s", color="#D93025", lw=2.0, label=f"Rank 4 Mixed (D_eff={d_eff_r4:.2f})")
    axes[0].plot(k_range, evr_all * 100, marker="^", color="#0D9488", lw=2.2, label=f"Full Ensemble (D_eff={d_eff_all:.2f})")
    axes[0].set_xlabel("Principal Component", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Explained Variance (%)", fontsize=11, fontweight="bold")
    axes[0].set_title("Eigenvalue Spectrum Broadening with Mixedness", fontsize=12, fontweight="bold")
    axes[0].set_xticks(k_range)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="best")

    # Panel 2: Latent 2D Projection colored by Purity
    sc1 = axes[1].scatter(Z_all[:, 0], Z_all[:, 1], c=purity_vals, cmap="viridis", s=25, alpha=0.8)
    cb1 = plt.colorbar(sc1, ax=axes[1])
    cb1.set_label("Purity $\\gamma = \\mathrm{Tr}(\\rho^2)$", fontsize=10, fontweight="bold")
    axes[1].set_xlabel(f"PC1 ({evr_all[0]*100:.1f}%)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel(f"PC2 ({evr_all[1]*100:.1f}%)", fontsize=11, fontweight="bold")
    axes[1].set_title("Latent Space: Mixedness / Entropy Axis", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Latent 2D Projection colored by Concurrence
    sc2 = axes[2].scatter(Z_all[:, 0], Z_all[:, 1], c=concurrence_vals, cmap="magma", s=25, alpha=0.8)
    cb2 = plt.colorbar(sc2, ax=axes[2])
    cb2.set_label("Concurrence $C(\\rho)$", fontsize=10, fontweight="bold")
    axes[2].set_xlabel(f"PC1 ({evr_all[0]*100:.1f}%)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel(f"PC2 ({evr_all[1]*100:.1f}%)", fontsize=11, fontweight="bold")
    axes[2].set_title("Latent Space: Entanglement Axis", fontsize=12, fontweight="bold")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs("reports/figures", exist_ok=True)
    fig_path = "reports/figures/exp2_mixed_redundancy.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    results = {
        "d_eff_pure": d_eff_pure,
        "d_eff_rank4": d_eff_r4,
        "d_eff_full_ensemble": d_eff_all,
        "evr_full_ensemble": evr_all.tolist(),
        "evr_pure": evr_pure.tolist(),
        "evr_rank4": evr_r4.tolist(),
        "loadings_pc1": {name: float(pca_all.components_[0, i]) for i, name in enumerate(FEATURE_SET_A_NAMES)},
        "loadings_pc2": {name: float(pca_all.components_[1, i]) for i, name in enumerate(FEATURE_SET_A_NAMES)},
        "figure_saved": fig_path,
        "scores_sample": Z_all[:80, :2].tolist(),
        "purity_sample": purity_vals[:80].tolist(),
        "concurrence_sample": concurrence_vals[:80].tolist(),
        "classes_sample": classes[:80],
    }

    print(f"[Exp 2] Finished. Pure D_eff: {d_eff_pure:.2f} -> Full D_eff: {d_eff_all:.2f}.")
    return results


if __name__ == "__main__":
    res = run_experiment_2()
    with open("reports/exp2_results.json", "w") as f:
        json.dump(res, f, indent=2)
