"""
experiments/exp4_raw_vs_measures.py
===================================
Experiment 4: Raw Pauli Observables vs Non-Linear Quantum Measures Representation.

Research Question:
Can PCA applied to raw Pauli expectation values (Feature Set B, 15 observables)
naturally isolate entanglement into principal axes without explicit non-linear formulas,
and how does it compare to PCA on targeted quantum measures (Feature Set A, 9 measures)?

Physical & Mathematical Foundations:
- Feature Set B is linear in density matrix rho: <O_k> = Tr(rho O_k).
  Because the Bloch vector representation rho = 1/4 (I + sum c_k O_k) is an exact
  isometry under the Hilbert-Schmidt inner product Tr(A^dagger B), PCA on Set B
  is simply finding the principal directions of variation in the state's Bloch/correlation space.
- Entanglement is intrinsically NON-LINEAR: the separable state set Sep is a convex body
  with non-linear algebraic boundaries (e.g. det(rho^TB) < 0, Wootters discriminant).
- Linear combinations of Pauli operators can only define hyperplane witnesses for specific
  subsets of states, never a universal entanglement coordinate.
- In contrast, Feature Set A feeds the non-linear spectral properties directly to PCA,
  enabling a 1D-to-2D manifold collapse where PC1 and PC2 directly correspond to
  Entanglement and Mixedness.
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
from src.generators import generate_ensemble
from src.pca import ManualPCA


def run_experiment_4(seed: int = 42):
    ensemble = generate_ensemble(seed=seed)
    states = [item[1] for item in ensemble]

    X_A = np.array([feature_vector_set_a(r) for r in states])
    X_B = np.array([feature_vector_set_b(r) for r in states])

    concurrences = np.array([compute_all_measures(r)["concurrence"] for r in states])
    purities = np.array([compute_all_measures(r)["purity"] for r in states])

    # Fit PCA on both representations (standardized)
    pca_A = ManualPCA(standardize=True).fit(X_A)
    pca_B = ManualPCA(standardize=True).fit(X_B)

    Z_A = pca_A.transform(X_A)
    Z_B = pca_B.transform(X_B)

    evr_A = pca_A.explained_variance_ratio_
    evr_B = pca_B.explained_variance_ratio_

    # Correlation between PC1/PC2 and Concurrence
    corr_A_pc1_c = float(np.corrcoef(Z_A[:, 0], concurrences)[0, 1])
    corr_A_pc2_c = float(np.corrcoef(Z_A[:, 1], concurrences)[0, 1])
    corr_B_pc1_c = float(np.corrcoef(Z_B[:, 0], concurrences)[0, 1])
    corr_B_pc2_c = float(np.corrcoef(Z_B[:, 1], concurrences)[0, 1])

    # Plot 1: Side-by-side Scree comparison & Loading Heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=150)

    # Panel (0, 0): Scree Plot Comparison
    axes[0, 0].plot(np.arange(1, len(evr_A) + 1), evr_A * 100, marker="o", color="#1A73E8", lw=2.2, label=f"Set A (Measures, 9 feats, Top 2={np.sum(evr_A[:2])*100:.1f}%)")
    axes[0, 0].plot(np.arange(1, len(evr_B) + 1), evr_B * 100, marker="s", color="#D93025", lw=2.0, label=f"Set B (Raw Paulis, 15 feats, Top 2={np.sum(evr_B[:2])*100:.1f}%)")
    axes[0, 0].set_xlabel("Principal Component Index", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Explained Variance (%)", fontsize=11, fontweight="bold")
    axes[0, 0].set_title("Scree Comparison: Non-Linear Measures vs Raw Pauli Observables", fontsize=12, fontweight="bold")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend(loc="best")

    # Panel (0, 1): Loading Heatmap for Set A (Top 3 PCs)
    loadings_A = pca_A.components_[:3, :]
    im_a = axes[0, 1].imshow(loadings_A, cmap="coolwarm", aspect="auto", vmin=-0.6, vmax=0.6)
    axes[0, 1].set_xticks(np.arange(len(FEATURE_SET_A_NAMES)))
    axes[0, 1].set_xticklabels(FEATURE_SET_A_NAMES, rotation=45, ha="right", fontsize=9)
    axes[0, 1].set_yticks([0, 1, 2])
    axes[0, 1].set_yticklabels(["PC1", "PC2", "PC3"], fontweight="bold")
    axes[0, 1].set_title("PCA Loadings Heatmap: Feature Set A (Measures)", fontsize=12, fontweight="bold")
    plt.colorbar(im_a, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Panel (1, 0): Set A Latent Space colored by Concurrence
    sc_a = axes[1, 0].scatter(Z_A[:, 0], Z_A[:, 1], c=concurrences, cmap="plasma", s=25, alpha=0.85)
    cb_a = plt.colorbar(sc_a, ax=axes[1, 0])
    cb_a.set_label("Concurrence", fontweight="bold")
    axes[1, 0].set_xlabel(f"PC1 ({evr_A[0]*100:.1f}%)", fontweight="bold")
    axes[1, 0].set_ylabel(f"PC2 ({evr_A[1]*100:.1f}%)", fontweight="bold")
    axes[1, 0].set_title(f"Set A Projection (|corr(PC1, C)| = {abs(corr_A_pc1_c):.3f})", fontsize=12, fontweight="bold")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Panel (1, 1): Set B Latent Space colored by Concurrence
    sc_b = axes[1, 1].scatter(Z_B[:, 0], Z_B[:, 1], c=concurrences, cmap="plasma", s=25, alpha=0.85)
    cb_b = plt.colorbar(sc_b, ax=axes[1, 1])
    cb_b.set_label("Concurrence", fontweight="bold")
    axes[1, 1].set_xlabel(f"PC1 ({evr_B[0]*100:.1f}%)", fontweight="bold")
    axes[1, 1].set_ylabel(f"PC2 ({evr_B[1]*100:.1f}%)", fontweight="bold")
    axes[1, 1].set_title(f"Set B Projection (|corr(PC1, C)| = {abs(corr_B_pc1_c):.3f})", fontsize=12, fontweight="bold")
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    os.makedirs("reports/figures", exist_ok=True)
    fig_path = "reports/figures/exp4_raw_vs_measures.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    results = {
        "evr_set_a": evr_A.tolist(),
        "evr_set_b": evr_B.tolist(),
        "top2_var_a": float(np.sum(evr_A[:2])),
        "top2_var_b": float(np.sum(evr_B[:2])),
        "corr_A_pc1_c": corr_A_pc1_c,
        "corr_A_pc2_c": corr_A_pc2_c,
        "corr_B_pc1_c": corr_B_pc1_c,
        "corr_B_pc2_c": corr_B_pc2_c,
        "loadings_set_a": {
            f"PC{k+1}": {name: float(loadings_A[k, i]) for i, name in enumerate(FEATURE_SET_A_NAMES)}
            for k in range(3)
        },
        "loadings_set_b_pc1": {name: float(pca_B.components_[0, i]) for i, name in enumerate(PAULI_15_NAMES)},
        "figure_saved": fig_path,
        "scores_a_sample": Z_A[:60, :2].tolist(),
        "scores_b_sample": Z_B[:60, :2].tolist(),
        "concurrence_sample": concurrences[:60].tolist(),
    }

    print(f"[Exp 4] Finished. Top 2 EVR: Set A = {np.sum(evr_A[:2])*100:.2f}%, Set B = {np.sum(evr_B[:2])*100:.2f}%.")
    return results


if __name__ == "__main__":
    res = run_experiment_4()
    with open("reports/exp4_results.json", "w") as f:
        json.dump(res, f, indent=2)
