"""
experiments/exp5_tomography_degradation.py
==========================================
Experiment 5: Finite-Shot Tomography Degradation and PCA Stability.

Research Question:
How does finite-shot statistical noise (N in {100, 1000, 10000, inf}) degrade
quantum state reconstruction, and how stable are principal component scores
and inferred entanglement manifolds under realistic measurement budgets?

Theory:
1. Quantum Shot Noise: In projective Pauli measurements, empirical frequencies
   converge at rate sigma_obs = O(1 / sqrt(N)).
2. State Estimation Error: The Frobenius reconstruction error ||rho - rho_hat||_F
   scales as O(1 / sqrt(N)).
3. Concurrence Bias: Because Wootters concurrence involves eigenvalue difference
   and a non-linear maximum with 0, finite shot noise introduces a positive bias
   near separable states (apparent entanglement from noise) and negative bias
   near maximally entangled states.
4. PCA Robustness: Because PCA projects onto the leading variance eigenvectors,
   uncorrelated shot noise in 15 Pauli observables is distributed across all 15 modes,
   making PC1 and PC2 relatively resilient filters against finite-shot noise!
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
    compute_all_measures,
    concurrence,
)
from src.generators import (
    bell_state,
    parametrized_pure_state,
    werner_state,
    generate_ensemble,
)
from src.tomography import simulate_pauli_tomography
from src.pca import ManualPCA


def run_experiment_5(n_trials: int = 25, seed: int = 42):
    rng = np.random.default_rng(seed)

    # 1. Fit reference PCA on ground-truth ensemble
    ref_ensemble = generate_ensemble(seed=seed)
    X_ref = np.array([feature_vector_set_a(item[1]) for item in ref_ensemble])
    pca = ManualPCA(standardize=True).fit(X_ref)

    shot_budgets = [100, 300, 1000, 3000, 10000, np.inf]
    shot_labels = ["100", "300", "1k", "3k", "10k", "Exact"]

    # Target test states
    test_states = [
        ("Bell_Phi+", bell_state("phi_plus")),
        ("Param_Entangled", parametrized_pure_state(np.pi / 6.0)),
        ("Werner_p07", werner_state(0.70)),
        ("Werner_p04", werner_state(0.40)),
        ("Product_00", parametrized_pure_state(0.0)),
    ]

    frob_errors_by_shots = {s: [] for s in shot_labels}
    pc_displacements_by_shots = {s: [] for s in shot_labels}
    concurrence_biases_by_shots = {s: [] for s in shot_labels}

    scatter_cloud = []

    for label_s, shots in zip(shot_labels, shot_budgets):
        shots_val = None if np.isinf(shots) else int(shots)

        for name, rho_true in test_states:
            c_true = concurrence(rho_true)
            z_true = pca.transform(feature_vector_set_a(rho_true).reshape(1, -1))[0]

            for _ in range(n_trials):
                _, rho_rec = simulate_pauli_tomography(rho_true, shots=shots_val, rng=rng)

                frob_err = float(np.linalg.norm(rho_true - rho_rec, ord="fro"))
                c_rec = concurrence(rho_rec)
                c_bias = float(c_rec - c_true)

                feat_rec = feature_vector_set_a(rho_rec)
                z_rec = pca.transform(feat_rec.reshape(1, -1))[0]
                pc_disp = float(np.linalg.norm(z_rec - z_true))

                frob_errors_by_shots[label_s].append(frob_err)
                concurrence_biases_by_shots[label_s].append(c_bias)
                pc_displacements_by_shots[label_s].append(pc_disp)

                if name == "Bell_Phi+" and len(scatter_cloud) < 300:
                    scatter_cloud.append({
                        "shots": label_s,
                        "z1": float(z_rec[0]),
                        "z2": float(z_rec[1]),
                        "concurrence": float(c_rec),
                    })

    # Average metrics
    mean_frob = [float(np.mean(frob_errors_by_shots[s])) for s in shot_labels]
    std_frob = [float(np.std(frob_errors_by_shots[s])) for s in shot_labels]
    mean_disp = [float(np.mean(pc_displacements_by_shots[s])) for s in shot_labels]

    # Plot 1: Publication figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), dpi=150)

    # Panel 1: Frobenius Reconstruction Error vs Shots (Log-Log)
    finite_shots = [100, 300, 1000, 3000, 10000]
    finite_errs = mean_frob[:5]

    axes[0].errorbar(
        finite_shots, finite_errs,
        yerr=std_frob[:5],
        fmt="o-", color="#D93025", lw=2.2, capsize=4,
        label="Observed Frobenius Error $\\|\\rho - \\hat{\\rho}\\|_{F}$"
    )

    # Theoretical 1/sqrt(N) guide curve
    scale = finite_errs[0] * np.sqrt(finite_shots[0])
    guide = [scale / np.sqrt(n) for n in finite_shots]
    axes[0].plot(finite_shots, guide, "--", color="#5F6368", lw=1.8, label="Quantum Shot Noise Limit $\\propto 1/\\sqrt{N}$")

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Measurement Shots per Pauli Basis ($N$)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("State Reconstruction Error $\\|\\rho - \\hat{\\rho}\\|_{F}$", fontsize=11, fontweight="bold")
    axes[0].set_title("Tomography Scaling: Shot Noise Convergence", fontsize=12, fontweight="bold")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[0].legend(loc="best")

    # Panel 2: Latent Score Cloud for Bell State under Finite Shots
    colors = {"100": "#EA4335", "300": "#FBBC04", "1k": "#34A853", "10k": "#4285F4", "Exact": "#188038"}
    for label_s in ["100", "1k", "10k", "Exact"]:
        sub = [item for item in scatter_cloud if item["shots"] == label_s]
        if sub:
            xs = [item["z1"] for item in sub]
            ys = [item["z2"] for item in sub]
            axes[1].scatter(xs, ys, color=colors.get(label_s, "#1A73E8"), alpha=0.7, s=35, label=f"N = {label_s}")

    axes[1].set_xlabel("PC1 (Entanglement Axis)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("PC2 (Mixedness Axis)", fontsize=11, fontweight="bold")
    axes[1].set_title("PCA Latent Stability for Bell State under Finite Shots", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="best")

    plt.tight_layout()
    os.makedirs("reports/figures", exist_ok=True)
    fig_path = "reports/figures/exp5_tomography_degradation.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    results = {
        "shot_labels": shot_labels,
        "mean_frob_errors": mean_frob,
        "std_frob_errors": std_frob,
        "mean_pc_displacements": mean_disp,
        "mean_concurrence_biases": {s: float(np.mean(concurrence_biases_by_shots[s])) for s in shot_labels},
        "figure_saved": fig_path,
        "scatter_cloud_sample": scatter_cloud[:60],
    }

    print(f"[Exp 5] Finished. N=100 error: {mean_frob[0]:.4f} -> N=10k error: {mean_frob[4]:.4f}.")
    return results


if __name__ == "__main__":
    res = run_experiment_5()
    with open("reports/exp5_results.json", "w") as f:
        json.dump(res, f, indent=2)
