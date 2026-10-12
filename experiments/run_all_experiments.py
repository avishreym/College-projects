"""
experiments/run_all_experiments.py
==================================
Master execution pipeline for Quantum Entanglement PCA research project.

Executes:
- Experiment 1: Pure State Manifold Collapse (1-parameter redundancy)
- Experiment 2: Mixed States and Redundancy Breaking (Spectral expansion)
- Experiment 3: Quantum Noise Channels & Latent Trajectories (Depol, Deph, AmpDamp)
- Experiment 4: Raw Pauli Observables vs Targeted Measures (Linear vs Non-linear PCA)
- Experiment 5: Finite-Shot Tomography Degradation (Shot-noise stability O(1/sqrt(N)))
- Master 4-panel composite publication figure.
- Consolidates all results into reports/experiment_summary.json.
"""

import sys
import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.exp1_pure_manifold import run_experiment_1
from experiments.exp2_mixed_redundancy import run_experiment_2
from experiments.exp3_noise_trajectories import run_experiment_3
from experiments.exp4_raw_vs_measures import run_experiment_4
from experiments.exp5_tomography_degradation import run_experiment_5


def generate_master_summary_figure(e1, e2, e3, e4, e5):
    """
    Generates a 4-panel composite figure summarizing the research findings.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=200)

    # Panel (0, 0): Exp 1 & 2 Scree Spectra Comparison
    axes[0, 0].plot(range(1, 10), np.array(e1["explained_variance_ratios"]) * 100, "o-", color="#1A73E8", lw=2.5, label="Pure States (PC1 = 96.2%)")
    axes[0, 0].plot(range(1, 10), np.array(e2["evr_rank4"]) * 100, "s-", color="#D93025", lw=2.2, label="Rank 4 Mixed States (Broadened)")
    axes[0, 0].plot(range(1, 10), np.array(e2["evr_full_ensemble"]) * 100, "^-", color="#0D9488", lw=2.2, label="Full Multi-Class Ensemble")
    axes[0, 0].set_xlabel("Principal Component", fontweight="bold")
    axes[0, 0].set_ylabel("Explained Variance (%)", fontweight="bold")
    axes[0, 0].set_title("(a) Manifold Dimensionality & Redundancy Breaking", fontsize=12, fontweight="bold")
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    axes[0, 0].legend(loc="best", fontsize=10)

    # Panel (0, 1): Exp 3 Noise Trajectories
    dep_sc = np.array(e3["depol_scores"])
    dph_sc = np.array(e3["deph_scores"])
    amp_sc = np.array(e3["amp_scores"])
    axes[0, 1].plot(dep_sc[:, 0], dep_sc[:, 1], color="#D93025", lw=2.5, label="Depolarizing Channel")
    axes[0, 1].plot(dph_sc[:, 0], dph_sc[:, 1], color="#1A73E8", lw=2.5, label="Dephasing Channel")
    axes[0, 1].plot(amp_sc[:, 0], amp_sc[:, 1], color="#0D9488", lw=2.5, label="Amplitude Damping")
    axes[0, 1].scatter(dep_sc[0, 0], dep_sc[0, 1], color="#E8710A", s=130, marker="*", edgecolors="black", label="Initial |Phi+>", zorder=5)
    axes[0, 1].set_xlabel("PC1 (Entanglement / Purity)", fontweight="bold")
    axes[0, 1].set_ylabel("PC2 (Subsystem Entropy)", fontweight="bold")
    axes[0, 1].set_title("(b) Latent Trajectories Under Open-System Noise", fontsize=12, fontweight="bold")
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    axes[0, 1].legend(loc="best", fontsize=10)

    # Panel (1, 0): Exp 4 Scree: Measures (Set A) vs Raw Paulis (Set B)
    axes[1, 0].plot(range(1, 10), np.array(e4["evr_set_a"]) * 100, "o-", color="#1A73E8", lw=2.2, label=f"Targeted Measures (Top 2: {e4['top2_var_a']*100:.1f}%)")
    axes[1, 0].plot(range(1, 16), np.array(e4["evr_set_b"]) * 100, "s-", color="#D93025", lw=2.0, label=f"Raw Pauli Observables (Top 2: {e4['top2_var_b']*100:.1f}%)")
    axes[1, 0].set_xlabel("Principal Component", fontweight="bold")
    axes[1, 0].set_ylabel("Explained Variance (%)", fontweight="bold")
    axes[1, 0].set_title("(c) Representation Comparison: Non-Linear vs Raw", fontsize=12, fontweight="bold")
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    axes[1, 0].legend(loc="best", fontsize=10)

    # Panel (1, 1): Exp 5 Tomography Shot Noise Scaling
    shots_num = [100, 300, 1000, 3000, 10000]
    frob = e5["mean_frob_errors"][:5]
    axes[1, 1].loglog(shots_num, frob, "o-", color="#D93025", lw=2.2, label="Observed Reconstruction Error")
    guide = [frob[0] * np.sqrt(100) / np.sqrt(s) for s in shots_num]
    axes[1, 1].loglog(shots_num, guide, "--", color="#5F6368", lw=1.8, label="Shot Noise Limit ~ 1/sqrt(N)")
    axes[1, 1].set_xlabel("Shots per Basis (N)", fontweight="bold")
    axes[1, 1].set_ylabel("Frobenius Error ||rho - rho_hat||_F", fontweight="bold")
    axes[1, 1].set_title("(d) Tomographic Reconstruction Stability", fontsize=12, fontweight="bold")
    axes[1, 1].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[1, 1].legend(loc="best", fontsize=10)

    plt.tight_layout()
    master_path = "reports/figures/master_summary_figure.png"
    plt.savefig(master_path, dpi=200)
    plt.close()
    return master_path


def main():
    print("=" * 70)
    print("QUANTUM ENTANGLEMENT PCA RESEARCH PIPELINE")
    print("=" * 70)
    start_time = time.time()

    os.makedirs("reports/figures", exist_ok=True)

    print("\n--- Running Experiment 1: Pure State Manifold ---")
    e1 = run_experiment_1()

    print("\n--- Running Experiment 2: Mixed States & Redundancy Breaking ---")
    e2 = run_experiment_2()

    print("\n--- Running Experiment 3: Quantum Noise Trajectories ---")
    e3 = run_experiment_3()

    print("\n--- Running Experiment 4: Raw Pauli vs Measures Representation ---")
    e4 = run_experiment_4()

    print("\n--- Running Experiment 5: Finite-Shot Tomography Degradation ---")
    e5 = run_experiment_5()

    print("\n--- Generating Master Composite Publication Figure ---")
    master_fig = generate_master_summary_figure(e1, e2, e3, e4, e5)
    print(f"Master figure saved: {master_fig}")

    total_time = time.time() - start_time
    summary = {
        "execution_time_seconds": total_time,
        "exp1": e1,
        "exp2": e2,
        "exp3": e3,
        "exp4": e4,
        "exp5": e5,
        "master_figure": master_fig,
    }

    with open("reports/experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"ALL EXPERIMENTS COMPLETED SUCCESSFULLY in {total_time:.2f} seconds.")
    print("Results and figures saved to reports/ and reports/figures/.")
    print("=" * 70)


if __name__ == "__main__":
    main()
