"""
experiments/exp3_noise_trajectories.py
======================================
Experiment 3: Quantum Noise Channels and Dynamical Trajectories in Latent Space.

Research Question:
How do distinct open quantum system noise mechanisms (depolarizing, dephasing,
and amplitude damping) deform the entanglement manifold and navigate PCA space?

Physical Dynamics:
1. Depolarizing Channel: Isotropic shrinkage toward the identity I/4.
   Entanglement drops to 0 at critical threshold p_c. Purity drops to 0.25.
2. Dephasing Channel: Destroys off-diagonal phase coherence in computational basis.
   Asymptotic state is classically correlated diagonal mixture:
   rho_infty = 1/2 (|00><00| + |11><11|).
   Entanglement drops to 0, but classical correlation and purity (gamma=0.5) remain.
3. Amplitude Damping Channel: Asymmetric energy relaxation toward the ground state |00>.
   Entanglement undergoes Entanglement Sudden Death (ESD).
   Asymptotic state is pure |00><00| (Purity=1.0, Concurrence=0.0).
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
    bell_state,
    apply_depolarizing_channel,
    apply_local_dephasing_channel,
    apply_amplitude_damping_channel,
    generate_ensemble,
)
from src.pca import ManualPCA


def run_experiment_3(n_steps: int = 50, seed: int = 42):
    # 1. Fit reference PCA on standard quantum ensemble
    ensemble = generate_ensemble(seed=seed)
    X_ref = np.array([feature_vector_set_a(item[1]) for item in ensemble])
    pca = ManualPCA(standardize=True)
    pca.fit(X_ref)

    # 2. Generate fine noise sweeps starting from |Phi+>
    rho_bell = bell_state("phi_plus")
    p_values = np.linspace(0.0, 1.0, n_steps)

    traj_depol = []
    traj_deph = []
    traj_amp = []

    c_depol, pur_depol = [], []
    c_deph, pur_deph = [], []
    c_amp, pur_amp = [], []

    for p in p_values:
        # Depolarizing
        r_dep = apply_depolarizing_channel(rho_bell, p)
        traj_depol.append(feature_vector_set_a(r_dep))
        m_dep = compute_all_measures(r_dep)
        c_depol.append(m_dep["concurrence"])
        pur_depol.append(m_dep["purity"])

        # Dephasing
        r_dph = apply_local_dephasing_channel(rho_bell, p)
        traj_deph.append(feature_vector_set_a(r_dph))
        m_dph = compute_all_measures(r_dph)
        c_deph.append(m_dph["concurrence"])
        pur_deph.append(m_dph["purity"])

        # Amplitude Damping
        r_ad = apply_amplitude_damping_channel(rho_bell, p)
        traj_amp.append(feature_vector_set_a(r_ad))
        m_ad = compute_all_measures(r_ad)
        c_amp.append(m_ad["concurrence"])
        pur_amp.append(m_ad["purity"])

    Z_depol = pca.transform(np.array(traj_depol))
    Z_deph = pca.transform(np.array(traj_deph))
    Z_amp = pca.transform(np.array(traj_amp))

    # Reference background projection
    Z_ref = pca.transform(X_ref)

    # 3. Create publication visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)

    # Panel 1: Latent PCA Trajectories
    axes[0].scatter(Z_ref[:, 0], Z_ref[:, 1], c="#E2E8F0", s=15, alpha=0.6, label="Ensemble Background")
    
    # Depolarizing curve
    axes[0].plot(Z_depol[:, 0], Z_depol[:, 1], color="#D93025", lw=2.5, label="Depolarizing (-> I/4)")
    axes[0].scatter(Z_depol[-1, 0], Z_depol[-1, 1], color="#D93025", s=90, marker="x")

    # Dephasing curve
    axes[0].plot(Z_deph[:, 0], Z_deph[:, 1], color="#1A73E8", lw=2.5, label="Dephasing (-> Classical Mix)")
    axes[0].scatter(Z_deph[-1, 0], Z_deph[-1, 1], color="#1A73E8", s=90, marker="s")

    # Amplitude damping curve
    axes[0].plot(Z_amp[:, 0], Z_amp[:, 1], color="#0D9488", lw=2.5, label="Amplitude Damping (-> |00>)")
    axes[0].scatter(Z_amp[-1, 0], Z_amp[-1, 1], color="#0D9488", s=90, marker="^")

    # Mark starting Bell state
    axes[0].scatter(Z_depol[0, 0], Z_depol[0, 1], color="#E8710A", s=140, marker="*", edgecolors="black", zorder=10, label="Initial |Phi+>")

    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11, fontweight="bold")
    axes[0].set_title("Dynamical Noise Trajectories in Latent PC Plane", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(loc="best", fontsize=9)

    # Panel 2: Concurrence Decay vs Noise Parameter p
    axes[1].plot(p_values, c_depol, color="#D93025", lw=2.2, label="Depolarizing Concurrence")
    axes[1].plot(p_values, c_deph, color="#1A73E8", lw=2.2, label="Dephasing Concurrence")
    axes[1].plot(p_values, c_amp, color="#0D9488", lw=2.2, label="Amplitude Damping Concurrence")
    axes[1].set_xlabel("Noise Strength Parameter $p$", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Wootters Concurrence $C(\\rho)$", fontsize=11, fontweight="bold")
    axes[1].set_title("Concurrence Decay & Entanglement Death", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(loc="best", fontsize=9)

    plt.tight_layout()
    os.makedirs("reports/figures", exist_ok=True)
    fig_path = "reports/figures/exp3_noise_trajectories.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    results = {
        "p_values": p_values.tolist(),
        "depol_scores": Z_depol[:, :2].tolist(),
        "deph_scores": Z_deph[:, :2].tolist(),
        "amp_scores": Z_amp[:, :2].tolist(),
        "depol_concurrence": c_depol,
        "deph_concurrence": c_deph,
        "amp_concurrence": c_amp,
        "depol_purity": pur_depol,
        "deph_purity": pur_deph,
        "amp_purity": pur_amp,
        "figure_saved": fig_path,
        "esd_threshold_depol": float(p_values[np.where(np.array(c_depol) < 1e-4)[0][0]]) if np.any(np.array(c_depol) < 1e-4) else None,
        "esd_threshold_amp": float(p_values[np.where(np.array(c_amp) < 1e-4)[0][0]]) if np.any(np.array(c_amp) < 1e-4) else None,
    }

    print("[Exp 3] Finished. Noise trajectories traced.")
    return results


if __name__ == "__main__":
    res = run_experiment_3()
    with open("reports/exp3_results.json", "w") as f:
        json.dump(res, f, indent=2)
