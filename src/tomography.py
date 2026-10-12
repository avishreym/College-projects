r"""
src/tomography.py
=================
Finite-shot Pauli measurement simulation and quantum state reconstruction.

Features:
1. Two-qubit projective measurement simulation across the 9 Pauli measurement settings:
   {XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ}.
2. Finite-shot multinomial sampling with configurable shot budget N in {100, 1000, 10000, inf}.
3. Linear Inversion state reconstruction:
   rho_lin = 1/4 * [ I \otimes I + sum c_{i0} (sigma_i \otimes I)
                                 + sum c_{0j} (I \otimes sigma_j)
                                 + sum c_{ij} (sigma_i \otimes sigma_j) ]
4. Physical state reconstruction via positive semi-definite simplex projection:
   Projects unphysical rho_lin (which may have negative eigenvalues due to shot noise)
   onto the nearest valid density matrix in Frobenius norm.
"""

from typing import Dict, Tuple, Optional, Union
import numpy as np

from src.measures import SIGMA_0, SIGMA_X, SIGMA_Y, SIGMA_Z, PAULI_VEC, PAULI_NAMES


# Projectors for single-qubit Pauli eigenstates:
# Pauli Z eigenstates |0>, |1>
P_Z_PLUS = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
P_Z_MINUS = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)

# Pauli X eigenstates |+>, |->
P_X_PLUS = 0.5 * np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128)
P_X_MINUS = 0.5 * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.complex128)

# Pauli Y eigenstates |+i>, |-i>
P_Y_PLUS = 0.5 * np.array([[1.0, -1.0j], [1.0j, 1.0]], dtype=np.complex128)
P_Y_MINUS = 0.5 * np.array([[1.0, 1.0j], [-1.0j, 1.0]], dtype=np.complex128)

BASIS_PROJECTORS = {
    "X": (P_X_PLUS, P_X_MINUS),
    "Y": (P_Y_PLUS, P_Y_MINUS),
    "Z": (P_Z_PLUS, P_Z_MINUS),
}


def project_to_probability_simplex(v: np.ndarray) -> np.ndarray:
    """
    Computes Euclidean projection of vector v onto probability simplex
    Delta = {p in R^d : p_i >= 0, sum p_i = 1}.
    Exact algorithm (Wang & Carreira-Perpinan, 2013).
    """
    d = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    indices = np.arange(1, d + 1)
    cond = u + (1.0 - cssv) / indices > 0
    rho = np.nonzero(cond)[0][-1]
    theta = (1.0 - cssv[rho]) / (rho + 1.0)
    return np.maximum(v + theta, 0.0)


def simulate_pauli_tomography(
    rho: np.ndarray,
    shots: Optional[int] = None,
    rng: Optional[np.random.Generator] = None
) -> Tuple[Dict[str, float], np.ndarray]:
    r"""
    Simulates projective measurements on two qubits across all 9 joint bases:
    (B_A, B_B) in {X, Y, Z}^2.

    If shots is None or shots == np.inf, returns exact theoretical expectation values.
    Otherwise, samples multinomial outcome counts for each setting with budget `shots`.

    Returns:
    (estimated_pauli_dict, reconstructed_density_matrix_physical)
    """
    if rng is None:
        rng = np.random.default_rng()

    # We will record correlation estimates and local marginal counts
    corr_estimates: Dict[str, float] = {}
    local_a_estimates: Dict[str, list] = {"X": [], "Y": [], "Z": []}
    local_b_estimates: Dict[str, list] = {"X": [], "Y": [], "Z": []}

    for b_a in ["X", "Y", "Z"]:
        p_a_plus, p_a_minus = BASIS_PROJECTORS[b_a]
        for b_b in ["X", "Y", "Z"]:
            p_b_plus, p_b_minus = BASIS_PROJECTORS[b_b]

            # 4 joint projectors
            proj_pp = np.kron(p_a_plus, p_b_plus)
            proj_pm = np.kron(p_a_plus, p_b_minus)
            proj_mp = np.kron(p_a_minus, p_b_plus)
            proj_mm = np.kron(p_a_minus, p_b_minus)

            prob_pp = float(np.real(np.trace(rho @ proj_pp)))
            prob_pm = float(np.real(np.trace(rho @ proj_pm)))
            prob_mp = float(np.real(np.trace(rho @ proj_mp)))
            prob_mm = float(np.real(np.trace(rho @ proj_mm)))

            probs = np.array([prob_pp, prob_pm, prob_mp, prob_mm])
            probs = np.maximum(0.0, probs)
            total_p = np.sum(probs)
            if total_p > 0:
                probs /= total_p
            else:
                probs = np.array([0.25, 0.25, 0.25, 0.25])

            if shots is None or shots == 0 or np.isinf(shots):
                # Infinite shots: exact probabilities
                n_pp, n_pm, n_mp, n_mm = probs
                tot = 1.0
            else:
                # Finite shots multinomial draw
                counts = rng.multinomial(shots, probs)
                n_pp, n_pm, n_mp, n_mm = counts
                tot = float(shots)

            # Expectation values:
            # <sigma_a \otimes sigma_b> = (n_pp - n_pm - n_mp + n_mm) / tot
            corr_exp = (n_pp - n_pm - n_mp + n_mm) / tot
            corr_estimates[f"{b_a}{b_b}"] = corr_exp

            # Marginal expectations from this joint setting
            exp_a = (n_pp + n_pm - n_mp - n_mm) / tot
            exp_b = (n_pp - n_pm + n_mp - n_mm) / tot
            local_a_estimates[b_a].append(exp_a)
            local_b_estimates[b_b].append(exp_b)

    # Average local marginal estimates across the 3 measurement settings
    full_obs: Dict[str, float] = {}
    for name in ["X", "Y", "Z"]:
        full_obs[f"{name}I"] = float(np.mean(local_a_estimates[name]))
        full_obs[f"I{name}"] = float(np.mean(local_b_estimates[name]))

    for pair, val in corr_estimates.items():
        full_obs[pair] = float(val)

    # State reconstruction via linear inversion:
    # rho_lin = 1/4 [ I + sum c_i (sigma_i \otimes I) + sum c_j (I \otimes sigma_j) + sum c_ij (sigma_i \otimes sigma_j) ]
    rho_lin = 0.25 * np.eye(4, dtype=np.complex128)

    for i, name_a in enumerate(PAULI_NAMES):
        rho_lin += 0.25 * full_obs[f"{name_a}I"] * np.kron(PAULI_VEC[i], SIGMA_0)

    for j, name_b in enumerate(PAULI_NAMES):
        rho_lin += 0.25 * full_obs[f"I{name_b}"] * np.kron(SIGMA_0, PAULI_VEC[j])

    for i, name_a in enumerate(PAULI_NAMES):
        for j, name_b in enumerate(PAULI_NAMES):
            rho_lin += 0.25 * full_obs[f"{name_a}{name_b}"] * np.kron(PAULI_VEC[i], PAULI_VEC[j])

    # Enforce Hermiticity
    rho_lin_herm = 0.5 * (rho_lin + rho_lin.conj().T)

    # Maximum-likelihood / Frobenius projection to valid density matrix simplex:
    evals, evecs = np.linalg.eigh(rho_lin_herm)
    # Project eigenvalues onto simplex
    projected_evals = project_to_probability_simplex(evals)
    rho_phys = evecs @ np.diag(projected_evals) @ evecs.conj().T

    # Condition final trace
    tr = np.real(np.trace(rho_phys))
    if tr > 0:
        rho_phys /= tr

    return full_obs, rho_phys
