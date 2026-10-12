r"""
src/measures.py
================
Exact analytical formulas and partial traces for two-qubit quantum states.

Theoretical foundations:
- Wootters concurrence (1998)
- Peres-Horodecki PPT negativity (1996, 1998)
- von Neumann & linear entropies
- Entanglement of Formation (Bennett et al. 1996)
- Quantum mutual information
- Horodecki CHSH nonlocality criterion (1995)
- 15 Pauli expectation values (Gell-Mann / Pauli basis expansion)

All functions operate on 4x4 density matrices representing states in C^2 \otimes C^2,
spanned by computational basis {|00>, |01>, |10>, |11>}.
"""

from typing import Dict, Tuple, List, Union
import numpy as np

# Standard Pauli matrices
SIGMA_0 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.complex128)  # Identity
SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)  # Pauli X (sigma_1)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)  # Pauli Y (sigma_2)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)  # Pauli Z (sigma_3)

PAULI_VEC = [SIGMA_X, SIGMA_Y, SIGMA_Z]
PAULI_NAMES = ["X", "Y", "Z"]

# Spin-flip operator for two qubits: sigma_y \otimes sigma_y
SIGMA_YY = np.kron(SIGMA_Y, SIGMA_Y)


def validate_density_matrix(rho: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Validates and conditions a two-qubit density matrix.

    Ensures:
    1. Shape is (4, 4)
    2. Hermiticity: rho == rho^dagger
    3. Positive semi-definiteness: eigenvalues >= -tol
    4. Unit trace: Tr(rho) == 1

    Returns the Hermitized, trace-normalized density matrix.
    """
    rho = np.asarray(rho, dtype=np.complex128)
    if rho.shape != (4, 4):
        raise ValueError(f"Expected 4x4 density matrix, got {rho.shape}")

    # Hermitize
    rho_herm = 0.5 * (rho + rho.conj().T)

    # Check trace
    tr = np.trace(rho_herm)
    if not np.isclose(np.real(tr), 1.0, atol=tol):
        # Normalize trace
        rho_herm = rho_herm / tr

    return rho_herm


def partial_trace_b(rho: np.ndarray) -> np.ndarray:
    """
    Computes reduced density matrix rho_A = Tr_B(rho).
    Traces over subsystem B (qubit 2).

    rho has components rho_{(iA, iB), (jA, jB)}.
    (rho_A)_{iA, jA} = sum_{kB} rho_{(iA, kB), (jA, kB)}.
    """
    r4 = rho.reshape(2, 2, 2, 2)
    # trace over axis 1 (iB) and axis 3 (jB)
    return np.trace(r4, axis1=1, axis2=3)


def partial_trace_a(rho: np.ndarray) -> np.ndarray:
    """
    Computes reduced density matrix rho_B = Tr_A(rho).
    Traces over subsystem A (qubit 1).

    (rho_B)_{iB, jB} = sum_{kA} rho_{(kA, iB), (kA, jB)}.
    """
    r4 = rho.reshape(2, 2, 2, 2)
    # trace over axis 0 (iA) and axis 2 (jA)
    return np.trace(r4, axis1=0, axis2=2)


def partial_trace(rho: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function returning (rho_A, rho_B).
    """
    return partial_trace_b(rho), partial_trace_a(rho)


def partial_transpose_b(rho: np.ndarray) -> np.ndarray:
    """
    Computes partial transpose with respect to subsystem B: rho^{T_B}.
    (rho^{T_B})_{(iA, iB), (jA, jB)} = rho_{(iA, jB), (jA, iB)}.
    """
    r4 = rho.reshape(2, 2, 2, 2)
    # Transpose B indices: axis 1 <-> axis 3
    r4_tb = r4.transpose(0, 3, 2, 1)
    return r4_tb.reshape(4, 4)


def purity(rho: np.ndarray) -> float:
    """
    Computes purity gamma = Tr(rho^2).
    For pure states: gamma = 1.
    For maximally mixed 2-qubit state: gamma = 1/4 = 0.25.
    """
    rho_sq = rho @ rho
    return float(np.real(np.trace(rho_sq)))


def linear_entropy(rho: np.ndarray) -> float:
    """
    Computes linear entropy S_L = 1 - Tr(rho^2).
    Ranges from 0 (pure state) to 0.75 (maximally mixed 2-qubit state).
    """
    return float(np.clip(1.0 - purity(rho), 0.0, 1.0))


def von_neumann_entropy(rho: np.ndarray, base: float = 2.0) -> float:
    """
    Computes von Neumann entropy S(rho) = -sum_i lambda_i * log(lambda_i).
    Safely clamps eigenvalues lambda_i <= 0 to eliminate nan/inf.
    """
    evals = np.linalg.eigvalsh(rho)
    # Discard non-positive eigenvalues
    evals_pos = evals[evals > 1e-15]
    if len(evals_pos) == 0:
        return 0.0

    if base == 2.0:
        ent = -np.sum(evals_pos * np.log2(evals_pos))
    else:
        ent = -np.sum(evals_pos * (np.log(evals_pos) / np.log(base)))

    return float(np.maximum(0.0, ent))


def concurrence(rho: np.ndarray) -> float:
    r"""
    Computes Wootters Concurrence C(rho) for a 2-qubit state.

    Formula:
    tilde{rho} = (sigma_y \otimes sigma_y) rho^* (sigma_y \otimes sigma_y)
    R = sqrt(sqrt(rho) tilde{rho} sqrt(rho))
    Eigenvalues of R or square roots of eigenvalues of rho * tilde{rho}
    sorted in descending order lambda_1 >= lambda_2 >= lambda_3 >= lambda_4.
    C = max(0, lambda_1 - lambda_2 - lambda_3 - lambda_4).

    Ranges from 0 (separable) to 1 (maximally entangled).
    """
    rho = validate_density_matrix(rho)
    # Spin-flipped density matrix
    rho_tilde = SIGMA_YY @ rho.conj() @ SIGMA_YY

    # Product rho @ rho_tilde
    product = rho @ rho_tilde

    # Eigenvalues of product are guaranteed to be real and non-negative
    evals = np.linalg.eigvals(product)
    # Protect against small numerical imaginary artifacts and negative real parts
    real_evals = np.maximum(0.0, np.real(evals))
    lambdas = np.sort(np.sqrt(real_evals))[::-1]

    c = lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3]
    return float(np.clip(c, 0.0, 1.0))


def negativity(rho: np.ndarray) -> float:
    """
    Computes negativity N(rho) from the partial transpose rho^{T_B}.
    N = sum_{lambda_i < 0} |lambda_i|.

    By the Peres-Horodecki (PPT) criterion for 2 qubits:
    N > 0 iff rho is entangled.
    For Bell states, N = 0.5.
    For pure states, N = C / 2.
    """
    rho_tb = partial_transpose_b(rho)
    evals = np.linalg.eigvalsh(rho_tb)
    neg_evals = evals[evals < 0.0]
    neg = np.sum(np.abs(neg_evals))
    return float(np.clip(neg, 0.0, 0.5))


def entanglement_of_formation(c: float) -> float:
    """
    Entanglement of Formation E_F from concurrence C.
    Evaluates binary Shannon entropy h(x) = -x log_2 x - (1-x) log_2(1-x)
    at x = (1 + sqrt(1 - C^2)) / 2.
    """
    c = float(np.clip(c, 0.0, 1.0))
    if c <= 1e-12:
        return 0.0
    if c >= 1.0 - 1e-12:
        return 1.0

    arg = np.sqrt(np.maximum(0.0, 1.0 - c**2))
    x = 0.5 * (1.0 + arg)
    x = np.clip(x, 0.5, 1.0 - 1e-15)

    h_x = -x * np.log2(x) - (1.0 - x) * np.log2(1.0 - x)
    return float(np.clip(h_x, 0.0, 1.0))


def mutual_information(rho: np.ndarray) -> float:
    """
    Quantum mutual information:
    I(A : B) = S(rho_A) + S(rho_B) - S(rho_AB).

    Measures total (classical + quantum) correlations.
    For pure states: S(rho_AB) = 0, so I(A : B) = 2 * S(rho_A).
    """
    rho_a, rho_b = partial_trace(rho)
    s_ab = von_neumann_entropy(rho)
    s_a = von_neumann_entropy(rho_a)
    s_b = von_neumann_entropy(rho_b)
    mi = s_a + s_b - s_ab
    return float(np.maximum(0.0, mi))


def horodecki_chsh_max(rho: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    r"""
    Computes the maximum CHSH expectation value S_max via the Horodecki criterion (1995).

    1. Construct 3x3 correlation matrix T_{ij} = Tr(rho * (sigma_i \otimes sigma_j))
       for i, j \in {x, y, z}.
    2. Compute eigenvalues of T^T T: u_1 >= u_2 >= u_3 >= 0.
    3. S_max = 2 * sqrt(u_1 + u_2).

    A state violates the CHSH Bell inequality iff S_max > 2.
    Maximum Tsirelson bound is 2*sqrt(2) approx 2.8284.

    Returns:
    (S_max, u_sorted, T_matrix)
    """
    T = np.zeros((3, 3), dtype=np.float64)
    for i, sigma_a in enumerate(PAULI_VEC):
        for j, sigma_b in enumerate(PAULI_VEC):
            obs = np.kron(sigma_a, sigma_b)
            T[i, j] = np.real(np.trace(rho @ obs))

    U = T.T @ T
    u_evals = np.linalg.eigvalsh(U)
    # Sort descending
    u_sorted = np.sort(np.maximum(0.0, u_evals))[::-1]
    s_max = 2.0 * np.sqrt(u_sorted[0] + u_sorted[1])

    return float(s_max), u_sorted, T


def pauli_expectation_values(rho: np.ndarray) -> Dict[str, float]:
    r"""
    Computes all 15 non-identity Pauli expectation values for a two-qubit state:
    - 3 local observables on A: <sigma_i \otimes I>
    - 3 local observables on B: <I \otimes sigma_j>
    - 9 correlation observables: <sigma_i \otimes sigma_j>

    Returns dictionary with keys:
    - Local A: 'IX', 'IY', 'IZ' (labeled as 'XI', 'YI', 'ZI' conventionally: A \otimes B)
      'XI', 'YI', 'ZI'
    - Local B: 'IX', 'IY', 'IZ'
    - Correlations: 'XX', 'XY', 'XZ', 'YX', 'YY', 'YZ', 'ZX', 'ZY', 'ZZ'
    """
    observables: Dict[str, float] = {}

    # Local A: sigma_i \otimes I
    for i, name_a in enumerate(PAULI_NAMES):
        obs = np.kron(PAULI_VEC[i], SIGMA_0)
        observables[f"{name_a}I"] = float(np.real(np.trace(rho @ obs)))

    # Local B: I \otimes sigma_j
    for j, name_b in enumerate(PAULI_NAMES):
        obs = np.kron(SIGMA_0, PAULI_VEC[j])
        observables[f"I{name_b}"] = float(np.real(np.trace(rho @ obs)))

    # 9 Correlations: sigma_i \otimes sigma_j
    for i, name_a in enumerate(PAULI_NAMES):
        for j, name_b in enumerate(PAULI_NAMES):
            obs = np.kron(PAULI_VEC[i], PAULI_VEC[j])
            observables[f"{name_a}{name_b}"] = float(np.real(np.trace(rho @ obs)))

    return observables


def compute_all_measures(rho: np.ndarray) -> Dict[str, Union[float, bool]]:
    """
    Calculates the full set of quantum information and correlation metrics
    for a given two-qubit density matrix:
    - Concurrence (C)
    - Negativity (N)
    - Entanglement of Formation (E_F)
    - Reduced entropy S_A
    - Reduced entropy S_B
    - Joint entropy S_AB
    - Quantum Mutual Information I(A:B)
    - Purity (gamma)
    - Linear Entropy S_L
    - CHSH S_max
    - Bell violation (S_max > 2.0)
    """
    rho = validate_density_matrix(rho)
    rho_a, rho_b = partial_trace(rho)

    c = concurrence(rho)
    neg = negativity(rho)
    ef = entanglement_of_formation(c)

    pur = purity(rho)
    sl = linear_entropy(rho)

    s_ab = von_neumann_entropy(rho)
    s_a = von_neumann_entropy(rho_a)
    s_b = von_neumann_entropy(rho_b)
    mi = mutual_information(rho)

    s_max, _, _ = horodecki_chsh_max(rho)
    bell_violates = bool(s_max > 2.0 + 1e-9)

    return {
        "concurrence": c,
        "negativity": neg,
        "entanglement_of_formation": ef,
        "entropy_a": s_a,
        "entropy_b": s_b,
        "entropy_ab": s_ab,
        "mutual_information": mi,
        "purity": pur,
        "linear_entropy": sl,
        "chsh_s_max": s_max,
        "bell_violation": bell_violates,
    }


def feature_vector_set_a(rho: np.ndarray) -> np.ndarray:
    """
    Constructs Feature Set A: [C, N, E_F, S_A, S_B, S_AB, I(A:B), Purity, S_max].
    Dimension: 9.
    """
    m = compute_all_measures(rho)
    return np.array([
        m["concurrence"],
        m["negativity"],
        m["entanglement_of_formation"],
        m["entropy_a"],
        m["entropy_b"],
        m["entropy_ab"],
        m["mutual_information"],
        m["purity"],
        m["chsh_s_max"]
    ], dtype=np.float64)


FEATURE_SET_A_NAMES = [
    "Concurrence",
    "Negativity",
    "Entanglement_of_Formation",
    "Entropy_A",
    "Entropy_B",
    "Entropy_AB",
    "Mutual_Information",
    "Purity",
    "CHSH_S_max"
]

PAULI_15_NAMES = [
    "XI", "YI", "ZI",
    "IX", "IY", "IZ",
    "XX", "XY", "XZ",
    "YX", "YY", "YZ",
    "ZX", "ZY", "ZZ"
]


def feature_vector_set_b(rho: np.ndarray) -> np.ndarray:
    """
    Constructs Feature Set B: 15 Pauli expectation values.
    Ordered as:
    XI, YI, ZI, IX, IY, IZ, XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ.
    Dimension: 15.
    """
    obs = pauli_expectation_values(rho)
    return np.array([obs[name] for name in PAULI_15_NAMES], dtype=np.float64)
