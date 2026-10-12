"""
src/generators.py
=================
Quantum State Generators for Two-Qubit Systems.

Provides generation functions for:
1. Standard basis & Bell basis pure states (|00>, |Phi+>, |Psi->, etc.)
2. Parametric pure states |psi(theta, phi)> = cos(theta)|00> + e^{i phi} sin(theta)|11>
3. Haar-random pure states (uniform distribution over CP^3 via complex Gaussian projection)
4. Werner states rho_W(p) = p |Psi-><Psi-| + (1-p)/4 * I_4
5. Isotropic states rho_iso(F) = F |Phi+><Phi+| + (1-F)/3 * (I_4 - |Phi+><Phi+|)
6. Random mixed states of variable rank r in {1, 2, 3, 4} (Hilbert-Schmidt / Ginibre ensemble)
7. Noise channels applied to Bell states:
   - Depolarizing channel: E(rho) = (1-p) rho + (p/4) I_4
   - Phase damping / Dephasing channel: Kraus operators K_0, K_1, K_2
   - Amplitude damping / Energy relaxation channel: Kraus operators for spontaneous decay
"""

from typing import Optional, List, Tuple
import numpy as np

# Basis vectors in C^4
KET_00 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
KET_01 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.complex128)
KET_10 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex128)
KET_11 = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.complex128)

# Bell states
KET_PHI_PLUS = (KET_00 + KET_11) / np.sqrt(2.0)
KET_PHI_MINUS = (KET_00 - KET_11) / np.sqrt(2.0)
KET_PSI_PLUS = (KET_01 + KET_10) / np.sqrt(2.0)
KET_PSI_MINUS = (KET_01 - KET_10) / np.sqrt(2.0)


def ket_to_density(ket: np.ndarray) -> np.ndarray:
    """Computes density matrix |ket><ket| from a state vector."""
    ket = np.asarray(ket, dtype=np.complex128)
    norm = np.linalg.norm(ket)
    if norm < 1e-15:
        raise ValueError("Cannot form density matrix from zero state vector.")
    normalized_ket = ket / norm
    return np.outer(normalized_ket, normalized_ket.conj())


def bell_state(variant: str = "phi_plus") -> np.ndarray:
    """
    Returns density matrix of specified Bell state:
    'phi_plus': (|00> + |11>) / sqrt(2)
    'phi_minus': (|00> - |11>) / sqrt(2)
    'psi_plus': (|01> + |10>) / sqrt(2)
    'psi_minus': (|01> - |10>) / sqrt(2)
    """
    variant = variant.lower()
    if variant in ("phi_plus", "phi+", "00+11"):
        return ket_to_density(KET_PHI_PLUS)
    elif variant in ("phi_minus", "phi-", "00-11"):
        return ket_to_density(KET_PHI_MINUS)
    elif variant in ("psi_plus", "psi+", "01+10"):
        return ket_to_density(KET_PSI_PLUS)
    elif variant in ("psi_minus", "psi-", "01-10"):
        return ket_to_density(KET_PSI_MINUS)
    else:
        raise ValueError(f"Unknown Bell variant: {variant}")


def product_state(variant: str = "00") -> np.ndarray:
    """
    Returns standard two-qubit product state:
    '00', '01', '10', '11', '++'
    """
    variant = variant.lower()
    if variant == "00":
        return ket_to_density(KET_00)
    elif variant == "01":
        return ket_to_density(KET_01)
    elif variant == "10":
        return ket_to_density(KET_10)
    elif variant == "11":
        return ket_to_density(KET_11)
    elif variant in ("++", "plus_plus"):
        # |+> = (|0> + |1>)/sqrt(2), |++> = 1/2 (|00> + |01> + |10> + |11>)
        ket = 0.5 * (KET_00 + KET_01 + KET_10 + KET_11)
        return ket_to_density(ket)
    else:
        raise ValueError(f"Unknown product state variant: {variant}")


def parametrized_pure_state(theta: float, phi: float = 0.0) -> np.ndarray:
    """
    Parametrized Schmidt decomposition pure state:
    |psi(theta, phi)> = cos(theta) |00> + e^{i phi} sin(theta) |11>

    Properties:
    - theta = 0: |00> (product state, C = 0)
    - theta = pi/4: (|00> + e^{i phi} |11>)/sqrt(2) (maximally entangled Bell state, C = 1)
    - Concurrence: C = |sin(2 theta)|
    """
    ket = np.cos(theta) * KET_00 + np.exp(1.0j * phi) * np.sin(theta) * KET_11
    return ket_to_density(ket)


def haar_random_pure_state(rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Generates a Haar-distributed random pure state in C^4.
    Sampled by generating complex standard Gaussian coordinates
    Re(v_k), Im(v_k) ~ N(0, 1) and normalizing.
    """
    if rng is None:
        rng = np.random.default_rng()

    real_part = rng.normal(size=4)
    imag_part = rng.normal(size=4)
    v = real_part + 1.0j * imag_part
    return ket_to_density(v)


def werner_state(p: float) -> np.ndarray:
    """
    Generates the Werner state:
    rho_W(p) = p |Psi-><Psi-| + (1 - p)/4 * I_4
    where p in [0, 1].

    Properties:
    - p = 1: singlet state |Psi-><Psi-| (pure, maximally entangled, C=1, N=0.5, S_max = 2*sqrt(2))
    - p = 0: maximally mixed state I/4 (separable, unpolarized)
    - Entangled (PPT violation, C > 0, N > 0) iff p > 1/3 ~ 0.3333
    - Bell-CHSH non-local (S_max > 2) iff p > 1/sqrt(2) ~ 0.7071
    - Nonlocality-entanglement gap: 1/3 < p <= 1/sqrt(2)
    """
    p = float(np.clip(p, 0.0, 1.0))
    rho_psi = ket_to_density(KET_PSI_MINUS)
    id_4 = np.eye(4, dtype=np.complex128)
    return p * rho_psi + (1.0 - p) * 0.25 * id_4


def isotropic_state(f: float) -> np.ndarray:
    """
    Generates the Isotropic state:
    rho_iso(F) = F |Phi+><Phi+| + (1 - F)/3 * (I_4 - |Phi+><Phi+|)
    where F in [0, 1] is the fidelity with |Phi+>.
    Entangled iff F > 0.5.
    """
    f = float(np.clip(f, 0.0, 1.0))
    rho_phi = ket_to_density(KET_PHI_PLUS)
    proj_ortho = (np.eye(4, dtype=np.complex128) - rho_phi) / 3.0
    return f * rho_phi + (1.0 - f) * proj_ortho


def random_mixed_state(
    rank: int = 4,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Generates a random mixed state of specific rank r in {1, 2, 3, 4}
    from the Hilbert-Schmidt / Ginibre ensemble:
    rho = G G^dagger / Tr(G G^dagger)
    where G in C^{4 x r} is a Gaussian random matrix.
    """
    if rank not in (1, 2, 3, 4):
        raise ValueError(f"Rank must be in {{1, 2, 3, 4}}, got {rank}")

    if rng is None:
        rng = np.random.default_rng()

    real_part = rng.normal(size=(4, rank))
    imag_part = rng.normal(size=(4, rank))
    G = real_part + 1.0j * imag_part

    rho_unnorm = G @ G.conj().T
    tr = np.real(np.trace(rho_unnorm))
    return rho_unnorm / tr


# ==============================================================================
# Quantum Noise Channels (Depolarizing, Dephasing, Amplitude Damping)
# ==============================================================================

def apply_depolarizing_channel(
    rho: np.ndarray,
    p: float,
    single_qubit: bool = False
) -> np.ndarray:
    """
    Applies depolarizing noise with probability p in [0, 1].

    For two qubits (global depolarizing channel):
    E(rho) = (1 - p) rho + (p / 4) * I_4
    """
    p = float(np.clip(p, 0.0, 1.0))
    id_4 = np.eye(4, dtype=np.complex128)
    return (1.0 - p) * rho + (p / 4.0) * id_4


def apply_local_dephasing_channel(
    rho: np.ndarray,
    p: float
) -> np.ndarray:
    """
    Applies identical phase damping (dephasing) noise of strength p to each qubit independently.
    Kraus operators for single qubit phase flip with probability gamma = p:
    K_0 = sqrt(1 - p/2) * I,  K_1 = sqrt(p/2) * Z
    Or exact phase damping:
    [ [rho_00, (1-p) rho_01],
      [(1-p) rho_10, rho_11] ]
    On two qubits, off-diagonal elements in the Z basis decay exponentially:
    rho_{jk, lm} -> (1-p)^{delta(j!=l) + delta(k!=m)} rho_{jk, lm}.
    """
    p = float(np.clip(p, 0.0, 1.0))
    eta = np.sqrt(np.maximum(0.0, 1.0 - p))

    # Construct single-qubit Kraus operators
    k0 = np.array([[1.0, 0.0], [0.0, eta]], dtype=np.complex128)
    k1 = np.array([[0.0, 0.0], [0.0, np.sqrt(p)]], dtype=np.complex128)
    single_kraus = [k0, k1]

    # Two-qubit Kraus operators = tensor products
    rho_out = np.zeros((4, 4), dtype=np.complex128)
    for ka in single_kraus:
        for kb in single_kraus:
            kab = np.kron(ka, kb)
            rho_out += kab @ rho @ kab.conj().T

    tr = np.real(np.trace(rho_out))
    return rho_out / tr


def apply_amplitude_damping_channel(
    rho: np.ndarray,
    gamma: float
) -> np.ndarray:
    """
    Applies independent amplitude damping (energy relaxation / decay |1> -> |0>)
    of decay parameter gamma in [0, 1] to both qubits.

    Single-qubit Kraus operators:
    K_0 = [[1, 0], [0, sqrt(1 - gamma)]]
    K_1 = [[0, sqrt(gamma)], [0, 0]]
    """
    gamma = float(np.clip(gamma, 0.0, 1.0))
    k0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=np.complex128)
    k1 = np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=np.complex128)
    single_kraus = [k0, k1]

    rho_out = np.zeros((4, 4), dtype=np.complex128)
    for ka in single_kraus:
        for kb in single_kraus:
            kab = np.kron(ka, kb)
            rho_out += kab @ rho @ kab.conj().T

    tr = np.real(np.trace(rho_out))
    return rho_out / tr


def generate_ensemble(
    n_pure: int = 150,
    n_werner: int = 50,
    n_mixed_r2: int = 75,
    n_mixed_r3: int = 75,
    n_mixed_r4: int = 100,
    n_noise_steps: int = 30,
    seed: int = 42
) -> List[Tuple[str, np.ndarray, dict]]:
    """
    Generates a balanced multi-class quantum state ensemble for PCA exploration:
    - Pure states (Haar random + parametrized Schmidt sweep)
    - Werner states across p in [0, 1]
    - Random mixed states of ranks 2, 3, 4
    - Noise channel sweeps starting from |Phi+> (depolarizing, dephasing, amplitude damping)

    Returns list of tuples: (label, rho, metadata_dict).
    """
    rng = np.random.default_rng(seed)
    ensemble = []

    # 1. Pure States
    # Parametrized sweep: theta in [0, pi/4], phi in [0, 2pi]
    thetas = np.linspace(0.0, np.pi / 4.0, 40)
    for th in thetas:
        rho = parametrized_pure_state(th, phi=0.0)
        ensemble.append(("Pure_Parametric", rho, {"theta": th, "phi": 0.0, "type": "Pure"}))

    # Haar random pure states
    for _ in range(n_pure):
        rho = haar_random_pure_state(rng)
        ensemble.append(("Pure_Haar", rho, {"type": "Pure"}))

    # 2. Werner state sweep
    ps = np.linspace(0.0, 1.0, n_werner)
    for p in ps:
        rho = werner_state(p)
        ensemble.append(("Werner", rho, {"p": p, "type": "Werner"}))

    # 3. Random mixed states (ranks 2, 3, 4)
    for _ in range(n_mixed_r2):
        rho = random_mixed_state(rank=2, rng=rng)
        ensemble.append(("Mixed_Rank2", rho, {"rank": 2, "type": "Mixed"}))
    for _ in range(n_mixed_r3):
        rho = random_mixed_state(rank=3, rng=rng)
        ensemble.append(("Mixed_Rank3", rho, {"rank": 3, "type": "Mixed"}))
    for _ in range(n_mixed_r4):
        rho = random_mixed_state(rank=4, rng=rng)
        ensemble.append(("Mixed_Rank4", rho, {"rank": 4, "type": "Mixed"}))

    # 4. Noise trajectories starting from |Phi+>
    rho_bell = bell_state("phi_plus")
    noise_vals = np.linspace(0.0, 1.0, n_noise_steps)

    for p in noise_vals:
        rho_depol = apply_depolarizing_channel(rho_bell, p)
        ensemble.append(("Noise_Depolarizing", rho_depol, {"p": p, "channel": "Depolarizing"}))

        rho_deph = apply_local_dephasing_channel(rho_bell, p)
        ensemble.append(("Noise_Dephasing", rho_deph, {"p": p, "channel": "Dephasing"}))

        rho_ad = apply_amplitude_damping_channel(rho_bell, p)
        ensemble.append(("Noise_AmpDamping", rho_ad, {"p": p, "channel": "AmpDamping"}))

    return ensemble
