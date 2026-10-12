"""
tests/test_validation.py
========================
Phase 1 Validation Suite: Unit tests against analytical ground truth.

Validates:
1. Max-entangled Bell state |Phi+>:
   - Concurrence C = 1.0
   - Negativity N = 0.5
   - Reduced von Neumann entropy S_A = 1.0
   - Entanglement of Formation E_F = 1.0
   - Maximum CHSH violation S_max = 2 * sqrt(2) ~ 2.828427
2. Separable Product states |00>, |++>:
   - Concurrence C = 0.0
   - Negativity N = 0.0
   - Quantum Mutual Information I(A:B) = 0.0
   - Reduced von Neumann entropy S_A = 0.0
   - CHSH S_max <= 2.0 (local realism respected)
3. Universal Pure State relations across Haar-random pure states:
   - N = C / 2
   - I(A:B) = 2 * S_A = 2 * S_B
   - S_AB = 0.0
   - Purity = 1.0
4. Parametrized Schmidt pure state sweep:
   - C(theta) = |sin(2 * theta)|
5. Werner state analytical thresholds rho_W(p) = p |Psi-><Psi-| + (1-p)/4 * I:
   - C > 0 and N > 0 iff p > 1/3
   - S_max > 2 iff p > 1/sqrt(2) ~ 0.707106
   - Nonlocality-entanglement gap: in range 1/3 < p <= 1/sqrt(2), state is entangled
     (C > 0, N > 0) yet admits a local hidden variable model under projective CHSH (S_max <= 2).
"""

import sys
import os
import pytest
import numpy as np

# Ensure src is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.measures import (
    validate_density_matrix,
    partial_trace,
    partial_transpose_b,
    purity,
    linear_entropy,
    von_neumann_entropy,
    concurrence,
    negativity,
    entanglement_of_formation,
    mutual_information,
    horodecki_chsh_max,
    pauli_expectation_values,
    compute_all_measures,
    feature_vector_set_a,
    feature_vector_set_b,
)
from src.generators import (
    bell_state,
    product_state,
    parametrized_pure_state,
    haar_random_pure_state,
    werner_state,
    isotropic_state,
    random_mixed_state,
)


class TestBellStatesGroundTruth:
    """Rigorous verification of the 4 maximally entangled Bell states."""

    @pytest.mark.parametrize("variant", ["phi_plus", "phi_minus", "psi_plus", "psi_minus"])
    def test_bell_state_properties(self, variant):
        rho = bell_state(variant)

        # 1. Density matrix validity
        assert rho.shape == (4, 4)
        np.testing.assert_allclose(np.trace(rho), 1.0, atol=1e-12)
        np.testing.assert_allclose(rho, rho.conj().T, atol=1e-12)

        # 2. Purity & Joint entropy
        pur = purity(rho)
        assert np.isclose(pur, 1.0, atol=1e-12), f"Bell purity {pur} != 1"
        s_ab = von_neumann_entropy(rho)
        assert np.isclose(s_ab, 0.0, atol=1e-12), f"Bell S_AB {s_ab} != 0"

        # 3. Reduced subsystem entropies
        rho_a, rho_b = partial_trace(rho)
        s_a = von_neumann_entropy(rho_a)
        s_b = von_neumann_entropy(rho_b)
        assert np.isclose(s_a, 1.0, atol=1e-10), f"Bell S_A {s_a} != 1.0"
        assert np.isclose(s_b, 1.0, atol=1e-10), f"Bell S_B {s_b} != 1.0"

        # 4. Concurrence & Negativity & Entanglement of Formation
        c = concurrence(rho)
        assert np.isclose(c, 1.0, atol=1e-10), f"Bell Concurrence {c} != 1.0"

        neg = negativity(rho)
        assert np.isclose(neg, 0.5, atol=1e-10), f"Bell Negativity {neg} != 0.5"

        ef = entanglement_of_formation(c)
        assert np.isclose(ef, 1.0, atol=1e-10), f"Bell E_F {ef} != 1.0"

        # 5. Quantum Mutual Information
        mi = mutual_information(rho)
        assert np.isclose(mi, 2.0, atol=1e-10), f"Bell I(A:B) {mi} != 2.0"

        # 6. Horodecki CHSH maximal violation (Tsirelson bound)
        s_max, u_vals, _ = horodecki_chsh_max(rho)
        expected_s_max = 2.0 * np.sqrt(2.0)
        assert np.isclose(s_max, expected_s_max, atol=1e-10), (
            f"Bell S_max {s_max} != 2*sqrt(2) = {expected_s_max}"
        )
        assert s_max > 2.0, "Bell state must violate local realism (CHSH > 2)"

    def test_bell_phi_plus_pauli_observables(self):
        """Check exact Pauli expectations for |Phi+> = (|00> + |11>)/sqrt(2)."""
        rho = bell_state("phi_plus")
        obs = pauli_expectation_values(rho)

        # Single qubit observables are all 0 for maximally entangled state
        assert np.isclose(obs["XI"], 0.0, atol=1e-12)
        assert np.isclose(obs["YI"], 0.0, atol=1e-12)
        assert np.isclose(obs["ZI"], 0.0, atol=1e-12)
        assert np.isclose(obs["IX"], 0.0, atol=1e-12)
        assert np.isclose(obs["IY"], 0.0, atol=1e-12)
        assert np.isclose(obs["IZ"], 0.0, atol=1e-12)

        # Known correlation values: XX = 1, YY = -1, ZZ = 1
        assert np.isclose(obs["XX"], 1.0, atol=1e-12)
        assert np.isclose(obs["YY"], -1.0, atol=1e-12)
        assert np.isclose(obs["ZZ"], 1.0, atol=1e-12)

        # Off-diagonal correlations are 0
        assert np.isclose(obs["XY"], 0.0, atol=1e-12)
        assert np.isclose(obs["XZ"], 0.0, atol=1e-12)
        assert np.isclose(obs["YX"], 0.0, atol=1e-12)
        assert np.isclose(obs["YZ"], 0.0, atol=1e-12)
        assert np.isclose(obs["ZX"], 0.0, atol=1e-12)
        assert np.isclose(obs["ZY"], 0.0, atol=1e-12)


class TestProductStatesGroundTruth:
    """Rigorous verification of separable product states."""

    @pytest.mark.parametrize("variant", ["00", "01", "10", "11", "++"])
    def test_product_states_entanglement_zero(self, variant):
        rho = product_state(variant)

        # 1. Purity is 1
        assert np.isclose(purity(rho), 1.0, atol=1e-12)

        # 2. Entanglement measures must be exactly 0
        c = concurrence(rho)
        neg = negativity(rho)
        ef = entanglement_of_formation(c)
        assert np.isclose(c, 0.0, atol=1e-10), f"Product concurrence {c} != 0"
        assert np.isclose(neg, 0.0, atol=1e-10), f"Product negativity {neg} != 0"
        assert np.isclose(ef, 0.0, atol=1e-10), f"Product E_F {ef} != 0"

        # 3. Subsystem and joint entropies are 0
        rho_a, rho_b = partial_trace(rho)
        s_a = von_neumann_entropy(rho_a)
        s_b = von_neumann_entropy(rho_b)
        s_ab = von_neumann_entropy(rho)
        assert np.isclose(s_a, 0.0, atol=1e-10)
        assert np.isclose(s_b, 0.0, atol=1e-10)
        assert np.isclose(s_ab, 0.0, atol=1e-10)

        # 4. Quantum Mutual Information is 0
        mi = mutual_information(rho)
        assert np.isclose(mi, 0.0, atol=1e-10)

        # 5. CHSH Nonlocality must satisfy classical bound S_max <= 2.0
        s_max, _, _ = horodecki_chsh_max(rho)
        assert s_max <= 2.0 + 1e-10, f"Product state S_max {s_max} > 2.0"


class TestPureStateTheorems:
    """
    Verification of fundamental quantum information theorems on pure states:
    1. Negativity = Concurrence / 2
    2. Mutual Information I(A:B) = 2 * S_A = 2 * S_B
    3. Purity == 1.0
    """

    def test_haar_random_pure_state_relations(self):
        rng = np.random.default_rng(2026)
        n_trials = 30

        for _ in range(n_trials):
            rho = haar_random_pure_state(rng)

            # Purity is 1
            assert np.isclose(purity(rho), 1.0, atol=1e-10)

            # Joint entropy is 0
            s_ab = von_neumann_entropy(rho)
            assert np.isclose(s_ab, 0.0, atol=1e-10)

            # Subsystem entropies are equal (Schmidt decomposition theorem)
            rho_a, rho_b = partial_trace(rho)
            s_a = von_neumann_entropy(rho_a)
            s_b = von_neumann_entropy(rho_b)
            assert np.isclose(s_a, s_b, atol=1e-10), f"S_A ({s_a}) != S_B ({s_b})"

            # Mutual information theorem: I(A:B) = 2 * S_A
            mi = mutual_information(rho)
            assert np.isclose(mi, 2.0 * s_a, atol=1e-10), f"I(A:B) {mi} != 2*S_A ({2*s_a})"

            # Pure state Negativity theorem: N = C / 2
            c = concurrence(rho)
            neg = negativity(rho)
            assert np.isclose(neg, 0.5 * c, atol=1e-8), (
                f"Pure state N ({neg}) != C/2 ({0.5 * c})"
            )

    def test_parametrized_schmidt_sweep(self):
        """Verify C(theta) = |sin(2*theta)| for |psi> = cos(theta)|00> + sin(theta)|11>."""
        thetas = np.linspace(0.0, np.pi / 2.0, 25)
        for theta in thetas:
            rho = parametrized_pure_state(theta, phi=0.0)
            c = concurrence(rho)
            expected_c = np.abs(np.sin(2.0 * theta))
            assert np.isclose(c, expected_c, atol=1e-10), (
                f"At theta={theta:.3f}: C={c:.6f} != expected={expected_c:.6f}"
            )


class TestWernerStateGroundTruth:
    """
    Werner states rho_W(p) = p |Psi-><Psi-| + (1 - p)/4 * I_4:
    - Entangled (C > 0, N > 0) <=> p > 1/3 ~ 0.333333
    - CHSH non-local (S_max > 2) <=> p > 1/sqrt(2) ~ 0.707107
    - Nonlocality-entanglement gap in (1/3, 1/sqrt(2)]
    """

    def test_werner_analytical_formula_curves(self):
        ps = np.linspace(0.0, 1.0, 51)
        for p in ps:
            rho = werner_state(p)
            c = concurrence(rho)
            neg = negativity(rho)
            s_max, _, _ = horodecki_chsh_max(rho)

            # Exact analytical formulas:
            expected_c = max(0.0, (3.0 * p - 1.0) / 2.0)
            expected_neg = max(0.0, (3.0 * p - 1.0) / 4.0)
            expected_s_max = 2.0 * np.sqrt(2.0) * p

            assert np.isclose(c, expected_c, atol=1e-8), (
                f"Werner p={p:.3f}: C={c:.6f} != expected={expected_c:.6f}"
            )
            assert np.isclose(neg, expected_neg, atol=1e-8), (
                f"Werner p={p:.3f}: Neg={neg:.6f} != expected={expected_neg:.6f}"
            )
            assert np.isclose(s_max, expected_s_max, atol=1e-8), (
                f"Werner p={p:.3f}: S_max={s_max:.6f} != expected={expected_s_max:.6f}"
            )

    def test_werner_thresholds_and_gap(self):
        # 1. Below entanglement threshold: p = 0.30 (< 1/3)
        rho_sep = werner_state(0.30)
        assert concurrence(rho_sep) == 0.0
        assert negativity(rho_sep) == 0.0
        s_max_sep, _, _ = horodecki_chsh_max(rho_sep)
        assert s_max_sep < 2.0

        # 2. In the gap: p = 0.50 (1/3 < p < 1/sqrt(2))
        rho_gap = werner_state(0.50)
        c_gap = concurrence(rho_gap)
        neg_gap = negativity(rho_gap)
        s_max_gap, _, _ = horodecki_chsh_max(rho_gap)

        assert c_gap > 0.0, "In gap, Werner state must be entangled (C > 0)"
        assert neg_gap > 0.0, "In gap, Werner state must have N > 0"
        assert s_max_gap <= 2.0, (
            f"In gap, Werner state cannot violate CHSH; got S_max = {s_max_gap}"
        )

        # 3. Above non-locality threshold: p = 0.85 (> 1/sqrt(2) ~ 0.7071)
        rho_nonlocal = werner_state(0.85)
        assert concurrence(rho_nonlocal) > 0.0
        assert negativity(rho_nonlocal) > 0.0
        s_max_nl, _, _ = horodecki_chsh_max(rho_nonlocal)
        assert s_max_nl > 2.0, "Above 1/sqrt(2), Werner state must violate CHSH (S_max > 2)"


class TestFeatureVectors:
    """Verify feature vector shapes and sanity."""

    def test_feature_sets_dimensions(self):
        rho = bell_state("phi_plus")
        feat_a = feature_vector_set_a(rho)
        feat_b = feature_vector_set_b(rho)

        assert feat_a.shape == (9,)
        assert feat_b.shape == (15,)
        assert not np.isnan(feat_a).any()
        assert not np.isnan(feat_b).any()


class TestPCAEngine:
    """Rigorous verification of ManualPCA and analytical characteristic equation."""

    def test_analytical_2x2_characteristic_equation(self):
        from src.pca import analytical_2x2_demonstration
        res = analytical_2x2_demonstration()
        assert res["match"] is True
        l1, l2 = res["lambda_exact"]
        assert np.isclose(l1, 5.0, atol=1e-12)
        assert np.isclose(l2, 0.0, atol=1e-12)

    def test_pca_sklearn_numerical_parity(self):
        from src.pca import benchmark_against_sklearn
        # Generate synthetic quantum feature data
        rng = np.random.default_rng(42)
        X = rng.normal(loc=0.5, scale=0.3, size=(100, 9))

        # Test both unstandardized and standardized
        res_raw = benchmark_against_sklearn(X, standardize=False)
        assert res_raw["is_numerically_equivalent"] is True
        assert res_raw["max_variance_diff"] < 1e-7
        assert res_raw["max_evr_diff"] < 1e-7

        res_std = benchmark_against_sklearn(X, standardize=True)
        assert res_std["is_numerically_equivalent"] is True
        assert res_std["max_variance_diff"] < 1e-7
        assert res_std["max_evr_diff"] < 1e-7

    def test_pca_inverse_transform_reconstruction(self):
        from src.pca import ManualPCA
        rng = np.random.default_rng(123)
        X = rng.normal(size=(50, 6))
        pca = ManualPCA(n_components=6, standardize=True)
        Z = pca.fit_transform(X)
        X_rec = pca.inverse_transform(Z)
        np.testing.assert_allclose(X, X_rec, atol=1e-10)


class TestTomographyReconstruction:
    """Verification of finite-shot simulation and physical state reconstruction."""

    def test_infinite_shots_exact_recovery(self):
        from src.tomography import simulate_pauli_tomography
        rho_true = bell_state("phi_plus")
        obs, rho_rec = simulate_pauli_tomography(rho_true, shots=None)

        # Frobenius distance between true and reconstructed should be 0
        diff = np.linalg.norm(rho_true - rho_rec, ord="fro")
        assert diff < 1e-12, f"Infinite-shot tomography error {diff} > 1e-12"

    def test_finite_shots_physicality(self):
        from src.tomography import simulate_pauli_tomography
        rng = np.random.default_rng(999)
        rho_true = bell_state("phi_plus")

        # Test very low shots (N=100) and moderate shots (N=1000)
        for shots in [100, 1000]:
            _, rho_rec = simulate_pauli_tomography(rho_true, shots=shots, rng=rng)

            # 1. Hermiticity
            np.testing.assert_allclose(rho_rec, rho_rec.conj().T, atol=1e-10)

            # 2. Unit trace
            assert np.isclose(np.real(np.trace(rho_rec)), 1.0, atol=1e-10)

            # 3. Positive semi-definiteness: all eigenvalues >= -1e-12
            evals = np.linalg.eigvalsh(rho_rec)
            assert np.all(evals >= -1e-12), f"Negative eigenvalues found: {evals}"

