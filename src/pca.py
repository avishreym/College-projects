"""
src/pca.py
==========
Manual Principal Component Analysis (PCA) Engine for Scientific Computing.

Mathematical derivation:
Given a data matrix X in R^{n x p} with n observations and p features:
1. Centering:
   mu_j = (1/n) * sum_{i=1}^n X_{ij}
   X_c = X - 1_n mu^T
2. Optional Standardization (z-score scaling):
   sigma_j = sqrt( (1/(n-1)) * sum_{i=1}^n (X_{ij} - mu_j)^2 )
   X_s = X_c / sigma_j
3. Sample Covariance Matrix:
   S = (1 / (n - 1)) * X_s^T X_s in R^{p x p}
4. Spectral Theorem / Eigendecomposition:
   S v_k = lambda_k v_k
   Since S is symmetric and positive semi-definite:
   S = V Lambda V^T, with lambda_1 >= lambda_2 >= ... >= lambda_p >= 0.
5. Explained Variance Ratios:
   EVR_k = lambda_k / sum_{j=1}^p lambda_j
6. Principal Component Scores:
   Z = X_s V  in R^{n x k}
"""

from typing import Optional, Dict, Tuple, Any
import numpy as np
import scipy.linalg


class ManualPCA:
    """
    Scratch implementation of Principal Component Analysis with strict
    scientific reproducibility, sign-determinism, and sklearn parity.
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        standardize: bool = True,
        eps: float = 1e-12
    ):
        """
        Args:
            n_components: Number of principal components to keep (default: all p).
            standardize: If True, uses correlation matrix (z-score scaling).
                         If False, uses raw covariance matrix.
            eps: Epsilon threshold to protect against division by zero in variance.
        """
        self.n_components = n_components
        self.standardize = standardize
        self.eps = eps

        # Fitted parameters
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.covariance_: Optional[np.ndarray] = None
        self.eigenvalues_: Optional[np.ndarray] = None
        self.components_: Optional[np.ndarray] = None  # Loadings matrix (k x p)
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.cumulative_variance_ratio_: Optional[np.ndarray] = None
        self.n_samples_: int = 0
        self.n_features_: int = 0

    def fit(self, X: np.ndarray) -> "ManualPCA":
        """
        Fits PCA model on data matrix X of shape (n_samples, n_features).
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")

        n, p = X.shape
        if n < 2:
            raise ValueError(f"PCA requires at least 2 samples, got n={n}")

        self.n_samples_ = n
        self.n_features_ = p
        k = self.n_components if self.n_components is not None else p
        k = min(k, n, p)

        # 1. Centering
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # 2. Standardization
        if self.standardize:
            sample_std = np.std(X_centered, axis=0, ddof=1)
            # Prevent division by zero for zero-variance features
            self.scale_ = np.where(sample_std > self.eps, sample_std, 1.0)
            X_norm = X_centered / self.scale_
        else:
            self.scale_ = np.ones(p, dtype=np.float64)
            X_norm = X_centered

        # 3. Sample Covariance Matrix S = (1/(n-1)) X^T X
        S = (X_norm.T @ X_norm) / (n - 1.0)
        # Ensure exact numerical symmetry
        S = 0.5 * (S + S.T)
        self.covariance_ = S

        # 4. Eigendecomposition via scipy.linalg.eigh (specialized for Hermitian/symmetric)
        eigenvalues, eigenvectors = scipy.linalg.eigh(S)

        # scipy.linalg.eigh returns eigenvalues in ascending order -> reverse to descending
        idx_sorted = np.argsort(eigenvalues)[::-1]
        sorted_evals = np.maximum(0.0, eigenvalues[idx_sorted])
        sorted_evecs = eigenvectors[:, idx_sorted]

        # 5. Deterministic sign convention (align with sklearn convention):
        # Force the component with largest absolute loading to have a positive sign
        for i in range(sorted_evecs.shape[1]):
            max_abs_idx = np.argmax(np.abs(sorted_evecs[:, i]))
            if sorted_evecs[max_abs_idx, i] < 0.0:
                sorted_evecs[:, i] *= -1.0

        # Store all eigenvalues and components
        self.eigenvalues_ = sorted_evals
        self.explained_variance_ = sorted_evals[:k]

        total_var = np.sum(sorted_evals)
        if total_var > self.eps:
            self.explained_variance_ratio_ = sorted_evals[:k] / total_var
        else:
            self.explained_variance_ratio_ = np.zeros(k, dtype=np.float64)

        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

        # Loadings matrix: convention is (n_components, n_features)
        # so each row is a principal component axis
        self.components_ = sorted_evecs[:, :k].T

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Projects data X into the principal component subspace:
        Z = X_norm @ components_.T
        """
        if self.components_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("PCA model is not fitted yet.")

        X = np.asarray(X, dtype=np.float64)
        X_norm = (X - self.mean_) / self.scale_
        return X_norm @ self.components_.T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fits model and returns scores Z."""
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """
        Reconstructs original feature space from principal component scores:
        X_rec = Z @ components_ * scale_ + mean_
        """
        if self.components_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("PCA model is not fitted yet.")

        Z = np.asarray(Z, dtype=np.float64)
        X_norm_rec = Z @ self.components_
        return X_norm_rec * self.scale_ + self.mean_


def analytical_2x2_demonstration() -> Dict[str, Any]:
    r"""
    Explicit theoretical derivation solving the 2x2 characteristic polynomial:
    Given a symmetric covariance matrix:
    S = [[s_11, s_12],
         [s_12, s_22]]

    The characteristic equation is:
    det(S - lambda * I) = 0
    (s_11 - lambda)(s_22 - lambda) - s_12^2 = 0
    lambda^2 - Tr(S) * lambda + det(S) = 0

    Roots:
    lambda_{1, 2} = (Tr(S) \pm sqrt(Tr(S)^2 - 4 * det(S))) / 2

    Demonstrated on an exact numerical example:
    S = [[4.0, 2.0],
         [2.0, 1.0]]
    Tr(S) = 5.0, det(S) = 4*1 - 2*2 = 0.0.
    Discriminant = 25 - 0 = 25.
    lambda_1 = (5 + 5)/2 = 5.0, lambda_2 = (5 - 5)/2 = 0.0.
    """
    S_example = np.array([[4.0, 2.0], [2.0, 1.0]], dtype=np.float64)
    tr_S = np.trace(S_example)
    det_S = np.linalg.det(S_example)

    discriminant = tr_S**2 - 4.0 * det_S
    lambda_1_exact = (tr_S + np.sqrt(discriminant)) / 2.0
    lambda_2_exact = (tr_S - np.sqrt(discriminant)) / 2.0

    # Numerical verification via scipy
    num_evals, num_evecs = scipy.linalg.eigh(S_example)
    num_evals_desc = np.sort(num_evals)[::-1]

    # Eigenvector for lambda_1 = 5.0:
    # (S - 5I) v = 0 => [[-1, 2], [2, -4]] [v_x, v_y]^T = 0 => v_x = 2 v_y
    # Normalized: [2/sqrt(5), 1/sqrt(5)]
    v1_exact = np.array([2.0, 1.0]) / np.sqrt(5.0)
    v2_exact = np.array([-1.0, 2.0]) / np.sqrt(5.0)

    return {
        "covariance_matrix": S_example,
        "trace": tr_S,
        "determinant": det_S,
        "discriminant": discriminant,
        "lambda_exact": (lambda_1_exact, lambda_2_exact),
        "lambda_numerical": (num_evals_desc[0], num_evals_desc[1]),
        "v1_exact": v1_exact,
        "v2_exact": v2_exact,
        "match": bool(
            np.isclose(lambda_1_exact, num_evals_desc[0], atol=1e-12) and
            np.isclose(lambda_2_exact, num_evals_desc[1], atol=1e-12)
        )
    }


def benchmark_against_sklearn(
    X: np.ndarray,
    standardize: bool = False
) -> Dict[str, Any]:
    """
    Benchmarks ManualPCA against sklearn.decomposition.PCA.
    Verifies numerical equivalence on eigenvalues, explained variance ratio,
    and transformed coordinates up to column sign ambiguity.
    """
    from sklearn.decomposition import PCA as SklearnPCA

    # Fit manual PCA
    manual = ManualPCA(standardize=standardize)
    Z_manual = manual.fit_transform(X)

    # For exact comparison with standard sklearn:
    # sklearn PCA centers but does not divide by std (raw covariance).
    if standardize:
        # Scale X manually for sklearn to compare standardized versions
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0, ddof=1)
        std = np.where(std > 1e-12, std, 1.0)
        X_for_sklearn = (X - mean) / std
    else:
        X_for_sklearn = X

    sklearn_pca = SklearnPCA()
    Z_sklearn = sklearn_pca.fit_transform(X_for_sklearn)

    # Compare eigenvalues (sample variances of components)
    var_manual = manual.explained_variance_
    var_sklearn = sklearn_pca.explained_variance_
    var_diff = np.max(np.abs(var_manual - var_sklearn))

    # Compare explained variance ratios
    evr_manual = manual.explained_variance_ratio_
    evr_sklearn = sklearn_pca.explained_variance_ratio_
    evr_diff = np.max(np.abs(evr_manual - evr_sklearn))

    # Compare projected coordinates Z up to sign flip per column
    k = Z_manual.shape[1]
    coord_diffs = []
    for col in range(k):
        diff_pos = np.max(np.abs(Z_manual[:, col] - Z_sklearn[:, col]))
        diff_neg = np.max(np.abs(Z_manual[:, col] + Z_sklearn[:, col]))
        coord_diffs.append(min(diff_pos, diff_neg))

    max_coord_diff = max(coord_diffs)

    is_equivalent = bool(var_diff < 1e-7 and evr_diff < 1e-7 and max_coord_diff < 1e-7)

    return {
        "max_variance_diff": float(var_diff),
        "max_evr_diff": float(evr_diff),
        "max_score_diff": float(max_coord_diff),
        "is_numerically_equivalent": is_equivalent
    }
