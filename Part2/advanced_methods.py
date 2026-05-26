import numpy as np
import sys
import os

# Cho phép import Part1, Part2 khi chạy file trực tiếp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class KernelRidgeRegression:
    def __init__(self, lmbda: float = 1.0, length_scale: float = 1.0):
        """
        Kernel Ridge Regression with RBF Kernel.
        
        Args:
            lmbda: Regularization parameter (lambda).
            length_scale: Length scale (l) for RBF kernel.
        """
        self.lmbda = lmbda
        self.length_scale = length_scale
        self.X_train = None
        self.dual_coeff = None

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Computes the RBF (Gaussian) kernel matrix with numerical stability."""
        # ||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b
        sq_norm1 = np.sum(X1**2, axis=1).reshape(-1, 1)
        sq_norm2 = np.sum(X2**2, axis=1).reshape(1, -1)
        distances_sq = sq_norm1 + sq_norm2 - 2 * np.dot(X1, X2.T)
        
        # clip to 0 to avoid negative values due to floating point error
        return np.exp(-np.maximum(0, distances_sq) / (2 * self.length_scale**2))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits the model using the dual representation: alpha = (K + lambda*I)^-1 * y"""
        self.X_train = X
        n_samples = X.shape[0]
        K = self._rbf_kernel(X, X)
        
        # Solving the linear system for dual coefficients
        self.dual_coeff = np.linalg.solve(K + self.lmbda * np.eye(n_samples), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts using the kernel trick: y_hat = k(x, X_train) * alpha."""
        K_test = self._rbf_kernel(X, self.X_train)
        return np.dot(K_test, self.dual_coeff)


class BayesianLinearRegression:
    def __init__(self, sigma_sq: float = 1.0, m0: np.ndarray = None, S0: np.ndarray = None):
        """
        Bayesian Linear Regression from scratch.
        
        Args:
            sigma_sq: Variance of the likelihood (sigma^2).
            m0: Prior mean (defaults to 0).
            S0: Prior covariance (defaults to Identity).
        """
        self.sigma_sq = sigma_sq
        self.m0 = m0
        self.S0 = S0
        self.mn = None
        self.Sn = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Computes posterior: Sn = (S0^-1 + 1/sigma^2 * X.T@X)^-1, mn = Sn(S0^-1@m0 + 1/sigma^2 * X.T@y)"""
        n_features = X.shape[1]
        
        if self.m0 is None: self.m0 = np.zeros(n_features)
        if self.S0 is None: self.S0 = np.eye(n_features)
            
        inv_S0 = np.linalg.inv(self.S0)
        
        # Precision matrix calculation
        precision_n = inv_S0 + (1 / self.sigma_sq) * np.dot(X.T, X)
        self.Sn = np.linalg.inv(precision_n)
        
        # Posterior mean calculation
        term1 = np.dot(inv_S0, self.m0)
        term2 = (1 / self.sigma_sq) * np.dot(X.T, y)
        self.mn = np.dot(self.Sn, term1 + term2)
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        """Predictive distribution mean: y = X @ mn"""
        y_mean = np.dot(X, self.mn)
        if not return_std:
            return y_mean
        
        # Predictive variance: sigma_y^2 = sigma^2 + diag(X @ Sn @ X.T)
        y_var = self.sigma_sq + np.sum(np.dot(X, self.Sn) * X, axis=1)
        return y_mean, np.sqrt(y_var)


if __name__ == "__main__":
    np.random.seed(42)
    
    def test_krr():
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        # Use smaller lambda and larger length_scale for simple linear fit
        model = KernelRidgeRegression(lmbda=1e-5, length_scale=2.0)
        model.fit(X, y)
        y_pred = model.predict(X)
        print(f"KRR Test - Original: {y}, Predicted: {y_pred.round(4)}")
        assert np.allclose(y, y_pred, atol=0.01), "KRR failed accuracy test"

    def test_blr():
        X = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]) # Bias included
        y = np.array([3.0, 5.0, 7.0]) # y = 2*x + 1
        model = BayesianLinearRegression(sigma_sq=1e-5)
        model.fit(X, y)
        y_pred = model.predict(X)
        print(f"BLR Test - Original: {y}, Predicted: {y_pred.round(4)}")
        assert np.allclose(y, y_pred, atol=0.01), "BLR failed accuracy test"

    print("Running unit tests...")
    test_krr()
    test_blr()
    print("All tests passed!")
