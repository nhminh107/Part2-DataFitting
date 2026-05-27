import unittest
import numpy as np
import os
import sys

# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Part1.ridge_lasso import ridge_fit, lasso_fit
from Part1.helper_function import add_intercept, matvec

class TestRegularization(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.n = 100
        self.X = np.random.randn(self.n, 3)
        # y phụ thuộc chủ yếu vào biến 1 và 2, biến 3 là nhiễu
        self.y = 2.0 + 5.0 * self.X[:, 0] - 3.0 * self.X[:, 1] + 0.1 * self.X[:, 2] + np.random.randn(self.n) * 0.1
        self.X_bias = add_intercept(self.X.tolist())

    def test_ridge_fit_accuracy(self):
        """Test 1 cho Ridge: Kiểm tra độ chính xác trên dữ liệu đơn giản."""
        beta = ridge_fit(self.X_bias, self.y.tolist(), lam=0.1)
        self.assertAlmostEqual(beta[0], 2.0, delta=0.2)
        self.assertAlmostEqual(beta[1], 5.0, delta=0.2)

    def test_ridge_shrinkage(self):
        """Test 2 cho Ridge: Kiểm tra hiện tượng co rút hệ số khi lambda tăng."""
        beta_small = ridge_fit(self.X_bias, self.y.tolist(), lam=0.01)
        beta_large = ridge_fit(self.X_bias, self.y.tolist(), lam=100.0)
        # Hệ số (trừ intercept) của Large lambda phải nhỏ hơn Small lambda
        self.assertLess(np.sum(np.square(beta_large[1:])), np.sum(np.square(beta_small[1:])))

    def test_lasso_fit_sparsity(self):
        """Test 1 cho Lasso: Kiểm tra tính thưa hóa (loại bỏ biến nhiễu)."""
        # Với lambda rất lớn, biến 3 (hệ số 0.1) phải về 0
        beta = lasso_fit(self.X_bias, self.y.tolist(), lam=20.0)
        self.assertAlmostEqual(beta[3], 0.0, places=1)

    def test_lasso_convergence(self):
        """Test 2 cho Lasso: Kiểm tra hội tụ với dữ liệu tuyến tính hoàn hảo."""
        X_simple = [[1, 1], [1, 2]]
        y_simple = [1, 2] # y = x
        beta = lasso_fit(X_simple, y_simple, lam=0.001)
        self.assertAlmostEqual(beta[1], 1.0, delta=0.05)

if __name__ == '__main__':
    unittest.main()
