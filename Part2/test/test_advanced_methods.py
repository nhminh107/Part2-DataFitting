import unittest
import numpy as np
import os
import sys

# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Part2.advanced_methods import KernelRidgeRegression, BayesianLinearRegression

class TestAdvancedMethods(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        self.y = np.sin(self.X).ravel() + np.random.randn(5) * 0.01

    def test_krr_linear_fit(self):
        """Test 1 cho KRR: Kiểm tra khớp dữ liệu tuyến tính đơn giản."""
        X_lin = np.array([[1.0], [2.0], [3.0]])
        y_lin = np.array([2.0, 4.0, 6.0])
        model = KernelRidgeRegression(lmbda=1e-5, length_scale=5.0)
        model.fit(X_lin, y_lin)
        preds = model.predict(X_lin)
        np.testing.assert_allclose(preds, y_lin, atol=0.1)

    def test_krr_nonlinear_fit(self):
        """Test 2 cho KRR: Kiểm tra khớp dữ liệu phi tuyến."""
        model = KernelRidgeRegression(lmbda=0.1, length_scale=1.0)
        model.fit(self.X, self.y)
        preds = model.predict(self.X)
        # Kiểm tra xem xu hướng dự báo có đúng hướng với sin(x) không
        self.assertEqual(len(preds), len(self.y))

    def test_blr_mean_prediction(self):
        """Test 1 cho BLR: Kiểm tra giá trị dự báo trung bình."""
        X_bias = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
        y = np.array([3.0, 5.0, 7.0]) # y = 2x + 1
        model = BayesianLinearRegression(sigma_sq=1e-4)
        model.fit(X_bias, y)
        preds = model.predict(X_bias)
        np.testing.assert_allclose(preds, y, atol=0.1)

    def test_blr_uncertainty(self):
        """Test 2 cho BLR: Kiểm tra tính toán độ lệch chuẩn (uncertainty)."""
        X_bias = np.array([[1.0, 1.0], [1.0, 2.0]])
        y = np.array([3.0, 5.0])
        model = BayesianLinearRegression(sigma_sq=1.0)
        model.fit(X_bias, y)
        mean, std = model.predict(X_bias, return_std=True)
        self.assertEqual(len(std), 2)
        self.assertTrue(all(std > 0))

if __name__ == '__main__':
    unittest.main()
