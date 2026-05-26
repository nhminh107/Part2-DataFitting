import sys
import os
import unittest
import numpy as np

# Thêm thư mục cha vào sys.path để import được module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Part1.ols_implementation import ols_fit, predict
from Part1.helper_function import add_intercept

class TestOLSFit(unittest.TestCase):

    def setUp(self):
        # Đặt seed để tái lập kết quả ngẫu nhiên
        np.random.seed(42)

    def test_simple_regression(self):
        """Test hồi quy đơn tuyến tính: y = 2 + 3x (không nhiễu)"""
        x = np.array([1., 2., 3., 4., 5.])
        y = 2.0 + 3.0 * x
        beta_hat, sigma2_hat = ols_fit(x, y)
        
        # Kiểm tra beta = [2, 3]
        np.testing.assert_allclose(beta_hat, [2, 3], atol=1e-10)
        
        # Sai số RSS phải rất nhỏ (sigma2_hat xấp xỉ 0)
        self.assertLess(sigma2_hat, 1e-10)

    def test_multiple_regression(self):
        """Test hồi quy bội: y = 1 + 2x1 + 3x2 + nhiễu"""
        n = 100
        X = np.random.randn(n, 2)
        true_beta = np.array([1.0, 2.0, 3.0])
        y = true_beta[0] + X @ true_beta[1:] + 0.1 * np.random.randn(n)
        
        beta_hat, _ = ols_fit(X, y)
        
        # Vì có nhiễu nên beta chỉ xấp xỉ true_beta, ngưỡng sai số 0.1
        np.testing.assert_allclose(beta_hat, true_beta, atol=0.1)

    def test_sigma2_unbiased(self):
        """Test kỳ vọng của phương sai nhiễu: E[σ̂²] ≈ σ²"""
        sigma_true = 2.0
        estimates = []
        for _ in range(200):
            x = np.random.randn(50, 1)
            y = 1.0 + 3.0 * x.ravel() + sigma_true * np.random.randn(50)
            _, sigma2_hat = ols_fit(x, y)
            estimates.append(sigma2_hat)
            
        mean_s2 = np.mean(estimates)
        # Kỳ vọng của phương sai tính được phải xấp xỉ 4.0
        self.assertLess(abs(mean_s2 - sigma_true**2), 0.5)

    def test_predict_function(self):
        """Test hàm predict() tạo ra dự đoán chính xác"""
        X_train = np.array([[1], [2], [3]])
        y_train = np.array([3, 5, 7]) # y = 1 + 2x
        
        beta_hat, _ = ols_fit(X_train, y_train)
        
        X_test = np.array([[4], [5]])
        y_pred = predict(X_test, beta_hat)
        
        np.testing.assert_allclose(y_pred, [9, 11], atol=1e-10)

    def test_dimension_mismatch(self):
        """Test bắt lỗi khi số lượng mẫu của X và y không khớp"""
        X = np.random.randn(10, 2)
        y = np.random.randn(9) # Thiếu 1 nhãn
        
        with self.assertRaises(ValueError) as context:
            ols_fit(X, y)
        
        self.assertIn("không khớp", str(context.exception))

    def test_insufficient_samples(self):
        """Test bắt lỗi khi n <= p + 1 (Không đủ mẫu để hồi quy)"""
        X = np.random.randn(2, 3) # 2 mẫu, 3 đặc trưng (cần thêm intercept -> p=4)
        y = np.random.randn(2)
        
        with self.assertRaises(ValueError) as context:
            ols_fit(X, y)
            
        self.assertIn("Cần n > p+1", str(context.exception))

    def test_singular_matrix(self):
        """Test bắt lỗi khi ma trận thiết kế bị suy biến (Đa cộng tuyến hoàn hảo)"""
        # Cột 1 và cột 2 giống hệt nhau
        X = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
        y = np.array([2, 4, 6, 8, 10])
        
        with self.assertRaises(ValueError) as context:
            ols_fit(X, y)
            
        self.assertIn("suy biến", str(context.exception))

if __name__ == '__main__':
    unittest.main(verbosity=2)
