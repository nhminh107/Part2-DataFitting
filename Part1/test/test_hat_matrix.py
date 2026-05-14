import sys
import os
import unittest
import numpy as np

# Thêm thư mục cha vào sys.path để import được module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hat_matrix import hat_matrix
from helper_function import add_intercept

class TestHatMatrix(unittest.TestCase):

    def setUp(self):
        # Đặt seed để tái lập kết quả ngẫu nhiên
        np.random.seed(42)

    def test_hat_matrix_idempotent(self):
        """Test hàm hat_matrix trả về đúng H và kiểm tra idempotent thành công (is_idempotent = True)"""
        X = np.random.randn(20, 3)
        H, is_idempotent = hat_matrix(X)
        
        # is_idempotent phải là True theo tính toán của hàm
        self.assertTrue(is_idempotent)
        
        # Kiểm tra H^2 = H (thủ công bằng numpy để verify chéo)
        H_np = np.array(H)
        np.testing.assert_allclose(H_np @ H_np, H_np, atol=1e-9)

    def test_hat_matrix_symmetric(self):
        """Test H là ma trận đối xứng (H^T = H)"""
        X = np.random.randn(15, 2)
        H, _ = hat_matrix(X)
        H_np = np.array(H)
        
        np.testing.assert_allclose(H_np.T, H_np, atol=1e-9)

if __name__ == '__main__':
    unittest.main(verbosity=2)
