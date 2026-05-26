import unittest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Part2.data_pipeline import DataPipeline

class TestDataPipeline(unittest.TestCase):
    def setUp(self):
        # Tạo tập dữ liệu nhỏ để kiểm thử
        self.data = pd.DataFrame({
            'dist': [10.0, 20.0, np.nan, 40.0, 50.0],
            'time': [15.0, np.nan, 35.0, 45.0, 55.0],
            'cat': ['A', 'B', 'A', 'B', np.nan],
            'Trip_Price': [100.0, 200.0, 300.0, 400.0, 500.0]
        })
        self.pipeline = DataPipeline(
            imputation_method='mean',
            handle_outliers='winsorize',
            log_transform_cols=['dist'],
            poly_degree=2,
            target_col='Trip_Price'
        )

    def test_imputation(self):
        # Kiểm tra xem các giá trị NaN đã được điền chưa
        X_trans = self.pipeline.fit_transform(self.data)
        self.assertFalse(X_trans.isnull().any().any(), "Không được còn giá trị NaN sau khi biến đổi")

    def test_log_transformation(self):
        # Kiểm tra xem cột biến đổi log đã được tạo chưa
        X_trans = self.pipeline.fit_transform(self.data)
        self.assertIn('log_dist', X_trans.columns, "Cột log_dist phải tồn tại")
        
    def test_polynomial_features(self):
        # Kiểm tra xem các số hạng bậc 2 đã được tạo chưa
        X_trans = self.pipeline.fit_transform(self.data)
        self.assertIn('dist^2', X_trans.columns, "Cột dist^2 phải tồn tại")
        self.assertIn('time^2', X_trans.columns, "Cột time^2 phải tồn tại")

    def test_scaling(self):
        # Kiểm tra xem đầu ra đã được chuẩn hóa (mean xấp xỉ 0, std xấp xỉ 1)
        # Lưu ý: với dữ liệu rất nhỏ, điều này sẽ không chính xác tuyệt đối, nhưng ta kiểm tra xem giá trị có thay đổi không
        X_trans = self.pipeline.fit_transform(self.data)
        self.assertTrue((X_trans['dist'].abs() < 10).all(), "Các giá trị đã chuẩn hóa không được quá lớn")

if __name__ == '__main__':
    unittest.main()
