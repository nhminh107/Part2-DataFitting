import unittest
import pandas as pd
import numpy as np
import os
import sys
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Vá lỗi thiếu import cho file model_comparison.py của nhóm
import Part2.model_comparison as mc
mc.plt = plt
mc.sns = sns

from Part2.data_pipeline import DataPipeline
from Part2.model_comparison import (
    select_features_vif, select_features_pvalue, 
    calculate_metrics, fit_ols, predict_ols,
    plot_residual_diagnostics, plot_feature_importance
)

class TestEDAOutliersSelection(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        n_samples = 100
        self.X = pd.DataFrame({
            'feat1': np.random.randn(n_samples),
            'feat2': np.random.randn(n_samples),
            'feat3': np.random.randn(n_samples) * 0.1,
            'cat1': np.random.choice(['A', 'B'], n_samples)
        })
        self.X['feat3'] = self.X['feat1'] * 2 + np.random.randn(n_samples) * 0.01
        self.X.loc[0, 'feat2'] = 100.0
        self.X.loc[1, 'feat2'] = -100.0
        self.y = 3 * self.X['feat1'] - 2 * self.X['feat2'] + np.random.randn(n_samples)

    # --- Tests cho EDA ---
    def test_eda_stats_normal(self):
        """Test 1 cho EDA: Chạy với dữ liệu bình thường."""
        pipeline = DataPipeline(imputation_method='mean')
        pipeline.df = self.X.copy()
        try:
            pipeline.EDA()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"EDA failed on normal data: {e}")

    def test_eda_empty_df(self):
        """Test 2 cho EDA: Kiểm tra xử lý DataFrame không có dữ liệu."""
        pipeline = DataPipeline()
        pipeline.df = None
        try:
            pipeline.EDA()
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"EDA crashed on None dataframe: {e}")

    # --- Tests cho Outliers ---
    def test_xu_ly_outliers_drop(self):
        """Test 1 cho Outliers: Phương pháp drop."""
        pipeline = DataPipeline(handle_outliers='drop', imputation_method='mean')
        # Vì fit_transform trả về tuple (X, y) nếu có target ngầm định hoặc explicit
        # Ở đây ta chỉ lấy X
        X_trans = pipeline.fit_transform(self.X)
        if isinstance(X_trans, tuple): X_trans = X_trans[0]
        self.assertLess(len(X_trans), 100)

    def test_xu_ly_outliers_winsorize(self):
        """Test 2 cho Outliers: Phương pháp winsorize."""
        pipeline = DataPipeline(handle_outliers='winsorize', imputation_method='mean')
        X_trans = pipeline.fit_transform(self.X)
        if isinstance(X_trans, tuple): X_trans = X_trans[0]
        self.assertEqual(len(X_trans), 100)
        self.assertLess(X_trans['feat2'].max(), 10.0)

    # --- Tests cho VIF ---
    def test_vif_selection_high_corr(self):
        """Test 1 cho VIF: Loại bỏ biến cộng tuyến."""
        selected = select_features_vif(self.X[['feat1', 'feat3']], threshold=5.0)
        self.assertEqual(len(selected), 1)

    def test_vif_selection_low_corr(self):
        """Test 2 cho VIF: Giữ nguyên biến độc lập."""
        df_indep = pd.DataFrame({'a': [1, 5, 2, 8], 'b': [9, 1, 7, 3]})
        selected = select_features_vif(df_indep, threshold=10.0)
        self.assertEqual(len(selected), 2)

    # --- Tests cho P-value ---
    def test_pvalue_selection_sig(self):
        """Test 1 cho P-value: Giữ lại biến có ý nghĩa."""
        X_sig = pd.DataFrame({'sig': np.linspace(0, 10, 100)})
        y_sig = 2 * X_sig['sig'] + np.random.randn(100) * 0.01
        selected = select_features_pvalue(X_sig, y_sig, alpha=0.05)
        self.assertIn('sig', selected)

    def test_pvalue_selection_noise(self):
        """Test 2 cho P-value: Loại bỏ biến nhiễu."""
        X_noise = pd.DataFrame({'noise': np.random.randn(100)})
        y = np.random.randn(100)
        selected = select_features_pvalue(X_noise, y, alpha=0.01)
        self.assertEqual(len(selected), 0)

    # --- Tests cho OLS Core ---
    def test_fit_ols_simple(self):
        """Test 1 cho fit_ols: Dữ liệu hoàn hảo."""
        X = pd.DataFrame({'x': [1, 2, 3]})
        y = np.array([2, 4, 6]) 
        beta = fit_ols(X, y)
        self.assertAlmostEqual(beta[1], 2.0, places=5)

    def test_fit_ols_with_noise(self):
        """Test 2 cho fit_ols: Dữ liệu có nhiễu."""
        X = pd.DataFrame({'x': np.random.randn(100)})
        y = 3 * X['x'] + 5 + np.random.randn(100) * 0.1
        beta = fit_ols(X, y)
        self.assertAlmostEqual(beta[0], 5.0, delta=0.2)
        self.assertAlmostEqual(beta[1], 3.0, delta=0.2)

    def test_predict_ols_basic(self):
        """Test 1 cho predict_ols."""
        X = pd.DataFrame({'x': [10]})
        beta = [1, 2] 
        pred = predict_ols(X, beta)
        self.assertEqual(pred[0], 21)

    def test_predict_ols_multi(self):
        """Test 2 cho predict_ols: Nhiều dòng."""
        X = pd.DataFrame({'x': [1, 2, 3]})
        beta = [0, 1] 
        preds = predict_ols(X, beta)
        np.testing.assert_array_equal(preds, [1, 2, 3])

    # --- Tests cho Metrics ---
    def test_metrics_mae(self):
        """Test 1 cho metrics: MAE."""
        m = calculate_metrics([1, 2], [1.5, 2.5])
        self.assertEqual(m['MAE'], 0.5)

    def test_metrics_r2(self):
        """Test 2 cho metrics: R2."""
        m = calculate_metrics([1, 2, 3], [1, 2, 3])
        self.assertEqual(m['R2'], 1.0)

    # --- Tests cho Visuals ---
    def test_plot_residuals_save(self):
        """Test 1 cho Plot Residuals: Kiểm tra xuất file."""
        X_mat = np.random.randn(10, 1)
        y = np.random.randn(10)
        plot_residual_diagnostics(X_mat, y, y, ['f1'])
        plot_dir = os.path.join(os.getcwd(), "Part2", "plot")
        self.assertTrue(os.path.exists(os.path.join(plot_dir, "Residual_Diagnostics.png")))

    def test_plot_importance_save(self):
        """Test 2 cho Plot Importance: Kiểm tra xuất file."""
        plot_feature_importance([1, 0.5], ['f1'])
        plot_dir = os.path.join(os.getcwd(), "Part2", "plot")
        self.assertTrue(os.path.exists(os.path.join(plot_dir, "Feature_Importance.png")))

if __name__ == '__main__':
    unittest.main()
