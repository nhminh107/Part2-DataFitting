import unittest
import pandas as pd
import numpy as np
import sys
import os

# Thêm thư mục cha vào sys.path để import được module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from ols_implementation import model_metrics, calculate_vif, ols_fit

class TestOLSMetricsLoc(unittest.TestCase):
    def setUp(self):
        #tạo dữ liệu giả lập
        np.random.seed(42)
        n = 100
        x1 = np.linspace(0, 10, n)
        x2 = 0.5 * x1 + np.random.normal(0, 0.1, n) # x2 tương quan cao với x1
        y = 2 * x1 + 1.5 * x2 + 5 + np.random.normal(0, 1, n)
        
        self.data = pd.DataFrame({'x1': x1, 'x2': x2, 'y': y})
        self.X = self.data[['x1', 'x2']]
        self.y = self.data['y']
        
        #tính beta_hat bằng ols_fit
        self.beta_hat, _ = ols_fit(self.X.values.tolist(), self.y.tolist())
        
        #tính y_pred
        #cần thêm intercept để tính y_pred
        from helper_function import add_intercept, matvec
        X_with_const = add_intercept(self.X.values.tolist())
        self.y_pred = matvec(X_with_const, self.beta_hat)
        
        self.y_true = self.y.tolist()
        self.p = 2
        self.n = n

    def test_model_metrics(self):
        results = model_metrics(self.y_true, self.y_pred, self.p)
        
        #kiểm chứng R2 bằng sklearn
        r2_sklearn = r2_score(self.y_true, self.y_pred)
        self.assertAlmostEqual(results['R2'], r2_sklearn, places=6)
        
        #kiểm chứng Adjusted R2
        adj_r2_expected = 1 - (1 - r2_sklearn) * (self.n - 1) / (self.n - self.p - 1)
        self.assertAlmostEqual(results['Adjusted R2'], adj_r2_expected, places=6)
        
        print(f"\nModel Metrics Test:")
        print(f"R2: {results['R2']:.4f} (Sklearn: {r2_sklearn:.4f})")
        print(f"F-statistic: {results['F-statistic']:.4f}")

    def test_vif(self):
        vif_results = calculate_vif(self.X)
        
        #kiểm chứng x1 bằng sklearn
        model_aux = LinearRegression()
        model_aux.fit(self.X[['x2']], self.X['x1'])
        r2_aux = model_aux.score(self.X[['x2']], self.X['x1'])
        vif_expected = 1 / (1 - r2_aux)
        
        self.assertAlmostEqual(vif_results['x1'], vif_expected, places=6)
        print(f"\nVIF Test:")
        print(f"VIF x1: {vif_results['x1']:.4f} (Expected: {vif_expected:.4f})")
        print(f"VIF x2: {vif_results['x2']:.4f}")

if __name__ == '__main__':
    unittest.main()