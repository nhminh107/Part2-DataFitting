import unittest
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from ols_implementation import model_metrics, calculate_vif

class TestOLSMetrics(unittest.TestCase):
    def setUp(self):
        # Tạo dữ liệu giả lập
        # y = 2*x1 + 3*x2 + 5 + noise
        self.data = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'x2': [1.1, 1.9, 3.2, 3.8, 5.1, 6.2, 6.9, 8.1, 9.2, 9.8], # x2 tương quan cao với x1
            'y': [10.1, 13.2, 17.5, 19.8, 25.1, 28.2, 31.9, 36.5, 41.2, 44.8]
        })
        self.X = self.data[['x1', 'x2']]
        self.y = self.data['y']
        
        # Huấn luyện bằng sklearn để lấy y_pred chuẩn
        self.model = LinearRegression()
        self.model.fit(self.X, self.y)
        self.y_pred = self.model.predict(self.X).tolist()
        self.y_true = self.y.tolist()
        self.p = 2 # số lượng feature
        self.n = len(self.y_true)

    def test_model_metrics(self):
        # 1. Chạy hàm tự viết
        results = model_metrics(self.y_true, self.y_pred, self.p)
        
        # 2. Kiểm chứng R2 bằng sklearn
        r2_sklearn = r2_score(self.y_true, self.y_pred)
        self.assertAlmostEqual(results['R2'], r2_sklearn, places=6)
        
        # 3. Kiểm chứng TSS, RSS
        y_mean = sum(self.y_true) / self.n
        tss_manual = sum((yi - y_mean)**2 for yi in self.y_true)
        rss_manual = sum((yi - yp)**2 for yi, yp in zip(self.y_true, self.y_pred))
        
        self.assertAlmostEqual(results['TSS'], tss_manual, places=6)
        self.assertAlmostEqual(results['RSS'], rss_manual, places=6)
        
        # 4. Kiểm chứng Adjusted R2
        adj_r2_manual = 1 - (1 - r2_sklearn) * (self.n - 1) / (self.n - self.p - 1)
        self.assertAlmostEqual(results['Adjusted R2'], adj_r2_manual, places=6)
        
        # 5. Kiểm chứng F-statistic
        f_stat_manual = ((tss_manual - rss_manual) / self.p) / (rss_manual / (self.n - self.p - 1))
        self.assertAlmostEqual(results['F-statistic'], f_stat_manual, places=6)

    def test_vif(self):
        # Chạy hàm tự viết
        vif_results = calculate_vif(self.X)
        
        # Kiểm chứng thủ công cho x1: x1 ~ x2
        # x1 = beta0 + beta1*x2
        x1_target = self.X['x1'].tolist()
        x2_feature = self.X[['x2']]
        
        model_aux = LinearRegression()
        model_aux.fit(x2_feature, x1_target)
        r2_aux = model_aux.score(x2_feature, x1_target)
        vif_expected = 1 / (1 - r2_aux)
        
        self.assertAlmostEqual(vif_results['x1'], vif_expected, places=6)
        print(f"\nVIF x1: {vif_results['x1']:.4f} (Expected: {vif_expected:.4f})")
        print(f"VIF x2: {vif_results['x2']:.4f}")

if __name__ == '__main__':
    unittest.main()
