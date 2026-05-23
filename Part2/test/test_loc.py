import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add root directory to sys.path to allow imports from Part1 and Part2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Part2.data_pipeline import DataPipeline
from Part2.model_comparison import select_features_vif, select_features_pvalue

class TestLocTasks(unittest.TestCase):
    def setUp(self):
        # Create a dummy dataset with outliers and multicollinearity
        np.random.seed(42)
        n_samples = 100
        self.X = pd.DataFrame({
            'feat1': np.random.randn(n_samples),
            'feat2': np.random.randn(n_samples),
            'feat3': np.random.randn(n_samples) * 0.1, # Will be highly correlated with feat1 later
            'cat1': np.random.choice(['A', 'B'], n_samples)
        })
        # Add multicollinearity: feat3 is almost feat1
        self.X['feat3'] = self.X['feat1'] * 2 + np.random.randn(n_samples) * 0.01
        
        # Add outliers to feat2
        self.X.loc[0, 'feat2'] = 100.0
        self.X.loc[1, 'feat2'] = -100.0
        
        # Target variable
        self.y = 3 * self.X['feat1'] - 2 * self.X['feat2'] + np.random.randn(n_samples)

    def test_outlier_dropping(self):
        # Create data where only 2 rows are definitely outliers
        np.random.seed(42)
        X = pd.DataFrame({'feat1': np.linspace(0, 10, 100)})
        y = 2 * X['feat1'] + np.random.randn(100) * 0.1
        
        # Add 2 extreme outliers
        X.loc[0, 'feat1'] = 100.0
        X.loc[1, 'feat1'] = -100.0
        
        pipeline = DataPipeline(handle_outliers='drop', imputation_method='mean')
        X_trans, y_trans = pipeline.fit_transform(X, y)
        
        # Check if exactly 2 rows were dropped
        print(f"Original len: {len(X)}, Transformed len: {len(X_trans)}")
        self.assertEqual(len(X_trans), 98)
        self.assertNotIn(0, X_trans.index)
        self.assertNotIn(1, X_trans.index)

    def test_vif_selection(self):
        # feat1 and feat3 are highly correlated, one should be dropped
        pipeline = DataPipeline(handle_outliers='winsorize', imputation_method='mean')
        X_trans = pipeline.fit_transform(self.X)
        
        print("\n--- VIF Selection Debug ---")
        selected_features = select_features_vif(X_trans, threshold=5.0)
        print(f"VIF Selected features: {selected_features}")
        self.assertTrue(len(selected_features) < len(X_trans.columns))

    def test_pvalue_selection(self):
        # Create a simpler dataset for p-value test
        np.random.seed(42)
        X = pd.DataFrame({
            'sig1': np.random.randn(100),
            'sig2': np.random.randn(100),
            'noise': np.random.randn(100)
        })
        # y only depends on sig1 and sig2
        y = 5 * X['sig1'] - 3 * X['sig2'] + np.random.randn(100) * 0.1
        
        pipeline = DataPipeline(handle_outliers='winsorize', imputation_method='mean')
        X_trans, y_trans = pipeline.fit_transform(X, y)
        
        print("\n--- P-value Selection Debug ---")
        selected_features = select_features_pvalue(X_trans, y_trans, alpha=0.05)
        print(f"P-value Selected features: {selected_features}")
        
        self.assertIn('sig1', selected_features)
        self.assertIn('sig2', selected_features)
        self.assertNotIn('noise', selected_features)


if __name__ == '__main__':
    unittest.main()
