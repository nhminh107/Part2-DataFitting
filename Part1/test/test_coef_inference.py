import sys
import os
import unittest
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ols_implementation import ols_fit, coef_inference

class TestCoefInference(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(42)

    def make_xy(self, n, p, noise, collinear=False):
        X = self.rng.normal(size=(n, p))
        if collinear and p >= 2:
            X[:, 1] = X[:, 0] + 0.1 * self.rng.normal(size=n)
        beta = self.rng.normal(size=p + 1)
        y = beta[0] + X @ beta[1:] + noise * self.rng.normal(size=n)
        return X, y

    def test_coef_inference_standard_errors(self):
        """Test Standard Errors calculation against numpy reference"""
        n, p, noise = 120, 3, 0.1
        X, y = self.make_xy(n, p, noise)
        
        # Fit model
        beta_hat, sigma2 = ols_fit(X.tolist(), y.tolist())
        
        # Prepare X with intercept for reference
        Xc = np.hstack([np.ones((n, 1)), X])
        XtX_inv = np.linalg.inv(Xc.T @ Xc)
        se_ref = np.sqrt(sigma2 * np.diag(XtX_inv))
        
        # Call coef_inference
        beta_hat_2d = [[b] for b in beta_hat]
        inf = coef_inference(Xc.tolist(), y.tolist(), beta_hat_2d, sigma2)
        
        np.testing.assert_allclose(inf["Standard Errors"], se_ref, rtol=1e-3, atol=1e-3)

    def test_coef_inference_t_stats(self):
        """Test t-statistics calculation: t = beta / SE"""
        n, p, noise = 60, 1, 0.1
        X, y = self.make_xy(n, p, noise)
        beta_hat, sigma2 = ols_fit(X.tolist(), y.tolist())
        Xc = np.hstack([np.ones((n, 1)), X])
        
        beta_hat_2d = [[b] for b in beta_hat]
        inf = coef_inference(Xc.tolist(), y.tolist(), beta_hat_2d, sigma2)
        
        se = inf["Standard Errors"]
        t_stats = inf["t_stats"]
        
        for i in range(len(beta_hat)):
            self.assertAlmostEqual(t_stats[i], beta_hat[i] / se[i], places=7)

    def test_coef_inference_intervals(self):
        """Test Confidence Intervals: beta +/- t_critical * SE"""
        n, p, noise = 200, 5, 0.2
        X, y = self.make_xy(n, p, noise)
        beta_hat, sigma2 = ols_fit(X.tolist(), y.tolist())
        Xc = np.hstack([np.ones((n, 1)), X])
        
        beta_hat_2d = [[b] for b in beta_hat]
        inf = coef_inference(Xc.tolist(), y.tolist(), beta_hat_2d, sigma2)
        
        t_crit = inf["t_critical"]
        se = inf["Standard Errors"]
        intervals = inf["Confidence Intervals"]
        
        for i in range(len(beta_hat)):
            expected_lower = beta_hat[i] - t_crit * se[i]
            expected_upper = beta_hat[i] + t_crit * se[i]
            self.assertAlmostEqual(intervals[i][0], expected_lower, places=7)
            self.assertAlmostEqual(intervals[i][1], expected_upper, places=7)

if __name__ == '__main__':
    unittest.main(verbosity=2)
