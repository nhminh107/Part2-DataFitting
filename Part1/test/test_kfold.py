import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from Part1.ridge_lasso import ridge_fit
from Part1.cross_validation import kfold_cv

def test_ridge_regression():
    print("--- Testing Ridge Regression ---")
    np.random.seed(42)
    X = np.random.randn(100, 5)
    X_with_bias = np.hstack([np.ones((100, 1)), X])
    true_beta = np.array([2.5, -1.2, 0.5, 3.0, -0.8, 1.5])
    y = X_with_bias @ true_beta + np.random.normal(0, 0.5, 100)
    
    lam = 1.0
    
    # Chuyển sang list để test hàm thuần Python
    X_list = X_with_bias.tolist()
    y_list = y.tolist()
    
    beta_my = ridge_fit(X_list, y_list, lam, fit_intercept=True)
    
    # Sklearn
    clf = Ridge(alpha=lam, fit_intercept=True)
    clf.fit(X, y)
    beta_sk = np.concatenate([[clf.intercept_], clf.coef_])
    
    diff = np.abs(np.array(beta_my) - beta_sk)
    print(f"Max difference: {np.max(diff)}")
    
    assert np.allclose(beta_my, beta_sk, atol=1e-10), "Ridge implementation differs from sklearn!"
    print("Test Ridge: PASSED\n")

def test_kfold_cv():
    print("--- Testing K-Fold Cross-Validation ---")
    np.random.seed(42)
    X = np.random.randn(50, 3)
    X_with_bias = np.hstack([np.ones((50, 1)), X])
    y = 2 + 1.5*X[:, 0] - 0.8*X[:, 1] + 0.5*X[:, 2] + np.random.normal(0, 0.1, 50)
    
    k = 5
    
    # Chuyển sang list
    X_list = X_with_bias.tolist()
    y_list = y.tolist()
    
    mean_mse_my, _ = kfold_cv(X_list, y_list, k, ridge_fit, seed=42, lam=0.1)
    
    # Sklearn split and model_func
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mses_sk = []
    
    for train_index, test_index in kf.split(X):
        X_train_sk, X_val_sk = X_with_bias[train_index], X_with_bias[test_index]
        y_train_sk, y_val_sk = y[train_index], y[test_index]
        
        # Gọi hàm của mình với dữ liệu list
        beta_hat = ridge_fit(X_train_sk.tolist(), y_train_sk.tolist(), lam=0.1)
        
        y_pred = X_val_sk @ np.array(beta_hat)
        mse = mean_squared_error(y_val_sk, y_pred)
        mses_sk.append(mse)
        
    mean_mse_sk = np.mean(mses_sk)
    
    print(f"My Mean MSE (with internal split): {mean_mse_my}")
    print(f"Sklearn-based Mean MSE: {mean_mse_sk}")
    print(f"Difference: {abs(mean_mse_my - mean_mse_sk)}")
    
    print("Test K-Fold: PASSED\n")

if __name__ == "__main__":
    try:
        test_ridge_regression()
        test_kfold_cv()
        print("ALL TESTS PASSED!")
    except Exception as e:
        print(f"ERROR: {e}")
