import numpy as np

def soft_threshold(rho, lam):
    if rho < -lam:
        return rho + lam
    elif rho > lam:
        return rho - lam
    else:
        return 0.0

def lasso_fit(X, y, lam, epochs=1000, fit_intercept=True):
    n = len(X)
    p = len(X[0])
    
    # Khởi tạo beta
    beta = [0.0] * p
    
    # Tiền tính toán z_j = sum_{i} X_{ij}^2
    z = [0.0] * p
    for j in range(p):
        z[j] = sum(X[i][j]**2 for i in range(n))
        
    for epoch in range(epochs):
        max_diff = 0.0
        for j in range(p):
            rho_j = 0.0
            for i in range(n):
                y_pred_i = sum(X[i][k] * beta[k] for k in range(p))
                rho_j += X[i][j] * (y[i] - y_pred_i + X[i][j] * beta[j])
                
            old_beta_j = beta[j]
            
            if fit_intercept and j == 0:
                if z[j] != 0:
                    beta[j] = rho_j / z[j]
            else:
                if z[j] != 0:
                    beta[j] = soft_threshold(rho_j, lam) / z[j]
                    
            max_diff = max(max_diff, abs(beta[j] - old_beta_j))
            
        if max_diff < 1e-4:
            break
            
    return beta

if __name__ == "__main__":
    from sklearn.linear_model import Lasso
    
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = 2.0 + 1.5 * X[:, 0] - 0.5 * X[:, 1] + 0.0 * X[:, 2] + np.random.randn(100) * 0.1
    
    X_bias = np.hstack([np.ones((100, 1)), X])
    
    lam = 1.0
    
    beta_my = lasso_fit(X_bias.tolist(), y.tolist(), lam=1.0)
    
    clf = Lasso(alpha=1.0 / len(y), fit_intercept=False)
    clf_true = Lasso(alpha=2.0 / len(y), fit_intercept=True)
    clf_true.fit(X, y)
    
    print("My beta:", beta_my)
    print("Sklearn beta:", [clf_true.intercept_] + clf_true.coef_.tolist())
