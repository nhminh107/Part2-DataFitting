import matplotlib.pyplot as plt
try:
    from Part1.helper_function import transpose_matrix, matmul, add_matrix, invert_matrix
except ModuleNotFoundError:
    from helper_function import transpose_matrix, matmul, add_matrix, invert_matrix

def ridge_fit(X, y, lam, fit_intercept=True):
    """
    Cài đặt Hồi quy Ridge.
    
    Input:
    X : list of lists (Ma trận n x p)
    y : list (Vector n x 1)
    lam : float (Tham số điều chỉnh)
    fit_intercept : bool (Nếu True, không regularize hệ số chặn beta_0)
    
    Output:
    beta : list (Vector hệ số hồi quy)
    """
    n = len(X)
    p = len(X[0])
    
    # Đảm bảo y là ma trận cột để dùng hàm multiply
    if not isinstance(y[0], list):
        y_mat = [[yi] for yi in y]
    else:
        y_mat = y

    # Tính X_transpose
    XT = transpose_matrix(X)
    
    # Tính X_transpose * X
    XTX = matmul(XT, X)
    
    # Tạo ma trận lam * I
    # I là ma trận đơn vị p x p
    lamI = [[0.0 for _ in range(p)] for _ in range(p)]
    for i in range(p):
        if fit_intercept and i == 0:
            lamI[i][i] = 0.0
        else:
            lamI[i][i] = float(lam)
            
    # A = XTX + lamI
    A = add_matrix(XTX, lamI)
    
    # Nghịch đảo ma trận A
    A_inv = invert_matrix(A)
    
    # Tính X_transpose * y
    XTy = matmul(XT, y_mat)
    
    # Tính beta = A_inv * XTy
    beta_mat = matmul(A_inv, XTy)
    
    # Chuyển beta về dạng list phẳng để dễ sử dụng
    beta = [row[0] for row in beta_mat]
    
    return beta

def soft_threshold(rho, lam):
    if rho < -lam:
        return rho + lam
    elif rho > lam:
        return rho - lam
    else:
        return 0.0

def lasso_fit(X, y, lam, epochs=1000, fit_intercept=True):
    """
    Cài đặt Hồi quy Lasso bằng thuật toán Coordinate Descent.
    
    Input:
    X : list of lists (Ma trận n x p)
    y : list (Vector n x 1)
    lam : float
    epochs : int (Số vòng lặp tối đa)
    fit_intercept : bool
    
    Output:
    beta : list (Vector hệ số hồi quy)
    """
    n = len(X)
    p = len(X[0])
    
    # Chuyển y thành 1D nếu nó là 2D
    y_1d = [row[0] if isinstance(row, list) else row for row in y]
    
    beta = [0.0] * p
    z = [0.0] * p
    for j in range(p):
        z[j] = sum(X[i][j]**2 for i in range(n))
        
    for epoch in range(epochs):
        max_diff = 0.0
        for j in range(p):
            if z[j] == 0:
                continue
                
            rho_j = 0.0
            for i in range(n):
                y_pred_i = sum(X[i][k] * beta[k] for k in range(p))
                rho_j += X[i][j] * (y_1d[i] - y_pred_i + X[i][j] * beta[j])
                
            old_beta_j = beta[j]
            
            if fit_intercept and j == 0:
                beta[j] = rho_j / z[j]
            else:
                beta[j] = soft_threshold(rho_j, lam / 2.0) / z[j]
                
            max_diff = max(max_diff, abs(beta[j] - old_beta_j))
            
        if max_diff < 1e-6:
            break
            
    return beta


def plot_ridge_trace(X, y, alphas, title="Ridge Trace"):
    """
    Vẽ biểu đồ Ridge Trace.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plot_ridge_trace. Install it with: pip install matplotlib")
        
    coefs = []
    for a in alphas:
        beta = ridge_fit(X, y, a)
        coefs.append(beta[1:])
        
    plt.figure(figsize=(10, 6))
    plt.plot(alphas, coefs)
    plt.xscale('log')
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('Coefficients (Beta)')
    plt.title(title)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()

if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(1)

    def make_xy(n, p, noise, collinear=False):
        X = rng.normal(size=(n, p))
        if collinear and p >= 2:
            X[:, 1] = X[:, 0] + 0.1 * rng.normal(size=n)
        beta = rng.normal(size=p + 1)
        y = beta[0] + X @ beta[1:] + noise * rng.normal(size=n)
        return X, y

    cases = [
        (120, 3, 0.2, False),
        (50, 1, 0.2, False),
        (200, 12, 0.2, False),
        (120, 4, 5.0, False),
        (150, 4, 0.2, True),
    ]

    for idx, (n, p, noise, collinear) in enumerate(cases, 1):
        print(f"ridge_fit case {idx}: n={n} p={p} noise={noise} collinear={collinear}")
        X, y = make_xy(n, p, noise, collinear)
        X_bias = np.hstack([np.ones((n, 1)), X])
        beta_my = ridge_fit(X_bias.tolist(), y.tolist(), lam=1.0, fit_intercept=True)
        lamI = np.diag([0.0] + [1.0] * p)
        beta_np = np.linalg.inv(X_bias.T @ X_bias + lamI) @ (X_bias.T @ y)
        np.testing.assert_allclose(beta_my, beta_np, rtol=1e-4, atol=1e-4)
