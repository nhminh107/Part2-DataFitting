import matplotlib.pyplot as plt
from Part1.helper_function import transpose_matrix, matmul, add_matrix, invert_matrix

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


