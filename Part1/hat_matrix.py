"""
hat_matrix.py — Hat Matrix / Ma trận chiếu (cài đặt từ đầu)
============================================================

Đồ án 2: Data Fitting và Phương pháp OLS
Môn: Toán Ứng Dụng và Thống Kê (MTH00051) — FIT HCMUS

Cơ sở lý thuyết:
-----------------
Hat Matrix (ma trận chiếu):
  H = X(XᵀX)⁻¹Xᵀ ∈ ℝⁿˣⁿ                — Công thức (5)

Tính chất:
  (i)   H² = H               (idempotent)
  (ii)  Hᵀ = H               (đối xứng / symmetric)
  (iii) Eigenvalues ∈ {0, 1}  (chỉ là 0 hoặc 1)
  (iv)  rank(H) = p + 1
  (v)   ŷ = Hy ;  ε̂ = (I − H)y

Leverage:
  hᵢᵢ = phần tử đường chéo thứ i của H.
  Quan sát có hᵢᵢ lớn → influential point.
"""

import numpy as np
import matplotlib.pyplot as plt
from ols_fit import ols_fit
from helper_function import (
    add_intercept, transpose_matrix, matmul, invert_matrix, to_2d_list, to_1d_list
)


# ============================================================
# HÀM CHÍNH: hat_matrix
# ============================================================

def hat_matrix(X):
    """
    Tính Hat Matrix H = X(XᵀX)⁻¹Xᵀ — cài đặt từ đầu (tính tay).
    """
    X_mat = add_intercept(X)
    n = len(X_mat)
    k = len(X_mat[0])

    # Bước 1: Xᵀ
    Xt = transpose_matrix(X_mat)
    
    # Bước 2: XᵀX
    XtX = matmul(Xt, X_mat)

    # Bước 3: (XᵀX)⁻¹
    XtX_inv = invert_matrix(XtX)

    # Bước 4: H = X (XᵀX)⁻¹ Xᵀ
    M1 = matmul(X_mat, XtX_inv)
    H = matmul(M1, Xt)

    # Leverage = đường chéo chính
    leverage = [H[i][i] for i in range(n)]

    # Rank (đối với ma trận chiếu, rank = trace)
    rank = int(round(sum(leverage)))

    # Eigenvalues của ma trận chiếu luôn gồm `rank` số 1 và `n - rank` số 0
    eigenvalues = [1.0] * rank + [0.0] * (n - rank)

    return {
        'H': np.array(H),
        'leverage': np.array(leverage),
        'rank': rank,
        'eigenvalues': np.array(eigenvalues),
        'X_design': np.array(X_mat),
    }


# ============================================================
# KIỂM TRA TÍNH CHẤT
# ============================================================

def verify_hat_properties(H, X, y=None, tol=1e-10):
    """
    Kiểm tra 5 tính chất của Hat Matrix.

    Parameters
    ----------
    H   : np.ndarray (n, n) — Hat matrix
    X   : np.ndarray (n, k) — ma trận thiết kế (đã có intercept)
    y   : np.ndarray (n,), optional — dùng để kiểm tra tính chất (v)
    tol : float — ngưỡng sai số

    Returns
    -------
    dict: kết quả kiểm tra từng tính chất
    """
    n = H.shape[0]
    k = X.shape[1] if X.ndim == 2 else 1
    results = {}

    # (i) Idempotent: H² = H
    H2 = H @ H
    err_idem = np.linalg.norm(H2 - H, 'fro')
    results['idempotent'] = {
        'passed': err_idem < tol,
        'error': err_idem,
        'desc': f'‖H² − H‖_F = {err_idem:.2e}'
    }

    # (ii) Symmetric: Hᵀ = H
    err_sym = np.linalg.norm(H.T - H, 'fro')
    results['symmetric'] = {
        'passed': err_sym < tol,
        'error': err_sym,
        'desc': f'‖Hᵀ − H‖_F = {err_sym:.2e}'
    }

    # (iii) Eigenvalues ∈ {0, 1}
    eigvals = np.linalg.eigvalsh(H)
    all_01 = all(np.isclose(e, 0, atol=tol) or np.isclose(e, 1, atol=tol)
                 for e in eigvals)
    results['eigenvalues_01'] = {
        'passed': all_01,
        'eigenvalues': eigvals,
        'desc': f'Eigenvalues: {np.round(eigvals, 6)}'
    }

    # (iv) rank(H) = p + 1
    rank_H = int(np.sum(eigvals > 0.5))
    results['rank'] = {
        'passed': rank_H == k,
        'rank_H': rank_H,
        'expected': k,
        'desc': f'rank(H) = {rank_H}, expected p+1 = {k}'
    }

    # (v) ŷ = Hy (nếu có y)
    if y is not None:
        y = np.asarray(y, dtype=float).ravel()
        y_hat_H = H @ y
        ols_result = ols_fit(X[:, 1:] if np.allclose(X[:, 0], 1) else X, y)
        y_hat_ols = ols_result['y_hat']
        err_fit = np.linalg.norm(y_hat_H - y_hat_ols)
        results['fitted_values'] = {
            'passed': err_fit < tol,
            'error': err_fit,
            'desc': f'‖Hy − Xβ̂‖ = {err_fit:.2e}'
        }

    return results


# ============================================================
# HÀM PHỤ TRỢ
# ============================================================

def compute_leverage(X):
    """Tính leverage hᵢᵢ = diag(H) mà không cần tạo toàn bộ H."""
    X_mat = add_intercept(X)
    n = len(X_mat)
    k = len(X_mat[0])
    
    Xt = transpose_matrix(X_mat)
    XtX = matmul(Xt, X_mat)
    XtX_inv = invert_matrix(XtX)
    
    leverage = []
    for i in range(n):
        x_i = X_mat[i]
        # temp = x_i^T * XtX_inv
        temp = [sum(x_i[m] * XtX_inv[m][j] for m in range(k)) for j in range(k)]
        # h_ii = temp * x_i
        h_ii = sum(temp[j] * x_i[j] for j in range(k))
        leverage.append(h_ii)
        
    return np.array(leverage)


def fitted_values_from_hat(H, y):
    """
    Tính ŷ = Hy  và  ε̂ = (I − H)y.
    """
    H_mat = to_2d_list(H)
    y_vec = to_1d_list(y)
    n = len(H_mat)
    
    y_hat = [sum(H_mat[i][j] * y_vec[j] for j in range(n)) for i in range(n)]
    residuals = [y_vec[i] - y_hat[i] for i in range(n)]
    
    return {'y_hat': np.array(y_hat), 'residuals': np.array(residuals)}



