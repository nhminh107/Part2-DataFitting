"""
ols_fit.py — Ordinary Least Squares Fitting (cài đặt từ đầu)
=============================================================

Đồ án 2: Data Fitting và Phương pháp OLS
Môn: Toán Ứng Dụng và Thống Kê (MTH00051) — FIT HCMUS

Cơ sở lý thuyết:
-----------------
Mô hình hồi quy tuyến tính:  y = Xβ + ε
  - X ∈ ℝⁿˣ⁽ᵖ⁺¹⁾  : ma trận thiết kế, cột đầu toàn 1 (intercept)
  - β ∈ ℝᵖ⁺¹       : vector hệ số cần ước lượng
  - ε ∈ ℝⁿ          : nhiễu, E[ε|X]=0, Var(ε|X)=σ²Iₙ

Nghiệm OLS (Normal Equations):
  β̂ = (XᵀX)⁻¹ Xᵀy                     — Công thức (4)

Ước lượng phương sai nhiễu:
  σ̂² = RSS / (n − p − 1)               — Công thức (7)
"""

import numpy as np
import matplotlib.pyplot as plt
from helper_function import (
    to_2d_list, to_1d_list, transpose_matrix, 
    matmul, matvec, invert_matrix, add_intercept
)

# ============================================================
# HÀM DỰ ĐOÁN
# ============================================================

def predict(X, beta_hat):
    """
    Dự đoán ŷ = Xβ̂ cho dữ liệu mới.
    """
    X_mat = add_intercept(X)
    b_vec = to_1d_list(beta_hat)
    return np.array(matvec(X_mat, b_vec))


# ============================================================
# HÀM CHÍNH: ols_fit
# ============================================================

def ols_fit(X, y):
    """
    Ordinary Least Squares
    """
    X_mat = add_intercept(X)
    y_vec = to_1d_list(y)
    
    n = len(X_mat)
    k = len(X_mat[0])

    if n != len(y_vec):
        raise ValueError(f"Số hàng X ({n}) và chiều dài y ({len(y_vec)}) không khớp.")
    if n <= k:
        raise ValueError(f"Cần n > p+1. Hiện n={n}, p+1={k}.")

    # Bước 1: Xᵀ
    Xt = transpose_matrix(X_mat)
    
    # Bước 2: XᵀX
    XtX = matmul(Xt, X_mat)

    # Bước 3: (XᵀX)⁻¹ 
    XtX_inv = invert_matrix(XtX)

    # Bước 4: Xᵀy
    Xty = matvec(Xt, y_vec)

    # Bước 5: β̂ = (XᵀX)⁻¹ Xᵀy
    beta_hat = matvec(XtX_inv, Xty)

    # Tính σ̂² = RSS / (n - p - 1)
    y_hat = matvec(X_mat, beta_hat)
    residuals = [y_vec[i] - y_hat[i] for i in range(n)]
    rss = sum(r * r for r in residuals)
    sigma2_hat = rss / (n - k)

    return beta_hat, sigma2_hat



