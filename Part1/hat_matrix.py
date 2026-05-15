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
    Tính Hat Matrix H = X(XᵀX)⁻¹Xᵀ và kiểm tra tính Idempotent.
    """
    X_mat = add_intercept(X)
    n = len(X_mat)

    # Bước 1: Xᵀ
    Xt = transpose_matrix(X_mat)
    
    # Bước 2: XᵀX
    XtX = matmul(Xt, X_mat)

    # Bước 3: (XᵀX)⁻¹
    XtX_inv = invert_matrix(XtX)

    # Bước 4: H = X (XᵀX)⁻¹ Xᵀ
    M1 = matmul(X_mat, XtX_inv)
    H = matmul(M1, Xt)

    # Kiểm tra tính Idempotent (H² = H)
    H2 = matmul(H, H)
    is_idempotent = True
    for i in range(n):
        for j in range(n):
            if abs(H2[i][j] - H[i][j]) > 1e-9:
                is_idempotent = False
                break
        if not is_idempotent:
            break

    return H, is_idempotent






