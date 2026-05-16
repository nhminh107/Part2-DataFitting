import numpy as np

# ============================================================
# CÁC HÀM ĐẠI SỐ TUYẾN TÍNH (TÍNH TAY) & PHỤ TRỢ
# ============================================================

def to_2d_list(X):
    """Chuyển đổi input thành mảng 2 chiều (list of lists)."""
    X_list = np.asarray(X).tolist()
    if not isinstance(X_list[0], (list, tuple)):
        return [[float(x)] for x in X_list]
    return [[float(val) for val in row] for row in X_list]

def to_1d_list(y):
    """Chuyển đổi input thành mảng 1 chiều (list)."""
    return [float(x) for x in np.asarray(y).ravel().tolist()]

def transpose_matrix(A):
    """Chuyển vị ma trận."""
    return [[A[i][j] for i in range(len(A))] for j in range(len(A[0]))]

def matmul(A, B):
    """Nhân hai ma trận."""
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    C = [[0.0] * p for _ in range(n)]
    for i in range(n):
        for j in range(p):
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(m))
    return C

def matvec(A, v):
    """Nhân ma trận với vector."""
    n = len(A)
    m = len(A[0])
    res = [0.0] * n
    for i in range(n):
        res[i] = sum(A[i][k] * v[k] for k in range(m))
    return res

def invert_matrix(A):
    """Nghịch đảo ma trận bằng phương pháp khử Gauss-Jordan (có tìm pivot)."""
    n = len(A)
    M = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(A)]
    
    for i in range(n):
        # Tìm pivot lớn nhất trong cột i để giảm sai số
        pivot_row = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[pivot_row][i]) < 1e-12:
            raise ValueError("Ma trận gần suy biến — có thể do đa cộng tuyến.")
        
        # Hoán vị dòng
        M[i], M[pivot_row] = M[pivot_row], M[i]
        
        # Khử chuẩn
        pivot = M[i][i]
        for j in range(i, 2 * n):
            M[i][j] /= pivot
            
        for k in range(n):
            if k != i:
                factor = M[k][i]
                for j in range(i, 2 * n):
                    M[k][j] -= factor * M[i][j]
                    
    return [row[n:] for row in M]

def add_intercept(X):
    """Thêm cột 1 (intercept) vào đầu ma trận X nếu chưa có."""
    X_2d = to_2d_list(X)
    if all(abs(row[0] - 1.0) < 1e-12 for row in X_2d):
        return X_2d
    return [[1.0] + row for row in X_2d]