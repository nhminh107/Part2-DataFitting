def add(A, B):
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def subtract(A, B):
    return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def multiply(A, B):
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    C = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C

def divide(A, B):
    return [[A[i][j] / B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def transpose(A):
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def inverse(A):
    n = len(A)
    AM = [row[:] + [1 if i == j else 0 for j in range(n)] for i, row in enumerate(A)]
    
    for fd in range(n):
        fdScaler = 1.0 / AM[fd][fd]
        for j in range(n * 2):
            AM[fd][j] *= fdScaler
        for i in range(n):
            if i != fd:
                crScaler = AM[i][fd]
                for j in range(n * 2):
                    AM[i][j] -= crScaler * AM[fd][j]
                    
    return [row[n:] for row in AM]