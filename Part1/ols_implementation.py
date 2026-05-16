from Part1.helper_function import (
    to_2d_list, to_1d_list, transpose_matrix, 
    matmul, matvec, invert_matrix, add_intercept
)
import math 
import numpy as np
import matplotlib.pyplot as plt

def model_metrics(y_true, y_pred, p):
    """
    Tính các chỉ số cốt lõi: RSS, TSS, R^2, Adjusted R^2 và F-test
    
    y_true: Danh sách giá trị thực tế (list)
    y_pred: Danh sách giá trị dự báo (list)
    p: Số lượng đặc trưng (không bao gồm intercept)
    
    returns: 
    dict: Chứa các chỉ số RSS, TSS, R2, Adjusted R2, F-statistic
    """
    n = len(y_true)
    if n <= p + 1:
        raise ValueError("Số lượng mẫu (n) phải lớn hơn số lượng đặc trưng (p) + 1")

    #tính trung bình y
    y_mean = sum(y_true) / n
    
    #RSS: Residual Sum of Squares
    rss = sum((yt - yp)**2 for yt, yp in zip(y_true, y_pred))
    
    #TSS: Total Sum of Squares
    tss = sum((yt - y_mean)**2 for yt in y_true)
    
    #R^2: Coefficient of Determination
    r2 = 1 - (rss / tss) if tss != 0 else 0
    
    #Adjusted R^2
    #công thức: 1 - (1 - R2) * (n - 1) / (n - p - 1)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    #F-statistic
    #công thức: ((TSS - RSS) / p) / (RSS / (n - p - 1))
    if rss == 0:
        f_stat = float('inf')
    else:
        f_stat = ((tss - rss) / p) / (rss / (n - p - 1))
    
    return {
        "RSS": rss,
        "TSS": tss,
        "R2": r2,
        "Adjusted R2": adj_r2,
        "F-statistic": f_stat
    }

def calculate_vif(X_df):
    """
    Tính hệ số phóng đại phương sai (VIF) cho từng đặc trưng.

    X_df: Pandas DataFrame chứa các đặc trưng
    
    returns:
    dict: VIF của từng đặc trưng
    """
    features = X_df.columns.tolist()
    vif_dict = {}
    
    for feature in features:
        #biến mục tiêu phụ là feature hiện tại
        y = X_df[feature].tolist()
        
        #các biến độc lập phụ là các feature còn lại
        other_features = [f for f in features if f != feature]
        X_others = X_df[other_features]
        
        try:
            #ols_fit trả về (beta_hat, sigma2_hat)
            beta_hat, _ = ols_fit(X_others, y)
            
            #dự báo y_hat cho hồi quy phụ
            #thêm intercept vào X_others để tính y_hat
            X_others_with_const = add_intercept(X_others.values.tolist())
            y_pred_aux = matvec(X_others_with_const, beta_hat)
            
            #tính R^2 cho hồi quy phụ
            #số lượng đặc trưng p_aux = len(other_features)
            metrics = model_metrics(y, y_pred_aux, len(other_features))
            r2_aux = metrics["R2"]
            
            #VIF = 1 / (1 - R^2)
            if r2_aux >= 1.0:
                vif = float('inf')
            else:
                vif = 1 / (1 - r2_aux)
                
        except Exception:
            #Trường hợp ma trận suy biến
            vif = float('inf')
            
        vif_dict[feature] = vif
        
    return vif_dict


def predict(X, beta_hat):
    """
    Dự đoán ŷ = Xβ̂ cho dữ liệu mới.
    """
    X_mat = add_intercept(X)
    b_vec = to_1d_list(beta_hat)
    return np.array(matvec(X_mat, b_vec))


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


def std_error(sigma2, X): 
    XT_X_inverse = invert_matrix(matmul(transpose_matrix(X), X))
    
    se_list = []

    for j in range(len(XT_X_inverse[0])): 
        SE_j = math.sqrt(sigma2*XT_X_inverse[j][j]) 
        se_list.append(SE_j) 

    return tuple(se_list)
import math

def get_t_critical(df):
    # Bảng tra cứu t-critical cho alpha = 0.05 (kiểm định 2 phía)
    # Các giá trị này được lấy từ bảng phân phối Student chuẩn
    t_table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042,
        40: 2.021, 50: 2.009, 60: 2.000, 80: 1.990, 100: 1.984
    }
    
    if df in t_table:
        return t_table[df]

    if df > 100:
        return 1.960  # Xấp xỉ phân phối chuẩn Z khi n lớn 
    
    # Tìm giá trị gần nhất trong bảng nếu không khớp hoàn toàn
    keys = sorted(t_table.keys())
    for i in range(len(keys) - 1):
        if keys[i] < df < keys[i+1]:
            # Trả về giá trị của mốc nhỏ hơn để đảm bảo an toàn (conservative)
            return t_table[keys[i]]
            
    return 2.0  # Giá trị mặc định 

def coef_inference(X, y, beta_hat, sigma2):
    #Tính Standard Errors
    Standard_Errors = std_error(sigma2, X) 

    # Tính t-statistics
    t_list = [] 
    for i in range(len(beta_hat)): 
        t_i = beta_hat[i][0] / Standard_Errors[i] 
        t_list.append(t_i)

    n = len(X)
    p_plus_1 = len(X[0])
    df = n - p_plus_1

    t_critical = get_t_critical(df)

    conf_intervals = []
    for i in range(len(beta_hat)):
        lower = beta_hat[i][0] - t_critical * Standard_Errors[i]
        upper = beta_hat[i][0] + t_critical * Standard_Errors[i]
        conf_intervals.append((lower, upper))

    significance = []
    for t_val in t_list:
        is_significant = abs(t_val) > t_critical
        significance.append("Có ý nghĩa" if is_significant else "Không có ý nghĩa")

    return {
        "df": df,
        "t_critical": t_critical,
        "Standard Errors": Standard_Errors,
        "t_stats": t_list,
        "Confidence Intervals": conf_intervals,
        "Significance_at_5pct": significance
    }
