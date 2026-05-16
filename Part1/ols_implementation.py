from Part1.helper_function import transpose, inverse, multiply
import math 

# Cái này return mẫu tạm để làm các hàm khác thôi
def ols_fit(X, y):
    beta_hat = [[1.5], [2.3], [-0.8]]
    sigma2 = 0.45
    return beta_hat, sigma2


def std_error(sigma2, X): 
    XT_X_inverse = inverse(multiply(transpose(X), X))
    
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
