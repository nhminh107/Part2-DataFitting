from helper_function import transpose, multiply, inverse

def model_metrics(y_true, y_pred, p):
    """
    Tính các chỉ số cốt lõi: RSS, TSS, R^2, Adjusted R^2 và F-test

    y_true: Danh sách giá trị thực tế (list)
    y_pred: Danh sách giá trị dự báo (list)
    p: Số lượng đặc trưng 
    
    Return: dict: Chứa các chỉ số RSS, TSS, R2, Adjusted R2, F-statistic
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
    Tính hệ số phóng đại phương sai (VIF) cho từng đặc trưng
    
    X_df: Pandas DataFrame chứa các đặc trưng
    
    Return: dict: VIF của từng đặc trưng
    """
    features = X_df.columns.tolist()
    vif_dict = {}
    
    for feature in features:
        #biến mục tiêu phụ là feature hiện tại
        y = [[val] for val in X_df[feature].tolist()]
        
        #các biến độc lập phụ là các feature còn lại
        other_features = [f for f in features if f != feature]
        X_others_raw = X_df[other_features].values.tolist()
        
        #thêm cột intercept
        X_others = [[1.0] + row for row in X_others_raw]
        
        try:
            #giải OLS: beta = (X^T X)^-1 X^T y
            XT = transpose(X_others)
            XTX = multiply(XT, X_others)
            XTX_inv = inverse(XTX)
            XTy = multiply(XT, y)
            beta = multiply(XTX_inv, XTy)
            
            #tính y_pred cho hồi quy phụ
            #y_pred = X * beta
            y_pred_aux = [multiply([row], beta)[0][0] for row in X_others]
            y_true_aux = [val[0] for val in y]
            
            #tính R^2 cho hồi quy phụ
            #số lượng đặc trưng p_aux = len(other_features)
            metrics = model_metrics(y_true_aux, y_pred_aux, len(other_features))
            r2_aux = metrics["R2"]
            
            #VIF = 1 / (1 - R^2)
            if r2_aux >= 1.0:
                vif = float('inf')
            else:
                vif = 1 / (1 - r2_aux)
                
        except Exception:
            #trường hợp ma trận suy biến
            vif = float('inf')
            
        vif_dict[feature] = vif
        
    return vif_dict
