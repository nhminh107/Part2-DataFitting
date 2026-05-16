import random

def kfold_cv(X, y, k, model_func, seed=42, **model_params):
    """
    Cài đặt K-Fold Cross-Validation.
    
    Input:
    X : list of lists
    y : list
    k : int
    model_func : callable (Hàm fit mô hình, trả về list beta)
    seed : int
    
    Output:
    mean_mse : float
    """
    n = len(y)
    indices = list(range(n))
    
    # Xáo trộn chỉ số
    if seed is not None:
        random.seed(seed)
        random.shuffle(indices)
        
    # Chia folds
    fold_size = n // k
    folds = []
    for i in range(k):
        start = i * fold_size
        # Fold cuối cùng lấy hết phần còn lại
        end = (i + 1) * fold_size if i != k - 1 else n
        folds.append(indices[start:end])
        
    mses = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = []
        for j in range(k):
            if i != j:
                train_idx.extend(folds[j])
                
        # Tạo tập train và val
        X_train = [X[idx] for idx in train_idx]
        y_train = [y[idx] for idx in train_idx]
        X_val = [X[idx] for idx in val_idx]
        y_val = [y[idx] for idx in val_idx]
        
        # Huấn luyện
        beta_hat = model_func(X_train, y_train, **model_params)
        
        # Dự đoán và tính MSE thủ công
        fold_se = 0
        for row_idx, row in enumerate(X_val):
            # dot product: row * beta_hat
            y_pred = sum(row[m] * beta_hat[m] for m in range(len(row)))
            fold_se += (y_val[row_idx] - y_pred) ** 2
            
        mses.append(fold_se / len(val_idx))
        
    return sum(mses) / k, mses
