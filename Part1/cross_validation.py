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

if __name__ == "__main__":
    import numpy as np
    from ridge_lasso import ridge_fit

    rng = np.random.default_rng(3)

    def make_xy(n, p, noise, collinear=False):
        X = rng.normal(size=(n, p))
        if collinear and p >= 2:
            X[:, 1] = X[:, 0] + 0.1 * rng.normal(size=n)
        beta = rng.normal(size=p + 1)
        y = beta[0] + X @ beta[1:] + noise * rng.normal(size=n)
        return X, y

    cases = [
        (120, 3, 0.2, False),
        (60, 1, 0.2, False),
        (120, 8, 0.2, False),
        (120, 4, 5.0, False),
        (150, 3, 0.2, True),
    ]

    for idx, (n, p, noise, collinear) in enumerate(cases, 1):
        print(f"kfold_cv case {idx}: n={n} p={p} noise={noise} collinear={collinear}")
        X, y = make_xy(n, p, noise, collinear)
        X_bias = np.hstack([np.ones((n, 1)), X])
        for k in (3, 5, 10):
            print(f"  k={k}")
            mean_mse_my, _ = kfold_cv(X_bias.tolist(), y.tolist(), k, ridge_fit, seed=42, lam=0.1)
            indices = list(range(n))
            rnd = random.Random(42)
            rnd.shuffle(indices)
            fold_size = n // k
            folds = []
            for i in range(k):
                start = i * fold_size
                end = (i + 1) * fold_size if i != k - 1 else n
                folds.append(indices[start:end])
            mses_np = []
            for i in range(k):
                val_idx = folds[i]
                train_idx = []
                for j in range(k):
                    if i != j:
                        train_idx.extend(folds[j])
                X_train = X_bias[train_idx]
                y_train = y[train_idx]
                X_val = X_bias[val_idx]
                y_val = y[val_idx]
                lamI = np.diag([0.0] + [0.1] * p)
                beta_np = np.linalg.inv(X_train.T @ X_train + lamI) @ (X_train.T @ y_train)
                y_pred = X_val @ beta_np
                mses_np.append(float(np.mean((y_val - y_pred) ** 2)))
            mean_mse_np = float(np.mean(mses_np))
            np.testing.assert_allclose(mean_mse_my, mean_mse_np, rtol=1e-6, atol=1e-6)
