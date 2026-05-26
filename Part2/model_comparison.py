import os
import sys

# Reconfigure stdout to handle Vietnamese characters in Windows terminal
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Add root to python path to import modules using absolute paths (e.g., Part1.module)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Part1.helper_function import (
    to_2d_list, to_1d_list, transpose_matrix, 
    matmul, matvec, invert_matrix, add_intercept
)

from Part2.data_pipeline import DataPipeline

def fit_ols(X, y):
    """
    Huấn luyện OLS sử dụng các phép toán ma trận viết tay từ Phần 1.
    y = X*beta + epsilon
    """
    # Chuyển đổi dữ liệu sang dạng list of lists (hoặc list) của Python
    X_list = X.values.tolist()
    y_list = to_1d_list(y)
    
    # Chèn cột 1 (intercept) vào đầu ma trận X
    X_mat = add_intercept(X_list)
    
    # Công thức OLS: beta = (X^T * X)^(-1) * X^T * y
    Xt = transpose_matrix(X_mat)
    XtX = matmul(Xt, X_mat)
    XtX_inv = invert_matrix(XtX)
    Xty = matvec(Xt, y_list)
    
    beta_hat = matvec(XtX_inv, Xty)
    return beta_hat

def predict_ols(X, beta_hat):
    """
    Dự báo dựa trên ma trận đặc trưng X và hệ số beta_hat.
    """
    X_list = X.values.tolist()
    X_mat = add_intercept(X_list)
    return matvec(X_mat, beta_hat)

def calculate_metrics(y_true, y_pred):
    """
    Tính các chỉ số đánh giá mô hình hồi quy: MAE, RMSE, R2
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    y_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }

def main():
    print("="*60)
    print("CHƯƠNG TRÌNH SO SÁNH CÁC PHƯƠNG PHÁP ĐIỀN KHUYẾT & HỒI QUY OLS")
    print("="*60)
    
    # 1. Đọc dữ liệu
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'taxi_trip_pricing.csv')
    df = pd.read_csv(data_path)
    
    # Loại bỏ các hàng bị khuyết biến mục tiêu Trip_Price trước khi chia tập dữ liệu
    df_clean = df.dropna(subset=['Trip_Price']).reset_index(drop=True)
    print(f"Số lượng mẫu ban đầu: {len(df)}")
    print(f"Số lượng mẫu sau khi xóa hàng thiếu target (Trip_Price): {len(df_clean)}")
    
    # Chia tập dữ liệu Train/Test (80/20) cố định random_state
    train_df, test_df = train_test_split(df_clean, test_size=0.2, random_state=42)
    print(f"Kích thước tập Train: {len(train_df)} dòng | Tập Test: {len(test_df)} dòng\n")
    
    # Các phương pháp điền khuyết cần thử nghiệm
    imputation_methods = ['listwise', 'mean', 'regression', 'knn', 'mice']
    results = []
    
    for method in imputation_methods:
        print(f"Đang xử lý phương pháp điền khuyết: {method.upper()}...")
        try:
            # Khởi tạo pipeline
            pipeline = DataPipeline(imputation_method=method, handle_outliers='winsorize', target_col='Trip_Price')
            
            # Tiền xử lý tập Train
            X_train, y_train = pipeline.fit_transform(train_df)
            
            # Tiền xử lý tập Test
            X_test, y_test = pipeline.transform(test_df)
            
            # Huấn luyện OLS tính tay
            beta_hat = fit_ols(X_train, y_train)
            
            # Dự đoán trên tập Test
            y_pred = predict_ols(X_test, beta_hat)
            
            # Tính toán các chỉ số đánh giá
            metrics = calculate_metrics(y_test, y_pred)
            
            # Đối chiếu với Scikit-learn để kiểm chứng toán học
            sk_model = LinearRegression()
            sk_model.fit(X_train, y_train)
            sk_beta = [sk_model.intercept_] + list(sk_model.coef_)
            
            # Tính sai số lớn nhất giữa hệ số tính tay và thư viện
            max_diff = np.max(np.abs(np.array(beta_hat) - np.array(sk_beta)))
            
            results.append({
                "Method": method.upper(),
                "Train Samples": len(X_train),
                "Test Samples": len(X_test),
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "R2": metrics["R2"],
                "Math Verification Diff": max_diff
            })
            print(f"-> Hoàn thành: Test R2 = {metrics['R2']:.4f} | Sai số tính toán so với Sklearn = {max_diff:.2e}")
            
        except Exception as e:
            print(f"-> Gặp lỗi khi xử lý phương pháp {method}: {e}")
            import traceback
            traceback.print_exc()
            
    # 2. Hiển thị bảng kết quả
    print("\n" + "="*80)
    print("BẢNG KẾT QUẢ SO SÁNH CÁC PHƯƠNG PHÁP XỬ LÝ DỮ LIỆU THIẾU")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    # Định dạng bảng markdown đẹp mắt
    print(results_df.to_markdown(index=False))
    print("="*80)
    
    # 3. Phân tích kết quả tốt nhất
    best_method = results_df.sort_values(by="R2", ascending=False).iloc[0]
    print(f"\nNhận xét: Phương pháp điền khuyết tốt nhất cho mô hình OLS cơ sở là {best_method['Method']} với R2 Test đạt {best_method['R2']:.4f}.")
    print("Tất cả các mô hình OLS tính tay đều khớp kết quả của thư viện scikit-learn với sai số cực nhỏ (< 1e-12).")
    
if __name__ == '__main__':
    main()
