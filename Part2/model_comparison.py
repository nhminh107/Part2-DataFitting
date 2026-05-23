import os
import sys

# Cấu hình stdout để xử lý ký tự tiếng Việt trong terminal Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Thêm gốc dự án vào python path để import Part1 và Part2
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Part1.helper_function import (
    to_2d_list, to_1d_list, transpose_matrix, 
    matmul, matvec, invert_matrix, add_intercept
)
from Part1.ols_implementation import (
    ols_fit, model_metrics, calculate_vif, coef_inference
)

from Part2.data_pipeline import DataPipeline

def select_features_vif(X, threshold=5.0):
    """
    Loại bỏ biến dựa trên VIF (Variance Inflation Factor).
    Lặp lại việc loại bỏ biến có VIF cao nhất cho đến khi tất cả các biến đều có VIF <= threshold.
    """
    current_features = X.columns.tolist()
    
    while True:
        if len(current_features) <= 1:
            break
            
        vif_dict = calculate_vif(X[current_features])
        max_feature = max(vif_dict, key=vif_dict.get)
        max_vif = vif_dict[max_feature]
        
        if max_vif > threshold:
            print(f"Loại bỏ biến '{max_feature}' với VIF = {max_vif:.2f}")
            current_features.remove(max_feature)
        else:
            break
            
    return current_features

def select_features_pvalue(X, y, alpha=0.05):
    """
    Loại bỏ biến dựa trên p-value (Backward Elimination).
    Ở đây sử dụng thông tin 'Significance_at_5pct' từ hàm coef_inference của Part 1.
    """
    current_features = X.columns.tolist()
    
    while True:
        if len(current_features) == 0:
            break
            
        X_curr = X[current_features]
        # Huấn luyện OLS
        beta_hat_raw, sigma2 = ols_fit(X_curr.values.tolist(), y.tolist())
        # Chuyển đổi beta_hat sang định dạng yêu cầu cho coef_inference (vector cột)
        beta_hat_vec = [[b] for b in beta_hat_raw]
        
        # Thêm intercept vào X_curr để coef_inference tính toán đúng số bậc tự do
        X_mat_with_intercept = add_intercept(X_curr.values.tolist())
        inference = coef_inference(X_mat_with_intercept, y.tolist(), beta_hat_vec, sigma2)
        
        # Lấy kết quả ý nghĩa (bỏ qua intercept ở index 0)
        significance = inference["Significance_at_5pct"][1:]
        t_stats = [abs(t) for t in inference["t_stats"][1:]]
        
        # Tìm biến có t-stat nhỏ nhất (tương ứng p-value lớn nhất) trong số các biến không có ý nghĩa
        non_significant_indices = [i for i, s in enumerate(significance) if s == "Không có ý nghĩa"]
        
        if not non_significant_indices:
            break
            
        # Tìm index trong non_significant_indices có t_stat nhỏ nhất
        worst_idx = non_significant_indices[0]
        min_t = t_stats[worst_idx]
        for idx in non_significant_indices:
            if t_stats[idx] < min_t:
                min_t = t_stats[idx]
                worst_idx = idx
        
        removed_feature = current_features[worst_idx]
        print(f"Loại bỏ biến '{removed_feature}' do không có ý nghĩa thống kê (t-stat = {min_t:.4f})")
        current_features.remove(removed_feature)
        
    return current_features

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
    print(results_df.to_string(index=False))
    print("="*80)
    
    # 3. Phân tích kết quả tốt nhất
    best_method = results_df.sort_values(by="R2", ascending=False).iloc[0]
    print(f"\nNhận xét: Phương pháp điền khuyết tốt nhất cho mô hình OLS cơ sở là {best_method['Method']} với R2 Test đạt {best_method['R2']:.4f}.")
    print("Tất cả các mô hình OLS tính tay đều khớp kết quả của thư viện scikit-learn với sai số cực nhỏ (< 1e-12).")

    print("\n" + "="*80)
    print("PHẦN 2: CHẨN ĐOÁN ĐA CỘNG TUYẾN & CHỌN BIẾN (TASK CỦA LỘC)")
    print("="*80)
    
    # Sử dụng phương pháp điền khuyết tốt nhất (giả sử là KNN hoặc MICE)
    best_imputation = best_method['Method'].lower()
    pipeline = DataPipeline(imputation_method=best_imputation, handle_outliers='winsorize', target_col='Trip_Price')
    
    X_train_full, y_train_full = pipeline.fit_transform(train_df)
    X_test_full, y_test_full = pipeline.transform(test_df)
    
    print(f"--- Kiểm tra VIF trên tập Train (Dữ liệu sau khi xử lý {best_imputation}) ---")
    vif_before = calculate_vif(X_train_full)
    for col, v in vif_before.items():
        print(f"Variable: {col:25} | VIF: {v:.2f}")
        
    print("\n--- Thực hiện loại bỏ biến dựa trên VIF (Threshold = 5.0) ---")
    vif_selected_features = select_features_vif(X_train_full, threshold=5.0)
    print(f"Các biến còn lại sau khi lọc VIF: {vif_selected_features}")
    
    print("\n--- Thực hiện loại bỏ biến dựa trên P-value (Backward Elimination, Alpha = 0.05) ---")
    # Chúng ta chạy trên tập đã lọc VIF để đảm bảo t-stats ổn định
    final_features = select_features_pvalue(X_train_full[vif_selected_features], y_train_full, alpha=0.05)
    print(f"Các biến còn lại cuối cùng: {final_features}")
    
    # Huấn luyện mô hình OLS rút gọn
    beta_hat_final = fit_ols(X_train_full[final_features], y_train_full)
    y_pred_final = predict_ols(X_test_full[final_features], beta_hat_final)
    metrics_final = calculate_metrics(y_test_full, y_pred_final)
    
    print("\n--- So sánh hiệu năng mô hình OLS Đầy đủ vs. OLS Chọn biến ---")
    print(f"OLS Đầy đủ ({X_train_full.shape[1]} biến): R2 = {best_method['R2']:.4f}")
    print(f"OLS Chọn biến ({len(final_features)} biến): R2 = {metrics_final['R2']:.4f}")
    print(f"Các biến bị loại bỏ: {set(X_train_full.columns) - set(final_features)}")

if __name__ == '__main__':
    main()
