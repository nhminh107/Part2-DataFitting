import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Part1.residual_analysis import residual_plots
from Part1.ols_implementation import ols_fit
def run_test():
    print("--- BẮT ĐẦU TEST HÀM RESIDUAL PLOTS ---")
    
    # 1. Tạo dữ liệu giả lập
    random.seed(42) # Cố định seed để kết quả ra y hệt nhau mỗi lần chạy 
    n = 100
    p = 2
    X = []
    y = []

    # Giả sử beta thực tế là: beta_0 = 3.5, beta_1 = 1.2, beta_2 = -2.0
    true_beta = [3.5, 1.2, -2.0]

    for _ in range(n):
        # Tạo ngẫu nhiên 2 đặc trưng x1, x2
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 10)
        X.append([x1, x2])
        
        # Thêm nhiễu ngẫu nhiên phân phối chuẩn với mean=0 và std=1.5
        epsilon = random.gauss(0, 1.5) 
        
        # Tính y = beta_0 + beta_1*x1 + beta_2*x2 + epsilon [cite: 59]
        y_val = true_beta[0] + true_beta[1]*x1 + true_beta[2]*x2 + epsilon
        y.append(y_val)

    beta_hat, sigma2 = ols_fit(X, y)
    
    print(f"Tham số beta thực tế: {true_beta}")
    print(f"Tham số beta ước lượng (beta_hat): {[round(b, 4) for b in beta_hat]}")
    print(f"Phương sai nhiễu ước lượng (sigma2): {round(sigma2, 4)}")
    print("Đang vẽ biểu đồ...")

    # 3. Gọi hàm vẽ
    residual_plots(X, y, beta_hat)
    print("Hoàn thành! Hãy kiểm tra file residual_plots.png.")

if __name__ == "__main__":
    run_test()