import math
import matplotlib.pyplot as plt
import scipy.stats as stats
from Part1.ols_implementation import hat_matrix
from Part1.helper_function import add_intercept

def residual_plots(X, y, beta_hat):
    """
    Vẽ 4 biểu đồ phân tích phần dư
    X: list of lists (không chứa cột 1)
    y: list 1 chiều
    beta_hat: list 1 chiều (chứa beta_0 ở vị trí đầu)
    """
    if plt is None or stats is None:
        raise ImportError("matplotlib and scipy are required for residual_plots. Install them with: pip install matplotlib scipy")
    
    n = len(y)
    p = len(X[0]) # Số lượng biến đặc trưng
    
    # Thêm cột 1 vào X để nhân với beta_hat
    X_mat = add_intercept(X)
    
    # 1. Tính giá trị dự đoán (y_hat) = X_mat * beta_hat
    y_hat = []
    for i in range(n):
        val = sum(X_mat[i][j] * beta_hat[j] for j in range(p + 1))
        y_hat.append(val)
        
    # 2. Tính phần dư (residuals) = y - y_hat
    residuals = [y[i] - y_hat[i] for i in range(n)]
    
    # 3. Lấy ma trận Hat và trích xuất đường chéo (h_ii)
    H, _ = hat_matrix(X)
    h_ii = [H[i][i] for i in range(n)] 
    
    # 4. Tính phương sai nhiễu (sigma^2) và Standardized Residuals
    rss = sum(r**2 for r in residuals)
    sigma2 = rss / (n - p - 1)
    
    std_residuals = []
    for i in range(n):
        # Cộng thêm 1e-9 để tránh lỗi chia cho 0 nếu h_ii vô tình bằng 1
        denominator = math.sqrt(sigma2 * (1 - h_ii[i])) if (1 - h_ii[i]) > 0 else 1e-9
        std_residuals.append(residuals[i] / denominator)
        
    # 5. Tính Cook's Distance
    cooks_d = []
    for i in range(n):
        val = (std_residuals[i]**2 / (p + 1)) * (h_ii[i] / (1 - h_ii[i] + 1e-9))
        cooks_d.append(val)
        
    # 6. VẼ 4 BIỂU ĐỒ BẰNG MATPLOTLIB
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Residual Analysis', fontsize=16)

    # Biểu đồ 1: Residuals vs Fitted
    axs[0, 0].scatter(y_hat, residuals, alpha=0.6, edgecolors='w')
    axs[0, 0].axhline(0, color='red', linestyle='--', lw=2)
    axs[0, 0].set_title('Residuals vs Fitted')
    axs[0, 0].set_xlabel('Fitted values')
    axs[0, 0].set_ylabel('Residuals')

    # Biểu đồ 2: Normal Q-Q
    stats.probplot(std_residuals, dist="norm", plot=axs[0, 1])
    axs[0, 1].get_lines()[1].set_color('red')
    axs[0, 1].set_title('Normal Q-Q')
    axs[0, 1].set_ylabel('Standardized Residuals')

    # Biểu đồ 3: Scale-Location
    sqrt_std_residuals = [math.sqrt(abs(sr)) for sr in std_residuals]
    axs[1, 0].scatter(y_hat, sqrt_std_residuals, alpha=0.6, edgecolors='w')
    axs[1, 0].set_title('Scale-Location')
    axs[1, 0].set_xlabel('Fitted values')
    axs[1, 0].set_ylabel('$\sqrt{|Standardized \ Residuals|}$')

    # Biểu đồ 4: Cook's Distance
    axs[1, 1].stem(range(n), cooks_d, markerfmt=",", linefmt="b-", basefmt=" ")
    axs[1, 1].set_title("Cook's Distance")
    axs[1, 1].set_xlabel('Observation Index')
    axs[1, 1].set_ylabel("Cook's distance")

    plt.tight_layout()
    plt.savefig('residual_plots.png', dpi=300, bbox_inches='tight')
    plt.show()


