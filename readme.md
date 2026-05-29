# Đồ án 2: Data Fitting và Phương pháp OLS

## Giới thiệu dự án
Đây là Đồ án 2 môn học **Toán Ứng dụng và Thống kê**. Dự án tập trung vào việc nghiên cứu nền tảng toán học của phương pháp Bình phương Cực tiểu (Ordinary Least Squares - OLS) và ứng dụng nó để giải quyết bài toán hồi quy trên dữ liệu thực tế.

Dự án được chia thành hai phần chính:
1. **Phần 1: Lý thuyết và Minh họa** - Cài đặt các thuật toán OLS và các kỹ thuật liên quan.
2. **Phần 2: Ứng dụng thực tế** - Xây dựng pipeline xử lý dữ liệu và huấn luyện mô hình dự đoán giá cước Taxi (Taxi Price Prediction) từ Kaggle.

---

## Thông tin nhóm thực hiện
**Nhóm 2:**
- **Ngô Hoàng Minh** - 24120381
- **Mai Thúc Hải Đăng** - 24120276
- **Nguyễn Thành Dự** - 24120288
- **Đỗ Ngọc Hải** - 24120300
- **Nguyễn Xuân Lộc** - 24120369

**Giảng viên lí thuyết:** Dương Việt Hằng
**Giảng viên hướng dẫn:** Võ Nam Thục Đoan, Lê Nhựt Nam.
**Trợ giảng:** Đỗ Đức Hào.

---

## Các tính năng chính

### Phần 1: Cài đặt OLS
- **`ols_fit`**: Ước lượng hệ số $\hat{\beta}$ và phương sai nhiễu $\hat{\sigma}^2$.
- **`hat_matrix`**: Tính ma trận hình mũ (Hat matrix) và kiểm tra tính lũy đẳng.
- **`model_metrics`**: Tính toán các chỉ số đánh giá $RSS, TSS, R^2, \bar{R}^2$ và kiểm định F.
- **`coef_inference`**: Tính sai số chuẩn, thống kê t, p-value và khoảng tin cậy 95%.
- **`calculate_vif`**: Đo lường đa cộng tuyến.
- **`ridge_fit` & `lasso_fit`**: Cài đặt hồi quy Ridge và Lasso (sử dụng Coordinate Descent).
- **`residual_plots`**: Vẽ 4 biểu đồ chẩn đoán phần dư (Residuals vs Fitted, Normal Q-Q, Scale-Location, Cook's Distance).
- **`kfold_cv`**: Kỹ thuật kiểm định chéo để lựa chọn siêu tham số $\lambda$.
- **Mô phỏng Monte Carlo**: Chứng minh định lý Gauss-Markov (tính không chệch và BLUE).

### Phần 2: Ứng dụng dự đoán giá Taxi
- **Bộ dữ liệu**: Taxi Price Prediction (hơn 1.000 dòng, có dữ liệu thiếu và ngoại lệ).
- **Data Pipeline**:
    - Xử lý giá trị thiếu (KNN Imputation).
    - Xử lý ngoại lệ (Winsorization bằng phương pháp IQR).
    - Mã hóa biến phân loại (One-hot, Ordinal Encoding).
    - Kỹ thuật đặc trưng (Log-transform cho khoảng cách, Đa thức bậc 2 cho thời gian).
    - Chuẩn hóa dữ liệu (Z-score).
- **Lựa chọn đặc trưng**: Sử dụng Backward Elimination dựa trên p-value và loại bỏ đa cộng tuyến bằng VIF.
- **Mô hình nâng cao**: Thử nghiệm thêm Kernel Ridge Regression (KRR) và Bayesian Linear Regression (BLR).

---

## Cấu trúc thư mục
```text
.
├── Part1/                  # Phần 1: Lý thuyết và minh họa
│   ├── ols_implementation.py # Cài đặt OLS từ đầu
│   ├── ridge_lasso.py        # Cài đặt Ridge và Lasso
│   ├── residual_analysis.py  # Phân tích phần dư
│   ├── cross_validation.py   # K-Fold Cross Validation
│   ├── helper_function.py    # Các hàm bổ trợ (Ma trận,...)
│   ├── part1_demo.ipynb      # Notebook demo lý thuyết
│   └── test/                 # Unit tests cho các hàm phần 1
├── Part2/                  # Phần 2: Ứng dụng thực tế
│   ├── data/                 # Chứa file dữ liệu taxi_trip_pricing.csv
│   ├── plot/                 # Các biểu đồ EDA và kết quả
│   ├── data_pipeline.py      # Pipeline xử lý dữ liệu
│   ├── model_comparison.py   # Huấn luyện và so sánh các mô hình
│   ├── advanced_methods.py   # Kernel Ridge và Bayesian LR
│   ├── part2_notebook.ipynb  # Notebook phân tích kết quả
│   └── test/                 # Unit tests cho pipeline và mô hình
├── report/                 # Báo cáo đồ án
│   ├── report.pdf            # File báo cáo PDF hoàn chỉnh
│   └── report.tex/           # Mã nguồn LaTeX
├── requirements.txt        # Các thư viện cần thiết
└── README.md               # Hướng dẫn sử dụng
```

---

## Hướng dẫn cài đặt và sử dụng

### Cài đặt
Yêu cầu Python 3.10 trở lên. Cài đặt các thư viện cần thiết bằng lệnh:
```bash
pip install -r requirements.txt
```

### Chạy Demo
- Để xem minh họa phần lý thuyết, mở file `Part1/part1_demo.ipynb`.
- Để xem quy trình xử lý dữ liệu thực tế và kết quả mô hình, mở file `Part2/part2_notebook.ipynb`.

### Chạy Tests
Nhóm đã xây dựng hệ thống unit test để đảm bảo tính chính xác của các hàm. Chạy test bằng lệnh:
```bash
python -m pytest Part1/test/
python -m pytest Part2/test/
```

---

## Kết quả đạt được
- **Độ chính xác**: Mô hình OLS sau khi chọn lọc biến đạt $R^2 \approx 81.53\%$ trên tập kiểm thử (test set) với chỉ 4 biến cốt lõi, đảm bảo tính đơn giản và khả năng giải thích cao.
- **Hệ thống**: Xây dựng thành công class `DataPipeline` linh hoạt, hỗ trợ tốt việc `fit` trên tập train và `transform` trên tập test, tránh rò rỉ dữ liệu (data leakage).
- **Kết luận**: Dự án khẳng định vai trò quan trọng của tiền xử lý dữ liệu và Feature Engineering trong việc cải thiện hiệu suất mô hình hồi quy.

---

## Tài liệu tham khảo
1. Gilbert Strang, *Introduction to Linear Algebra*, 6th ed., 2023.
2. Gareth James et al., *An Introduction to Statistical Learning*, 2nd ed., 2021.
3. Christopher M. Bishop, *Pattern Recognition and Machine Learning*, 2006.
4. Scikit-learn developers, *Scikit-learn: Machine Learning in Python*
5. The pandas development team, *pandas-dev/pandas: Pandas. Zenodo*, 2020.
