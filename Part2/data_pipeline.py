import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os
import seaborn as sns
path = os.getcwd() 
data_path = os.path.join(path, "Part2", "data", "taxi_trip_pricing.csv")
pd.set_option('display.max_columns', None)
class DataPipeline: 
    def __init__(self):
        self.df = pd.read_csv(data_path)
    def EDA(self):
        print("########## THỐNG KÊ MÔ TẢ ############")
        summary = self.df.describe()
        print(summary)
        print("########## PHÂN PHỐI TỪNG BIẾN ##########")
        numeric_col = self.df.select_dtypes(include=np.number).columns

        for col in numeric_col: 
            fig, axes = plt.subplots(1, 2, figsize=(12,5))
            sns.histplot(self.df[col], kde = True, ax = axes[0])
            sns.boxplot(x = self.df[col], ax=axes[1]) 
            fig.suptitle("Histogram & boxplot of " + str(col), color='blue', ha='center')
            plt_name = "EDA_"+str(col)+".png"
            #plt.savefig(plt_name, bbox_inches='tight')
            plt.show()

        print("########## MA TRẬN TƯƠNG QUAN ###########")
        corr_matrix = self.df.corr(numeric_only=True)

        plt.figure(figsize=(12,8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.suptitle("Correlation Matrix (Numerical feature only)")
        plt.savefig("Correlation_Matrix.png", bbox_inches='tight')
        plt.show()

        print("########## CHECK DUPLICATES ###########")
        print(f"Số records trùng lặp: {self.df.duplicated().sum()}")

        print("########## MISSING VALUES #############")
# 1. Tính toán dữ liệu thiếu
        missing_count = self.df.isnull().sum()
        missing_percentage = (missing_count / len(self.df)) * 100

        # 2. Tạo một DataFrame để quản lý dữ liệu thiếu (chỉ lấy các cột có missing > 0)
        missing_df = pd.DataFrame({
            'Column': missing_count.index,
            'Missing Count': missing_count.values,
            'Percentage (%)': missing_percentage.values
        }).sort_values(by='Missing Count', ascending=False)

        # Chỉ vẽ các cột thực sự có dữ liệu thiếu để biểu đồ không bị rối
        missing_df = missing_df[missing_df['Missing Count'] > 0]

        # 3. Vẽ biểu đồ
        if not missing_df.empty:
            fig, ax1 = plt.subplots(figsize=(12, 6))

            # Vẽ cột cho số lượng (Bar plot)
            sns.barplot(x='Column', y='Missing Count', data=missing_df, ax=ax1, palette='viridis')
            ax1.set_ylabel('Số lượng bản ghi thiếu', color='b')
            plt.xticks(rotation=45)

            # Tạo trục thứ hai để vẽ phần trăm (nếu muốn kết hợp hoặc chỉ cần dùng 1 trục)
            # Ở đây tôi sẽ in text phần trăm ngay trên đầu cột để bạn dễ quan sát
            for i, p in enumerate(ax1.patches):
                percentage = missing_df.iloc[i]['Percentage (%)']
                ax1.annotate(f'{percentage:.2f}%', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha='center', va='center', 
                            xytext=(0, 9), 
                            textcoords='offset points')

            rows_with_nan = self.df.isnull().any(axis=1).sum()
            ratio_nan_rows = (rows_with_nan / len(self.df)) * 100
            plt.suptitle("Thống kê dữ liệu thiếu theo từng cột", color='blue', ha='center')
            plt.title(f"Số dòng chứa NaN: {rows_with_nan}; Tỉ lệ: {ratio_nan_rows:.2f}%")
            plt.savefig("Count_nan")
            plt.show()
        else:
            print("Không có dữ liệu thiếu trong DataFrame.")     


        print("######### PHÁT HIỆN OUTLIERS - Phương pháp IQr ##############")
        q_25 = summary.loc['25%']
        q_50 = summary.loc['50%']
        q_75 = summary.loc['75%']

        for col in numeric_col: 
            q3, q1 = q_75[col] , q_25[col] 
            iqr = q3 - q1
            lower_bound = q1 - 1.5*iqr
            upper_bound = q3 + 1.5*iqr 

            mask = (self.df[col] > upper_bound) | (self.df[col] < lower_bound) 
            print(f"Số outlier ở cột {col}:", mask.sum())

dt = DataPipeline() 
dt.EDA()