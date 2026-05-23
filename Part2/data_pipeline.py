import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import os
import sys

# Reconfigure stdout to handle Vietnamese characters in Windows terminal
try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

import seaborn as sns
from sklearn.linear_model import Ridge
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Set default settings
path = os.getcwd() 
data_path = os.path.join(path, "Part2", "data", "taxi_trip_pricing.csv")
pd.set_option('display.max_columns', None)

class DataPipeline: 
    def __init__(self, data_path=None, imputation_method='knn', handle_outliers='winsorize', target_col='Trip_Price'):
        self.imputation_method = imputation_method
        self.handle_outliers = handle_outliers
        self.target_col = target_col
        
        # Maintain backward compatibility with the original constructor
        if data_path is None:
            # Look in the current working directory or dynamic path
            cw = os.getcwd()
            possible_path = os.path.join(cw, "Part2", "data", "taxi_trip_pricing.csv")
            if os.path.exists(possible_path):
                self.df = pd.read_csv(possible_path)
            elif os.path.exists(os.path.join(cw, "data", "taxi_trip_pricing.csv")):
                self.df = pd.read_csv(os.path.join(cw, "data", "taxi_trip_pricing.csv"))
            else:
                self.df = None
        else:
            if os.path.exists(data_path):
                self.df = pd.read_csv(data_path)
            else:
                self.df = None

        # Containers for training statistics
        self.impute_values = {}
        self.scale_means = {}
        self.scale_stds = {}
        self.outlier_bounds = {}
        self.categorical_cols = []
        self.numeric_cols = []
        self.dummy_columns_map = {}
        
        # Imputers
        self.knn_imputer = None
        self.mice_imputer = None
        self.regression_models = {}

    def EDA(self):
        if self.df is None:
            print("No data available to perform EDA.")
            return
            
        print("########## THỐNG KÊ MÔ TẢ ############")
        summary = self.df.describe()
        print(summary)
        
        print("########## PHÂN PHỐI TỪNG BIẾN ##########")
        numeric_col = self.df.select_dtypes(include=np.number).columns
        
        # Ensure the plot directory exists
        plot_dir = os.path.join(os.getcwd(), "Part2", "plot")
        if not os.path.exists(plot_dir):
            plot_dir = os.path.join(os.getcwd(), "plot")
            os.makedirs(plot_dir, exist_ok=True)

        for col in numeric_col: 
            fig, axes = plt.subplots(1, 2, figsize=(12,5))
            sns.histplot(self.df[col], kde=True, ax=axes[0])
            sns.boxplot(x=self.df[col], ax=axes[1]) 
            fig.suptitle("Histogram & boxplot of " + str(col), color='blue', ha='center')
            plt_name = os.path.join(plot_dir, "EDA_" + str(col) + ".png")
            plt.savefig(plt_name, bbox_inches='tight')
            plt.show()

        print("########## MA TRẬN TƯƠNG QUAN ###########")
        corr_matrix = self.df.corr(numeric_only=True)

        plt.figure(figsize=(12,8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.suptitle("Correlation Matrix (Numerical feature only)")
        corr_img_path = os.path.join(plot_dir, "Correlation_Matrix.png")
        plt.savefig(corr_img_path, bbox_inches='tight')
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

        missing_df = missing_df[missing_df['Missing Count'] > 0]

        # 3. Vẽ biểu đồ
        if not missing_df.empty:
            fig, ax1 = plt.subplots(figsize=(12, 6))

            sns.barplot(x='Column', y='Missing Count', data=missing_df, ax=ax1, palette='viridis', hue='Column', legend=False)
            ax1.set_ylabel('Số lượng bản ghi thiếu', color='b')
            plt.xticks(rotation=45)

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
            nan_img_path = os.path.join(plot_dir, "Count_nan.png")
            plt.savefig(nan_img_path, bbox_inches='tight')
            plt.show()
        else:
            print("Không có dữ liệu thiếu trong DataFrame.")     

        print("######### PHÁT HIỆN OUTLIERS - Phương pháp IQR ##############")
        q_25 = summary.loc['25%']
        q_75 = summary.loc['75%']

        for col in numeric_col: 
            q3, q1 = q_75[col] , q_25[col] 
            iqr = q3 - q1
            lower_bound = q1 - 1.5*iqr
            upper_bound = q3 + 1.5*iqr 

            mask = (self.df[col] > upper_bound) | (self.df[col] < lower_bound) 
            print(f"Số outlier ở cột {col}:", mask.sum())

    def fit(self, X, y=None):
        X_df = X.copy()
        
        # If target column is present in X, drop it (and drop target NaNs if any)
        if self.target_col in X_df.columns:
            X_df = X_df.dropna(subset=[self.target_col])
            X_df = X_df.drop(columns=[self.target_col])
            
        # If imputation is listwise, drop all missing rows from features immediately
        if self.imputation_method == 'listwise':
            X_df = X_df.dropna()
            
        # Identify column types
        self.categorical_cols = X_df.select_dtypes(include=['object']).columns.tolist()
        self.numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 1. Fit standard values for imputation
        for col in self.categorical_cols:
            mode_val = X_df[col].mode()
            self.impute_values[col] = mode_val.iloc[0] if not mode_val.empty else "Missing"
            
        for col in self.numeric_cols:
            self.impute_values[col] = X_df[col].mean() # Default is mean
            
        # 2. Fit advanced imputers
        if self.imputation_method == 'knn':
            X_temp = X_df.copy()
            # Mode fill categoricals to make matrix numerical representation
            for col in self.categorical_cols:
                X_temp[col] = X_temp[col].fillna(self.impute_values[col])
            self.knn_imputer = KNNImputer(n_neighbors=5)
            self.knn_imputer.fit(X_temp[self.numeric_cols])
            
        elif self.imputation_method == 'mice':
            X_temp = X_df.copy()
            for col in self.categorical_cols:
                X_temp[col] = X_temp[col].fillna(self.impute_values[col])
            self.mice_imputer = IterativeImputer(max_iter=10, random_state=42)
            self.mice_imputer.fit(X_temp[self.numeric_cols])
            
        elif self.imputation_method == 'regression':
            X_temp = X_df.copy()
            # Fill both numerical and categorical missing values with basic mean/mode first
            # to prevent NaN errors when fitting helper Ridge models
            for col in self.categorical_cols:
                X_temp[col] = X_temp[col].fillna(self.impute_values[col])
            for col in self.numeric_cols:
                X_temp[col] = X_temp[col].fillna(self.impute_values[col])
            
            # One-hot encode with float cast to prevent boolean type dummies
            X_temp_encoded = pd.get_dummies(X_temp, columns=self.categorical_cols, drop_first=True).astype(float)
            
            self.regression_models = {}
            for col in self.numeric_cols:
                if X_df[col].isnull().any():
                    non_null_mask = X_df[col].notnull()
                    if non_null_mask.sum() > 10:
                        pred_cols = [c for c in X_temp_encoded.columns if c != col]
                        X_train_reg = X_temp_encoded.loc[non_null_mask, pred_cols]
                        y_train_reg = X_df.loc[non_null_mask, col]
                        
                        # We use Ridge regression to perform robust imputation without singularity issues
                        model = Ridge(alpha=1.0)
                        model.fit(X_train_reg, y_train_reg)
                        self.regression_models[col] = (model, pred_cols)
                        
        # 3. Simulate training imputation to compute Outlier bounds on imputed values
        X_imputed = X_df.copy()
        for col in self.categorical_cols:
            X_imputed[col] = X_imputed[col].fillna(self.impute_values[col])
            
        if self.imputation_method == 'knn':
            X_imputed[self.numeric_cols] = self.knn_imputer.transform(X_imputed[self.numeric_cols])
        elif self.imputation_method == 'mice':
            X_imputed[self.numeric_cols] = self.mice_imputer.transform(X_imputed[self.numeric_cols])
        elif self.imputation_method == 'regression':
            for col, (model, pred_cols) in self.regression_models.items():
                null_mask = X_imputed[col].isnull()
                if null_mask.any():
                    X_temp_for_reg = X_imputed.copy()
                    for c in self.numeric_cols:
                        if c != col:
                            X_temp_for_reg[c] = X_temp_for_reg[c].fillna(self.impute_values[c])
                    X_temp_encoded = pd.get_dummies(X_temp_for_reg, columns=self.categorical_cols, drop_first=True).astype(float)
                    for c in pred_cols:
                        if c not in X_temp_encoded.columns:
                            X_temp_encoded[c] = 0.0
                    X_pred = X_temp_encoded.loc[null_mask, pred_cols]
                    X_imputed.loc[null_mask, col] = model.predict(X_pred)
            # final mean imputation fallback
            for col in self.numeric_cols:
                X_imputed[col] = X_imputed[col].fillna(self.impute_values[col])
        elif self.imputation_method == 'listwise':
            # NaNs already removed, do nothing
            pass
        else: # mean/median/mode
            for col in self.numeric_cols:
                X_imputed[col] = X_imputed[col].fillna(self.impute_values[col])
                
        # 4. Outlier Bounds Calculation & Scaling (Standardization)
        for col in self.numeric_cols:
            q1 = X_imputed[col].quantile(0.25)
            q3 = X_imputed[col].quantile(0.75)
            iqr = q3 - q1
            self.outlier_bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
            
            if self.handle_outliers == 'winsorize':
                X_imputed[col] = X_imputed[col].clip(self.outlier_bounds[col][0], self.outlier_bounds[col][1])
                
            self.scale_means[col] = X_imputed[col].mean()
            self.scale_stds[col] = X_imputed[col].std()
            if pd.isna(self.scale_stds[col]) or self.scale_stds[col] == 0:
                self.scale_stds[col] = 1.0
                
        # 5. Determine categorical dummy mappings (avoid perfect multicollinearity)
        self.dummy_columns_map = {}
        for col in self.categorical_cols:
            unique_vals = sorted(X_imputed[col].unique().tolist())
            if len(unique_vals) > 1:
                self.dummy_columns_map[col] = unique_vals[1:] # Drop first
            else:
                self.dummy_columns_map[col] = [] # Avoid dummy trap if 1 category
                
        return self

    def transform(self, X, y=None):
        X_df = X.copy()
        
        y_df = None
        if y is not None:
            y_df = pd.Series(y, index=X_df.index)
            # drop rows where target is NaN
            non_nan_y_mask = y_df.notnull()
            X_df = X_df[non_nan_y_mask]
            y_df = y_df[non_nan_y_mask]
            
        if self.target_col in X_df.columns:
            if y_df is None:
                y_df = X_df[self.target_col]
                non_nan_y_mask = y_df.notnull()
                X_df = X_df[non_nan_y_mask]
                y_df = y_df[non_nan_y_mask]
            X_df = X_df.drop(columns=[self.target_col])
            
        # 1. Apply Imputation
        for col in self.categorical_cols:
            X_df[col] = X_df[col].fillna(self.impute_values[col])
            
        if self.imputation_method == 'listwise':
            nan_rows = X_df.isnull().any(axis=1)
            X_df = X_df[~nan_rows]
            if y_df is not None:
                y_df = y_df[~nan_rows]
        elif self.imputation_method == 'knn':
            X_df[self.numeric_cols] = self.knn_imputer.transform(X_df[self.numeric_cols])
        elif self.imputation_method == 'mice':
            X_df[self.numeric_cols] = self.mice_imputer.transform(X_df[self.numeric_cols])
        elif self.imputation_method == 'regression':
            for col, (model, pred_cols) in self.regression_models.items():
                null_mask = X_df[col].isnull()
                if null_mask.any():
                    X_temp_for_reg = X_df.copy()
                    for c in self.numeric_cols:
                        if c != col:
                            X_temp_for_reg[c] = X_temp_for_reg[c].fillna(self.impute_values[c])
                    X_temp_encoded = pd.get_dummies(X_temp_for_reg, columns=self.categorical_cols, drop_first=True).astype(float)
                    for c in pred_cols:
                        if c not in X_temp_encoded.columns:
                            X_temp_encoded[c] = 0.0
                    X_pred = X_temp_encoded.loc[null_mask, pred_cols]
                    X_df.loc[null_mask, col] = model.predict(X_pred)
            # final mean fallback
            for col in self.numeric_cols:
                X_df[col] = X_df[col].fillna(self.impute_values[col])
        else: # mean_median_mode
            for col in self.numeric_cols:
                X_df[col] = X_df[col].fillna(self.impute_values[col])
                
        # 2. Apply Outlier Winsorization
        if self.handle_outliers == 'winsorize':
            for col in self.numeric_cols:
                lower, upper = self.outlier_bounds[col]
                X_df[col] = X_df[col].clip(lower, upper)
                
        # 3. Categorical encoding
        X_encoded = X_df.copy()
        for col in self.categorical_cols:
            for val in self.dummy_columns_map.get(col, []):
                dummy_col_name = f"{col}_{val}"
                X_encoded[dummy_col_name] = (X_df[col] == val).astype(float)
        # Drop original categoricals
        X_encoded = X_encoded.drop(columns=self.categorical_cols)
        
        # 4. Standardize numeric columns
        for col in self.numeric_cols:
            X_encoded[col] = (X_encoded[col] - self.scale_means[col]) / self.scale_stds[col]
            
        if y_df is not None:
            return X_encoded, y_df
        return X_encoded

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X, y)

if __name__ == '__main__':
    # Script entry point to run original EDA
    dt = DataPipeline()
    dt.EDA()