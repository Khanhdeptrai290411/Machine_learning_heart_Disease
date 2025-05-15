import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Thiết lập phong cách seaborn, không dùng lưới
sns.set(style="white")

# Đọc dữ liệu từ các file CSV
try:
    uci_data = pd.read_csv('model_comparison_results(test_300).csv')
    cdd_data = pd.read_csv('History_CDD\model_comparison_results_CDD(3).csv')
    cross_data = pd.read_csv('cross_History\cross_dataset_evaluation_results(3).csv')
except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
    exit(1)

# Debug: In tên cột để kiểm tra
print("Columns in model_comparison_results(300).csv:", uci_data.columns.tolist())
print("Columns in model_comparison_results_CDD(3).csv:", cdd_data.columns.tolist())
print("Columns in cross_dataset_evaluation_results(3).csv:", cross_data.columns.tolist())

# Chuẩn hóa tên mô hình để dễ hiển thị
model_name_mapping = {
    'logistic_regression_model_ML(300).pkl': 'LG_UCI',
    'svm_model.pkl': 'SVM_UCI',
    'xgboost_model.json': 'XGB_UCI',
    'logistic_regression_model_ML(CDD2).pkl': 'LG_CDD',
    'svm_model_CDD(final).pkl': 'SVM_CDD',
    'xgboost_model_CDD(final).json': 'XGB_CDD',
    'CDD_SVM': 'SVM_CDD',
    'CDD_LG': 'LG_CDD',
    'CDD_XGB': 'XGB_CDD',
    'UCI_SVM': 'SVM_UCI',
    'UCI_LG': 'LG_UCI',
    'UCI_XGB': 'XGB_UCI'
}

# Xử lý dữ liệu UCI
if 'Model' in uci_data.columns:
    uci_data['Model'] = uci_data['Model'].map(model_name_mapping)
else:
    print("Error: 'Model' column not found in model_comparison_results(3).csv")
    exit(1)
uci_data['Test_Dataset'] = 'UCI'
uci_data = uci_data.rename(columns={'F1-Score': 'F1_Score'})
uci_data['AUC'] = np.nan  # Không có AUC trong file này

# Xử lý dữ liệu CDD
if 'Model' in cdd_data.columns:
    cdd_data['Model'] = cdd_data['Model'].map(model_name_mapping)
else:
    print("Error: 'Model' column not found in model_comparison_results_CDD.csv")
    exit(1)
cdd_data['Test_Dataset'] = 'CDD'
cdd_data = cdd_data.rename(columns={'F1-Score': 'F1_Score'})
cdd_data['AUC'] = np.nan  # Không có AUC trong file này

# Xử lý dữ liệu cross-dataset
if 'Model' in cross_data.columns:
    cross_data['Model'] = cross_data['Model'].map(model_name_mapping)
else:
    print("Error: 'Model' column not found in cross_dataset_evaluation_results.csv")
    exit(1)
cross_data = cross_data[['Model', 'Test_Dataset', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC']]

# Kết hợp dữ liệu
combined_data = pd.concat([uci_data, cdd_data, cross_data], ignore_index=True)

# Debug: In dữ liệu kết hợp để kiểm tra
print("Combined data sample:\n", combined_data.head())

# Chuẩn bị dữ liệu cho biểu đồ
models = combined_data.apply(lambda x: f"{x['Model']} ({x['Test_Dataset']})", axis=1).values
x = np.arange(len(models))
width = 0.15  # Độ rộng cột, điều chỉnh để chứa 5 chỉ số

# Tạo biểu đồ
fig, ax = plt.subplots(figsize=(16, 6))

# Vẽ cột cho từng chỉ số
bars1 = ax.bar(x - 2*width, combined_data["Accuracy"], width, label="Accuracy", color='#3B82F6')
bars2 = ax.bar(x - width, combined_data["Precision"], width, label="Precision", color='#10B981')
bars3 = ax.bar(x, combined_data["Recall"], width, label="Recall", color='#F59E0B')
bars4 = ax.bar(x + width, combined_data["F1_Score"], width, label="F1-Score", color='#EF4444')
bars5 = ax.bar(x + 2*width, combined_data["AUC"], width, label="AUC", color='#8B5CF6')

# Ghi giá trị trên cột
for bars in [bars1, bars2, bars3, bars4, bars5]:
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):  # Chỉ ghi giá trị nếu không phải NaN
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.2f}',
                    ha='center', va='bottom', fontsize=8)

# Tùy chỉnh biểu đồ
ax.set_title("Model Comparison - Performance Metrics (UCI, CDD, Cross-Dataset)", fontsize=14, fontweight='bold')
ax.set_xlabel("Model (Test Dataset)", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha="right")
ax.set_ylim(0, 1.2)  # Điều chỉnh trục y để phù hợp với giá trị
ax.legend(title="Metrics", fontsize=10)
ax.grid(False)  # Không hiển thị lưới

plt.tight_layout()
plt.savefig('model_performance_comparison4.png')
plt.close()

# In thông báo
print("Biểu đồ đã được lưu vào 'model_performance_comparison.png'")