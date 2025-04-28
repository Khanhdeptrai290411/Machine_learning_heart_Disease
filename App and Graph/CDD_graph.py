import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Đọc dữ liệu từ các file CSV
uci_data = pd.read_csv('model_comparison_results(3).csv')
cdd_data = pd.read_csv('model_comparison_results_CDD.csv')
cross_data = pd.read_csv('cross_dataset_evaluation_results.csv')

# Chuẩn hóa tên mô hình để dễ hiển thị
model_name_mapping = {
    'random_forest_model_UCI(CV).pkl': 'RF_UCI',
    'svm_model_ML(2).pkl': 'SVM_UCI',
    'xgboost_model_UCI(have_CV(1)).json': 'XGB_UCI',
    'random_forest_model_CDD(CV).pkl': 'RF_CDD',
    'svm_model_CDD(CV).pkl': 'SVM_CDD',
    'xgboost_model_CDD(no_CV(1)).json': 'XGB_CDD',
    'CDD_SVM': 'SVM_CDD',
    'CDD_RF': 'RF_CDD',
    'CDD_XGB': 'XGB_CDD',
    'UCI_SVM': 'SVM_UCI',
    'UCI_RF': 'RF_UCI',
    'UCI_XGB': 'XGB_UCI'
}

# Xử lý dữ liệu UCI
uci_data['Model'] = uci_data['Model'].map(model_name_mapping)
uci_data['Test_Dataset'] = 'UCI'
uci_data = uci_data.rename(columns={'F1-Score': 'F1_Score'})
uci_data['AUC'] = np.nan  # Không có AUC trong file này

# Xử lý dữ liệu CDD
cdd_data['Model'] = cdd_data['Model'].map(model_name_mapping)
cdd_data['Test_Dataset'] = 'CDD'
cdd_data = cdd_data.rename(columns={'F1-Score': 'F1_Score'})
cdd_data['AUC'] = np.nan  # Không có AUC trong file này

# Xử lý dữ liệu cross-dataset
cross_data['Model'] = cross_data['Model'].map(model_name_mapping)
cross_data = cross_data[['Model', 'Test_Dataset', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC']]

# Kết hợp dữ liệu
combined_data = pd.concat([uci_data, cdd_data, cross_data], ignore_index=True)

# Lọc các chỉ số cần vẽ
metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'AUC']
data_to_plot = combined_data[['Model', 'Test_Dataset'] + metrics]

# Chuẩn bị dữ liệu cho biểu đồ
models = data_to_plot.apply(lambda x: f"{x['Model']} ({x['Test_Dataset']})", axis=1).values
n_models = len(models)
n_metrics = len(metrics)
bar_width = 0.15
index = np.arange(n_models)

# Tạo biểu đồ
plt.figure(figsize=(16, 8))

# Vẽ cột cho từng chỉ số
colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
for i, metric in enumerate(metrics):
    values = data_to_plot[metric].values
    plt.bar(index + i * bar_width, values, bar_width, label=metric, color=colors[i])

# Tùy chỉnh biểu đồ
plt.xlabel('Model (Test Dataset)')
plt.ylabel('Score')
plt.title('Performance Comparison of Models on UCI and CDD Datasets')
plt.xticks(index + bar_width * (n_metrics - 1) / 2, models, rotation=45, ha='right')
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

# Lưu biểu đồ
plt.savefig('model_performance_comparison.png')
plt.close()

# In thông báo
print("Biểu đồ đã được lưu vào 'model_performance_comparison.png'")