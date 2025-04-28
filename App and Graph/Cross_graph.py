import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Thiết lập phong cách seaborn
sns.set(style="white")

# Đọc dữ liệu
cross_data = pd.read_csv('cross_dataset_evaluation_results.csv')

# Nếu cần chuẩn hóa tên Model (có thể bỏ qua nếu tên đã OK)
model_name_mapping = {
    'CDD_SVM': 'SVM_CDD',
    'CDD_RF': 'RF_CDD',
    'CDD_XGB': 'XGB_CDD',
    'UCI_SVM': 'SVM_UCI',
    'UCI_RF': 'RF_UCI',
    'UCI_XGB': 'XGB_UCI'
}
cross_data['Model'] = cross_data['Model'].map(model_name_mapping)

# Chuẩn bị dữ liệu
models = cross_data.apply(lambda x: f"{x['Model']} ({x['Test_Dataset']})", axis=1).values
x = np.arange(len(models))
width = 0.15

# Tạo biểu đồ
fig, ax = plt.subplots(figsize=(16, 6))

# Vẽ các cột
bars1 = ax.bar(x - 2*width, cross_data["Accuracy"], width, label="Accuracy", color='#3B82F6')
bars2 = ax.bar(x - width, cross_data["Precision"], width, label="Precision", color='#10B981')
bars3 = ax.bar(x, cross_data["Recall"], width, label="Recall", color='#F59E0B')
bars4 = ax.bar(x + width, cross_data["F1_Score"], width, label="F1-Score", color='#EF4444')
bars5 = ax.bar(x + 2*width, cross_data["AUC"], width, label="AUC", color='#8B5CF6')

# Ghi giá trị trên từng cột
for bars in [bars1, bars2, bars3, bars4, bars5]:
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{height:.2f}',
                    ha='center', va='bottom', fontsize=8)

# Tùy chỉnh
ax.set_title("Model Comparison - Cross Dataset Evaluation", fontsize=14, fontweight='bold')
ax.set_xlabel("Model (Test Dataset)", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha="right")
ax.set_ylim(0, 1.2)
ax.legend(title="Metrics", fontsize=10)
ax.grid(False)

plt.tight_layout()
plt.savefig('cross_model_performance_comparison.png')
plt.close()

print("Biểu đồ đã được lưu vào 'cross_model_performance_comparison.png'")
