import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Thiết lập phong cách seaborn, không dùng lưới
sns.set(style="white")

# Đọc dữ liệu từ file CSV
df_results = pd.read_csv("model_comparison_results.csv")

# Danh sách các mô hình (rút gọn tên để dễ đọc)
models = df_results["Model"].str.replace("model_ML", "").str.replace(".pkl", "").str.replace(".json", "").str.replace("decision_tree_", "DT_").str.replace("knn_", "KNN_").str.replace("logistic_regression_", "LR_").str.replace("random_forest_", "RF_").str.replace("svm_", "SVM_").str.replace("xgboost_", "XGB_").str.replace("(2)", "").str.replace("(3)", "")

# Danh sách các chỉ số
metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "Loss"]

# Tạo 6 biểu đồ con (subplots) với bố cục 3x2 (5 biểu đồ + 1 ô trống)
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(18, 18), sharey=False)
axes = axes.flatten()  # Chuyển mảng 2D thành 1D để dễ xử lý

# Màu sắc khác biệt cho từng mô hình (dùng bảng màu seaborn)
colors = sns.color_palette("Set2", n_colors=len(models))

# Vị trí các mô hình
x = np.arange(len(models))
width = 0.6  # Độ rộng cột

# Vẽ từng biểu đồ cho mỗi chỉ số
for i, metric in enumerate(metrics):
    ax = axes[i]
    bars = ax.bar(x, df_results[metric], width, color=colors, label=models)

    # Ghi giá trị trên đầu cột
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{height:.2f}', 
                ha='center', va='bottom', fontsize=9, color='black')

    # Định dạng biểu đồ
    ax.set_title(f"{metric} Comparison", fontsize=12, fontweight='bold')
    ax.set_ylabel("Score", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0, ha="center", fontsize=9)  # Nhãn nằm ngang
    # Tự động điều chỉnh trục y dựa trên giá trị tối đa
    ax.set_ylim(0, max(df_results[metric]) * 1.2)
    ax.grid(False)  # Tắt lưới

# Xóa ô trống (ô cuối cùng)
axes[-1].axis('off')

# Thêm legend nằm ngang ở dưới cùng
handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(models))]
fig.legend(handles, models, loc='lower center', bbox_to_anchor=(0.5, 0.01), 
           ncol=len(models), title="Models", fontsize=10)

# Điều chỉnh layout để tăng khoảng cách giữa các biểu đồ
plt.tight_layout()
plt.subplots_adjust(hspace=0.5, wspace=0.4, bottom=0.1)  # Tăng khoảng cách dọc (hspace) và ngang (wspace)
plt.show()