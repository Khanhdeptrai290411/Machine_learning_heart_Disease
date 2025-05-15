import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss

# Danh sách 6 mô hình đã lưu
model_paths = [
    r"UCI_model_800\svm_model_UCI(800).pkl",
    r"UCI_model_800\xgboost_model_ML(800).json",
    # r"UCI_model_800\logistic_regression_model_ML(8001).pkl",
    # r"UCI_Model_300\decision_tree_model_ML(2).pkl",
    # r"UCI_Model_300\knn_model_ML(300(1)).pkl",
    # r"UCI_Model_300\logistic_regression_model_ML(300).pkl",
    # r"UCI_Model_300\random_forest_model_UCI(300(1)).pkl",
    # r"UCI_Model_300\svm_model_UCI(300).pkl",
    # r"UCI_Model_300\xgboost_model_ML(300_UCI).json"
]

# Load dữ liệu test
test_data = pd.read_csv(r"Data\heart (1).csv")
X_test = test_data.drop(columns=["target"])
y_test = test_data["target"]

# Lưu kết quả đánh giá từng mô hình
results = []

for model_path in model_paths:
    print(f"Testing model: {model_path}")

    # Lấy tên file từ đường dẫn
    model_name = os.path.basename(model_path)

    # Tải mô hình
    if model_path.endswith(".pkl"):
        model = joblib.load(model_path)
    elif model_path.endswith(".json"):
        model = xgb.XGBClassifier()
        model.load_model(model_path)
    else:
        print(f"Unsupported model format: {model_path}")
        continue

    # Dự đoán nhãn
    y_pred = model.predict(X_test)

    # Dự đoán xác suất để tính log_loss
    y_pred_proba = model.predict_proba(X_test)

    # Đánh giá hiệu suất
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)

    # Lưu kết quả
    results.append({
        "Model": model_name,  # Chỉ lấy tên file
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Loss": logloss
    })

# Tạo DataFrame và lưu file CSV
df_results = pd.DataFrame(results)
print(df_results)
df_results.to_csv("model_comparison_results(test_8000).csv", index=False)