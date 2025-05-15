import os
import joblib
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

# Danh sách mô hình và scaler tương ứng nếu có
models_info = [
     {
        "model_path": r"UCI_model_800\logistic_regression_model_ML(8001).pkl",
        "scaler_path": r"UCI_model_800\scaler_logistic_regression_model_ML(8001).pkl"
    },
    {
        "model_path": r"UCI_model_800\svm_model_UCI(800).pkl",
        "scaler_path": None
    },
    {
        "model_path": r"UCI_model_800\xgboost_model_ML(800).json",
        "scaler_path": None  # XGBoost không cần scaler
    }
   
]

# Load dữ liệu test
test_data = pd.read_csv(r"Data\heart (1).csv")
X_test = test_data.drop(columns=["target"])
y_test = test_data["target"]

# Lưu kết quả đánh giá từng mô hình
results = []

for info in models_info:
    model_path = info["model_path"]
    scaler_path = info["scaler_path"]
    model_name = os.path.basename(model_path)

    print(f"🔍 Testing model: {model_name}")

    # Load model
    if model_path.endswith(".pkl"):
        model = joblib.load(model_path)
    elif model_path.endswith(".json"):
        model = xgb.XGBClassifier()
        model.load_model(model_path)
    else:
        print(f"⚠️ Unsupported model format: {model_path}")
        continue

    # Load và áp dụng scaler nếu cần
    if scaler_path:
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            X_test_scaled = scaler.transform(X_test)
        else:
            print(f"⚠️ Thiếu scaler cho mô hình {model_name}, bỏ qua...")
            continue
    else:
        X_test_scaled = X_test  # Không cần scaler

    # Dự đoán
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)

    # Tính toán các chỉ số đánh giá
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    logloss = log_loss(y_test, y_pred_proba)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Loss": logloss
    })

# Hiển thị kết quả
df_results = pd.DataFrame(results)
print("\n📊 Tổng hợp kết quả:")
print(df_results)

# Lưu ra file
df_results.to_csv("model_comparison_results(test_80033).csv", index=False)
