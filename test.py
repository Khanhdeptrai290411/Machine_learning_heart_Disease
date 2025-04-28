import os
import warnings
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

# Ignore all sklearn warnings
warnings.filterwarnings("ignore")

# Danh sách các mô hình cần test
model_paths = [
    r"CDD_Model\random_forest_model_CDD(CV).pkl",
    r"CDD_Model\svm_model_CDD(CV).pkl",
    r"CDD_Model\xgboost_model_CDD(no_CV(1)).json",
]
# Load dữ liệu test
test_data = pd.read_csv("Data/Cardiovascular_Disease_Dataset.csv")

# Kiểm tra tỷ lệ lớp
print("Tỷ lệ lớp trong tập kiểm tra:")
print(test_data["target"].value_counts(normalize=True))

# Lưu X_gốc và y
X_test_full = test_data.drop(columns=["target"])
y_test = test_data["target"]

# Xóa 'patientid' để có phiên bản X_test không có patientid
if 'patientid' in X_test_full.columns:
    X_test_nopid = X_test_full.drop(columns=["patientid"])
else:
    X_test_nopid = X_test_full.copy()

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_test_full_scaled = scaler.fit_transform(X_test_full)
X_test_nopid_scaled = scaler.fit_transform(X_test_nopid)
# Lưu kết quả
results = []

for model_path in model_paths:
    print(f"Testing model: {model_path}")

    model_name = os.path.basename(model_path)

    # Load model
    if model_path.endswith(".pkl"):
        model = joblib.load(model_path)
    elif model_path.endswith(".json"):
        model = xgb.XGBClassifier()
        model.load_model(model_path)
    else:
        print(f"Unsupported model format: {model_path}")
        continue

    # Kiểm tra số lượng feature mà model expect
    n_features_model = getattr(model, "n_features_in_", None)

    # Chọn X_test phù hợp
    if n_features_model == 13:
        X_test = X_test_full_scaled
    elif n_features_model == 12:
        X_test = X_test_nopid_scaled
    else:
        raise ValueError(f"Unexpected number of features expected by model {model_name}: {n_features_model}")

    # Predict
    y_pred = model.predict(X_test)

    # Predict probability
    try:
        y_pred_proba = model.predict_proba(X_test)
        logloss = log_loss(y_test, y_pred_proba)
    except AttributeError:
        logloss = "N/A"

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Save result
    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Loss": logloss
    })

# Tạo DataFrame kết quả
df_results = pd.DataFrame(results)
print(df_results)
df_results.to_csv("model_comparison_results_CDD.csv", index=False)