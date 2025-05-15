import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
import os

# Định nghĩa hàm để căn chỉnh đặc trưng giữa CDD và UCI
def align_features(df_cdd, df_uci):
    # Mapping tên đặc trưng từ CDD sang UCI
    feature_mapping = {
        'age': 'age',
        'gender': 'sex',
        'chestpain': 'cp',
        'restingBP': 'trestbps',
        'serumcholestrol': 'chol',
        'fastingbloodsugar': 'fbs',
        'restingrelectro': 'restecg',
        'maxheartrate': 'thalach',
        'exerciseangia': 'exang',
        'oldpeak': 'oldpeak',
        'slope': 'slope',
        'noofmajorvessels': 'ca'
    }
    
    # In ra các cột để debug
    print("Cột trong CDD:", df_cdd.columns.tolist())
    print("Cột trong UCI:", df_uci.columns.tolist())
    
    # Lấy danh sách đặc trưng
    cdd_features = list(df_cdd.columns.drop(['patientid', 'target']))
    uci_features = list(df_uci.columns.drop('target'))
    
    # Đặc trưng chung cho CDD và UCI (loại bỏ 'thal')
    common_features = [f for f in cdd_features if f in [k for k, v in feature_mapping.items() if v in uci_features]]
    
    # Đặc trưng cho UCI (bao gồm 'thal')
    uci_model_features = [feature_mapping.get(f, f) for f in common_features] + ['thal']
    
    # Đặc trưng cho CDD_SVM (giả sử tất cả đặc trưng CDD + một đặc trưng giả để đủ 13)
    cdd_model_features = cdd_features 
    
    # In ra các đặc trưng để debug
    print("Các đặc trưng chung (cho CDD):", common_features)
    print("Số đặc trưng chung:", len(common_features))
    print("Các đặc trưng cho mô hình UCI:", uci_model_features)
    print("Số đặc trưng cho mô hình UCI:", len(uci_model_features))
    print("Các đặc trưng cho mô hình CDD:", cdd_model_features)
    print("Số đặc trưng cho mô hình CDD:", len(cdd_model_features))
    
    # Chuẩn bị dữ liệu cho CDD
    cdd_X = df_cdd[common_features]
    cdd_y = df_cdd['target']
    
    # Chuẩn bị dữ liệu cho UCI
    uci_X = df_uci[[feature_mapping.get(f, f) for f in common_features]]
    uci_y = df_uci['target']
    
    # Thêm cột 'thal' giả cho CDD và đổi tên cột để khớp với UCI
    cdd_X_for_uci = cdd_X.copy()
    cdd_X_for_uci['thal'] = 0  # Gán giá trị giả
    cdd_X_for_uci.columns = uci_model_features  # Đổi tên cột sang tên UCI
    
    # Thêm cột giả cho uci_X để khớp với CDD_SVM (13 đặc trưng)
    uci_X_for_cdd = uci_X.copy()

    uci_X_for_cdd.columns = cdd_model_features  # Đổi tên cột sang tên CDD
    
    # Đảm bảo số cột khớp
    if len(common_features) != uci_X.shape[1]:
        raise ValueError(f"Số đặc trưng không khớp: common_features ({len(common_features)}) != uci_X ({uci_X.shape[1]})")
    if len(uci_model_features) != cdd_X_for_uci.shape[1]:
        raise ValueError(f"Số đặc trưng không khớp: uci_model_features ({len(uci_model_features)}) != cdd_X_for_uci ({cdd_X_for_uci.shape[1]})")
    if len(cdd_model_features) != uci_X_for_cdd.shape[1]:
        raise ValueError(f"Số đặc trưng không khớp: cdd_model_features ({len(cdd_model_features)}) != uci_X_for_cdd ({uci_X_for_cdd.shape[1]})")
    
    return cdd_X, cdd_y, uci_X, uci_y, common_features, cdd_X_for_uci, uci_model_features, uci_X_for_cdd, cdd_model_features

# Hàm đánh giá mô hình
def evaluate_model(model, X_test, y_test, model_name, dataset_name):
    # Xử lý dữ liệu kiểm tra tùy thuộc vào mô hình
    if 'SVM' in model_name and 'CDD' in model_name:
        # CDD_SVM được huấn luyện không có tên đặc trưng
        X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    elif 'RF' in model_name and 'CDD' in model_name:
        # CDD_RF được huấn luyện không có tên đặc trưng
        X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    else:
        # UCI_SVM, UCI_RF, và các mô hình khác được huấn luyện với tên đặc trưng
        X_test_np = X_test
    
    # Xử lý dự đoán
    if isinstance(model, xgb.Booster):
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test_np)
        y_pred_proba = model.predict_proba(X_test_np)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # In ra một vài dự đoán để debug
    print(f"Dự đoán mẫu cho {model_name} trên {dataset_name}: {y_pred[:5]}")
    print(f"Nhãn thực tế mẫu: {y_test[:5].values}")
    
    # Tính các chỉ số
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else np.nan
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    return {
        'Model': model_name,
        'Test_Dataset': dataset_name,
        'Accuracy': accuracy,
        'F1_Score': f1,
        'AUC': auc,
        'Precision': precision,
        'Recall': recall
    }

# Đường dẫn đến các mô hình
model_paths = {
    'CDD_SVM': r'CDD_Model\svm_model_CDD(final).pkl',
    'CDD_LG': r'CDD_Model\logistic_regression_model_ML(CDD2).pkl',
    'CDD_XGB': r'CDD_Model\xgboost_model_CDD(final).json',
    'UCI_SVM': r'UCI_Model_300\svm_model_UCI(300).pkl',
    'UCI_LG': r'UCI_Model_300\logistic_regression_model_ML(300).pkl',
    'UCI_XGB': r'UCI_Model_300\xgboost_model_ML(300_UCI).json'
}

# Tải dữ liệu
cdd_data = pd.read_csv(r'Data\Cardiovascular_Disease_Dataset.csv')
uci_data = pd.read_csv(r'Data\heart.csv')

# Kiểm tra nhãn để đảm bảo nhị phân
print("Nhãn CDD duy nhất:", np.unique(cdd_data['target']))
print("Nhãn UCI duy nhất:", np.unique(uci_data['target']))

# Căn chỉnh đặc trưng
cdd_X, cdd_y, uci_X, uci_y, common_features, cdd_X_for_uci, uci_model_features, uci_X_for_cdd, cdd_model_features = align_features(cdd_data, uci_data)

# Chuẩn bị để lưu kết quả
results = []

# Tải và đánh giá các mô hình
for model_name, model_path in model_paths.items():
    print(f"Đánh giá mô hình: {model_name}")
    
    # Tải mô hình
    if model_name.endswith('XGB'):
        model = xgb.Booster()
        model.load_model(model_path)
        # Chuyển dữ liệu sang DMatrix để XGBoost
        X_cdd_dmatrix = xgb.DMatrix(cdd_X, feature_names=common_features)
        X_uci_dmatrix = xgb.DMatrix(uci_X, feature_names=common_features)  # Sử dụng uci_X (12 đặc trưng) cho CDD_XGB
        X_cdd_dmatrix_for_uci = xgb.DMatrix(cdd_X_for_uci, feature_names=uci_model_features)
    else:
        model = joblib.load(model_path)
        X_cdd_dmatrix = cdd_X
        if 'SVM' in model_name and 'CDD' in model_name:
            X_uci_dmatrix = uci_X_for_cdd  # Sử dụng uci_X_for_cdd (13 đặc trưng) cho CDD_SVM
        else:
            X_uci_dmatrix = uci_X  # Sử dụng uci_X (12 đặc trưng) cho CDD_RF và các mô hình khác
        X_cdd_dmatrix_for_uci = cdd_X_for_uci
    
    # Kiểm tra số đặc trưng của mô hình
    if hasattr(model, 'n_features_in_'):
        print(f"Số đặc trưng kỳ vọng của {model_name}: {model.n_features_in_}")
    
    # Đánh giá cross-dataset
    if model_name.startswith('CDD'):
        # Mô hình huấn luyện trên CDD, kiểm tra trên UCI
        result = evaluate_model(model, X_uci_dmatrix, uci_y, model_name, 'UCI')
        results.append(result)
    elif model_name.startswith('UCI'):
        # Mô hình huấn luyện trên UCI, kiểm tra trên CDD
        result = evaluate_model(model, X_cdd_dmatrix_for_uci, cdd_y, model_name, 'CDD')
        results.append(result)

# Lưu kết quả vào file CSV
results_df = pd.DataFrame(results)
results_df.to_csv('cross_dataset_evaluation_results(4).csv', index=False)
print("Kết quả đã được lưu vào 'cross_dataset_evaluation_results.csv'")

# Hiển thị kết quả
print("\nKết quả đánh giá:")
print(results_df)
