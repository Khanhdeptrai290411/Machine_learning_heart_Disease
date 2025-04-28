import streamlit as st
import pandas as pd
import joblib
import os
import xgboost as xgb
import numpy as np

# Đường dẫn đến các mô hình
MODEL_PATHS = {
    
    "KNN": r"Last_Model\knn_model_ML.pkl",
    "Logistic Regression": r"Last_Model\logistic_regression_model_ML.pkl",
    "Random Forest": r"Last_Model\random_forest_model_ML(3).pkl",
    "SVM": r"Last_Model\svm_model_ML(2).pkl",
    "XGBoost": r"Last_Model\xgboost_model_ML(3).json",
    "Decision Tree": r"Last_Model\decision_tree_model_ML(3).pkl"
}

# Load các mô hình
models = {}
model_options = []
for model_name, model_path in MODEL_PATHS.items():
    if os.path.exists(model_path):
        try:
            if model_name == "XGBoost":
                # Load mô hình XGBoost
                model = xgb.XGBClassifier()
                model.load_model(model_path)
                model_type = 'xgboost'
            else:
                # Load các mô hình khác (Decision Tree, KNN, Logistic Regression, Random Forest, SVM)
                with open(model_path, "rb") as f:
                    model = joblib.load(model_path)
                model_type = 'other'
            model_options.append(model_name)
            models[model_name] = (model, model_type)
        except Exception as e:
            st.error(f"Lỗi khi load mô hình {model_name}: {str(e)}")

# Kiểm tra xem có mô hình nào được load thành công không
if not models:
    st.error("Không thể load bất kỳ mô hình nào. Vui lòng kiểm tra lại.")
    st.stop()

# Tạo hoặc tải dữ liệu lịch sử
HISTORY_FILE = "history_predictions.csv"
if os.path.exists(HISTORY_FILE):
    history_df = pd.read_csv(HISTORY_FILE)
else:
    history_df = pd.DataFrame(columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", 
                                       "thalach", "exang", "oldpeak", "slope", "ca", "thal", 
                                       "prediction", "model_used"], dtype=float)

# Tạo giao diện với sidebar
st.sidebar.title("Heart Disease Prediction System")
page = st.sidebar.radio("Chọn trang", ["Prediction", "Record"])

if page == "Prediction":
    st.title("Heart Disease Prediction")
    
    # Chọn mô hình
    if not model_options:
        st.error("Không có mô hình nào khả dụng để dự đoán.")
        st.stop()
    
    selected_model_name = st.selectbox("Chọn mô hình để dự đoán:", model_options)
    selected_model, model_type = models[selected_model_name]
    
    # Nhập dữ liệu từ người dùng
    age = st.number_input("Age", min_value=20, max_value=100, value=50)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (0 = False, 1 = True)", [0, 1])
    restecg = st.selectbox("Resting ECG Results (0-2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved (bpm)", min_value=60, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", [0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise (mm)", min_value=0.0, max_value=6.0, value=1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (0-2)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (0-4)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

    threshold = 0.4 # Cố định ngưỡng tại 0.5 (có thể thay đổi giá trị này)
    # Thêm thanh trượt để điều chỉnh ngưỡng
    # threshold = st.slider("Prediction Threshold (for class 1)", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
    
    # st.write(f"Threshold: If probability of Positive (1) > {threshold}, predict Positive.")
    
    # Dự đoán
    if st.button("Predict"):
        input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]],
                                  columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"])
        
        try:
            # Chuyển DataFrame thành NumPy array
            input_array = input_data.to_numpy()
            
            # Dự đoán dựa trên loại mô hình
            if model_type == 'xgboost':
                # XGBoost: Dùng DMatrix cho predict(), NumPy array cho predict_proba()
                dmatrix = xgb.DMatrix(input_array)
                prediction_prob = selected_model.predict_proba(input_array)[0]  # Lấy xác suất
                # Điều chỉnh dự đoán dựa trên ngưỡng
                prediction = 1 if prediction_prob[1] > threshold else 0
            else:
                # Các mô hình khác (Decision Tree, KNN, Logistic Regression, Random Forest, SVM)
                prediction_prob = selected_model.predict_proba(input_array)[0]
                prediction = 1 if prediction_prob[1] > threshold else 0
            
            result = "Positive (Có nguy cơ mắc bệnh tim)" if prediction == 1 else "Negative (Không có nguy cơ)"
            
            # Hiển thị kết quả và xác suất
            st.success(f"Prediction (using {selected_model_name}): {result}")
            st.write(f"Probability of Negative (0): {prediction_prob[0]:.2f}")
            st.write(f"Probability of Positive (1): {prediction_prob[1]:.2f}")
            
            # Cảnh báo nếu dự đoán không chắc chắn
            if 0.4 < prediction_prob[1] < 0.6:
                st.warning("Prediction is uncertain (probability close to 0.5). Consider adjusting the threshold or trying another model.")
            
            # Lưu kết quả vào history
            new_record = input_data.copy()
            new_record["prediction"] = float(prediction)
            new_record["model_used"] = selected_model_name
            history_df = pd.concat([history_df, new_record], ignore_index=True)
            history_df.to_csv(HISTORY_FILE, index=False)
        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {str(e)}")

elif page == "Record":
    st.title("Prediction Records")
    if history_df.empty:
        st.write("No records found.")
    else:
        st.dataframe(history_df)

