import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Định nghĩa các đặc trưng
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

# Load mô hình đã train
MODEL_PATH = r"Last_Model\decision_tree_model_ML(2).pkl"
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = joblib.load(MODEL_PATH)
        if not isinstance(model, DecisionTreeClassifier):
            st.warning("Mô hình không phải là DecisionTreeClassifier.")
    except Exception as e:
        st.error(f"Không thể tải mô hình: {str(e)}")
        st.stop()
else:
    st.error("Không tìm thấy file mô hình. Vui lòng kiểm tra đường dẫn.")
    st.stop()

# Kiểm tra tên đặc trưng
if hasattr(model, "feature_names_in_"):
    st.info(f"Model trained with feature names: {model.feature_names_in_}")
else:
    st.warning("Model was not trained with feature names. Using default order: {FEATURES}")

# Tạo hoặc tải dữ liệu lịch sử
HISTORY_FILE = "history_decision_tree.csv"
if os.path.exists(HISTORY_FILE):
    history_df = pd.read_csv(HISTORY_FILE)
else:
    history_df = pd.DataFrame(columns=FEATURES + ["prediction", "probability"])

# Tạo giao diện với sidebar
st.sidebar.title("Heart Disease Prediction System")
page = st.sidebar.radio("Chọn trang", ["Prediction", "Record"])

if page == "Prediction":
    st.title("Heart Disease Prediction")
    
    # Nhập dữ liệu từ người dùng
    with st.form("input_form"):
        age = st.number_input("Age", min_value=20, max_value=100, value=50)
        sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"][x])
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], format_func=lambda x: "False" if x == 0 else "True")
        restecg = st.selectbox("Resting ECG Results", [0, 1, 2], format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x])
        thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
        exang = st.selectbox("Exercise Induced Angina", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.)