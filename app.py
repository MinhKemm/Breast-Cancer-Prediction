import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Cấu hình giao diện ---
st.set_page_config(page_title="Chẩn đoán Ung thư Vú", layout="wide")

# --- Đường dẫn các file ---
# Vì file app.py nằm ở gốc, ta trỏ vào các thư mục con
SCALER_PATH = os.path.join("Data", "scaler.pkl")
MODEL_DIR = os.path.join("Notebooks", "models")

@st.cache_resource
def load_assets():
    # Load Scaler
    scaler = joblib.load(SCALER_PATH)
    
    # Load Models
    model_names = {
        'Random Forest': 'random_forest.pkl',
        'SVM': 'svm.pkl',
        'KNN': 'knn.pkl',
        'Logistic Regression': 'logistic_regression.pkl',
        'XGBoost': 'xgboost.pkl',
        'AdaBoost': 'adaboost.pkl',
        'Gradient Boosting': 'gradient_boosting.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'OSEL': 'osel.pkl'
    }
    
    loaded_models = {}
    for name, filename in model_names.items():
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            loaded_models[name] = joblib.load(path)
            
    return scaler, loaded_models

# Thử load dữ liệu
try:
    scaler, models = load_assets()
except Exception as e:
    st.error(f"Lỗi khi load dữ liệu: {e}. Hãy đảm bảo đã có file scaler.pkl trong thư mục Data.")
    st.stop()

# --- Giao diện người dùng ---
st.title("Hệ thống Chẩn đoán Ung thư Vú")
st.sidebar.header("Tùy chọn")
selected_model = st.sidebar.selectbox("Chọn mô hình dự đoán", list(models.keys()))

st.write("### Nhập các chỉ số xét nghiệm (30 tham số)")

# Danh sách 30 tham số (đúng thứ tự cột dữ liệu)
features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

# Tạo các cột để nhập liệu cho đẹp
input_values = []
cols = st.columns(3)
for i, feat in enumerate(features):
    with cols[i % 3]:
        val = st.number_input(f"{feat}", value=0.0, format="%.4f")
        input_values.append(val)

st.divider()

if st.button("Dự đoán ngay", type="primary"):
    # 1. Chuyển thành mảng 2D
    raw_data = np.array([input_values])
    
    # 2. Chuẩn hóa bằng scaler đã lưu
    scaled_data = scaler.transform(raw_data)
    
    # 3. Dự đoán bằng model đã chọn
    model = models[selected_model]
    prediction = model.predict(scaled_data)
    
    # 4. Hiển thị kết quả
    if prediction[0] == 1:
        st.error(f"Kết quả từ {selected_model}: **ÁC TÍNH (Malignant)**")
    else:
        st.success(f"Kết quả từ {selected_model}: **LÀNH TÍNH (Benign)**")
    
    # Hiển thị xác suất nếu có
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(scaled_data)[0]
        st.info(f"Độ tin cậy: {np.max(prob)*100:.2f}%")