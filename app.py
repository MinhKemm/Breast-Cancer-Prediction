import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(page_title="Breast Cancer Diagnosis", layout="wide")

@st.cache_resource
def load_assets():
    scaler = joblib.load(os.path.join("Data", "scaler.pkl"))
    model_names = {
        'Random Forest': 'random_forest.pkl', 'SVM': 'svm.pkl', 
        'KNN': 'knn.pkl', 'Logistic Regression': 'logistic_regression.pkl',
        'XGBoost': 'xgboost.pkl', 'AdaBoost': 'adaboost.pkl',
        'Gradient Boosting': 'gradient_boosting.pkl', 'Decision Tree': 'decision_tree.pkl',
        'OSEL': 'osel.pkl'
    }
    loaded_models = {name: joblib.load(os.path.join("Notebooks", "models", path)) 
                     for name, path in model_names.items() if os.path.exists(os.path.join("Notebooks", "models", path))}
    return scaler, loaded_models

scaler, models = load_assets()

features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

st.title("Breast Cancer Prediction")

selected_model_name = st.sidebar.selectbox("Chọn mô hình dự đoán:", list(models.keys()))
model = models[selected_model_name]

tab1, tab2 = st.tabs(["Nhập thủ công", "Nhập nhanh (Paste chuỗi)"])

input_final = None

with tab1:
    st.write("Nhập từng thông số vào các ô bên dưới:")
    manual_data = []
    cols = st.columns(4)
    for i, feat in enumerate(features):
        with cols[i % 4]:
            val = st.number_input(f"{feat}", value=0.0, format="%.4f", key=f"manual_{feat}")
            manual_data.append(val)
    if st.button("Dự đoán bằng dữ liệu nhập tay"):
        input_final = np.array([manual_data])

with tab2:
    st.write("Dán chuỗi 30 thông số cách nhau bằng dấu phẩy (Ví dụ từ file CSV):")
    raw_string = st.text_area("Chuỗi dữ liệu:", placeholder="17.99, 10.38, 122.8, ...", height=100)
    
    if st.button("Dự đoán bằng chuỗi đã dán"):
        try:
            # Xử lý chuỗi: tách dấu phẩy, bỏ khoảng trắng và chuyển sang float
            list_values = [float(x.strip()) for x in raw_string.split(",")]
            
            if len(list_values) != 30:
                st.error(f"Dữ liệu không hợp lệ! Bạn cần nhập đủ 30 thông số. Hiện tại đang có: {len(list_values)}")
            else:
                input_final = np.array([list_values])
        except ValueError:
            st.error("Lỗi: Chuỗi nhập vào chứa ký tự không phải là số.")

if input_final is not None:
    st.markdown("---")
    input_scaled = scaler.transform(input_final)

    prediction = model.predict(input_scaled)

    st.subheader("Kết quả phân tích chi tiết")

    with st.container():
        res_col1, res_col2 = st.columns([1, 1])
        
        with res_col1:
            st.markdown(f"**Mô hình đang dùng:** `{selected_model_name}`")
            if prediction[0] == 1:
                st.error("### CHẨN ĐOÁN: ÁC TÍNH (M)")
                st.markdown("*Cần sự can thiệp và kiểm tra chuyên sâu từ bác sĩ.*")
            else:
                st.success("### CHẨN ĐOÁN: LÀNH TÍNH (B)")
                st.markdown("*Các chỉ số nằm trong ngưỡng ít nguy hiểm.*")

        with res_col2:
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_scaled)[0]
                confidence = np.max(prob) * 100

                st.metric("Độ tin cậy của dự đoán", f"{confidence:.2f}%")
                st.progress(confidence / 100)
                st.caption("Xác suất phân loại dựa trên tập dữ liệu huấn luyện.")
            else:
                st.info("Mô hình này không cung cấp dữ liệu về xác suất (Độ tin cậy).")

    st.caption("---")
    st.caption("Khuyến cáo: Đây là công cụ hỗ trợ dựa trên AI, không thay thế cho chẩn đoán lâm sàng chính thức.")