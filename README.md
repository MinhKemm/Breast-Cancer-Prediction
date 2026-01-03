# 🩺 Dự đoán Ung thư vú (Breast Cancer Prediction)

## 📋 Giới thiệu Dự án
Dự án tập trung vào việc xây dựng và so sánh các mô hình **Machine Learning** nhằm dự đoán khả năng mắc **ung thư vú** dựa trên các đặc trưng lâm sàng.  
Mục tiêu là lựa chọn mô hình có **độ chính xác và độ tin cậy cao nhất**, đặc biệt theo các chỉ số **F1-Score** và **Recall**, nhằm hỗ trợ quá trình chẩn đoán y tế.

Dự án triển khai nhiều thuật toán **Ensemble Learning** như:
- AdaBoost  
- Random Forest  
- Gradient Boosting  
- **OSEL (Optimal Stacking Ensemble Learning)** – mô hình Stacking tối ưu

---

## 📂 Cấu trúc Thư mục
Dự án được tổ chức theo cấu trúc sau:

```plaintext
.
├── Data/                   # Dữ liệu thô và dữ liệu đã xử lý (.npy)
├── Notebooks/              # Các Jupyter Notebook thực nghiệm
│   ├── Data_preprocessing.ipynb
│   ├── AdaBoost.ipynb
│   ├── Random_Forest.ipynb
│   ├── Gradient_boosting.ipynb
│   ├── OSEL.ipynb          # Mô hình Stacking Ensemble
│   └── models/             # Các mô hình đã huấn luyện (.pkl)
├── Results/                # Biểu đồ so sánh (Accuracy, F1, Recall)
├── Compare_models.py       # Script đánh giá & so sánh mô hình
├── requirements.txt        # Danh sách thư viện cần thiết
└── README.md               # Tài liệu hướng dẫn

🛠 Yêu cầu Hệ thống (Requirements)
Ngôn ngữ: Python ≥ 3.9
📦 Thư viện
Xử lý dữ liệu: pandas, numpy
Học máy: scikit-learn, xgboost, joblib
Trực quan hóa: matplotlib, seaborn
💡 TIP: Cài đặt nhanh bằng lệnh:
pip install -r requirements.txt
🚀 Hướng dẫn Cài đặt và Sử dụng
1️⃣ Chuẩn bị môi trường
Clone hoặc tải project về máy, sau đó di chuyển vào thư mục dự án:
cd Ten_Thu_Muc_Project
2️⃣ Cài đặt thư viện
pip install pandas numpy matplotlib seaborn scikit-learn joblib xgboost
3️⃣ Quy trình thực hiện
Bước 1 – Tiền xử lý dữ liệu
Chạy notebook:
Notebooks/Data_preprocessing.ipynb
Chuẩn hóa dữ liệu
Chia tập Train / Test
Lưu kết quả vào thư mục Data/
Bước 2 – Huấn luyện mô hình
Chạy lần lượt các notebook trong thư mục Notebooks/:
AdaBoost
Random Forest
Gradient Boosting
OSEL
👉 Các mô hình sau khi huấn luyện được lưu tại Notebooks/models/.
Bước 3 – So sánh kết quả
python Compare_models.py
📊 Các mô hình triển khai
Dự án đánh giá trên nhiều thuật toán, tiêu biểu gồm:
AdaBoost
Tập trung cải thiện các mẫu khó phân loại.
Random Forest
Giảm overfitting bằng cách kết hợp nhiều cây quyết định.
Gradient Boosting
Tối ưu sai số thông qua quá trình Gradient Descent.
OSEL (Optimal Stacking Ensemble Learning)
Kiến trúc Stacking với Meta-learner (XGBoost) để tận dụng ưu điểm của các mô hình cơ sở.
📈 Kết quả
Sau khi chạy Compare_models.py, các biểu đồ so sánh:
Accuracy
F1-Score
Recall
sẽ được tự động lưu trong thư mục Results/.
📌 Kết quả cho thấy mô hình OSEL thường đạt hiệu suất ổn định và cao nhất trên tập dữ liệu thử nghiệm.

pip install -r requirements.txt
