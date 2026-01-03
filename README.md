# 🩺 Breast Cancer Prediction  
**Dự đoán Ung thư vú bằng Machine Learning**

---

## 📋 Giới thiệu Dự án
Dự án tập trung vào việc xây dựng, huấn luyện và so sánh các mô hình **Machine Learning** nhằm dự đoán khả năng mắc **ung thư vú** dựa trên các đặc trưng lâm sàng.

Mục tiêu chính:
- Tối ưu **độ chính xác chẩn đoán**
- Đặc biệt chú trọng các chỉ số **F1-Score** và **Recall**
- Hỗ trợ quá trình ra quyết định trong y tế

Các thuật toán **Ensemble Learning** được triển khai:
- **AdaBoost**
- **Random Forest**
- **Gradient Boosting**
- **OSEL (Optimal Stacking Ensemble Learning)** – mô hình Stacking tối ưu

---

## 📂 Cấu trúc Thư mục
```plaintext
.
├── Data/                   # Dữ liệu thô & dữ liệu đã xử lý (.npy)
├── Notebooks/              # Các Jupyter Notebook thực nghiệm
│   ├── Data_preprocessing.ipynb
│   ├── AdaBoost.ipynb
│   ├── Random_Forest.ipynb
│   ├── Gradient_boosting.ipynb
│   ├── OSEL.ipynb          # Stacking Ensemble (OSEL)
│   └── models/             # Các mô hình đã huấn luyện (.pkl)
├── Results/                # Biểu đồ so sánh (Accuracy, F1, Recall)
├── Compare_models.py       # Script đánh giá & so sánh mô hình
├── requirements.txt        # Danh sách thư viện cần thiết
└── README.md               # Tài liệu hướng dẫn
```

---

## 🛠 Yêu cầu Hệ thống (Requirements)

- **Ngôn ngữ:** Python ≥ 3.9

### 📦 Thư viện
- **Xử lý dữ liệu:** `pandas`, `numpy`
- **Học máy:** `scikit-learn`, `xgboost`, `joblib`
- **Trực quan hóa:** `matplotlib`, `seaborn`

💡 **Cài đặt nhanh (khuyến nghị):**
```bash
pip install -r requirements.txt
```

---

## 🚀 Hướng dẫn Cài đặt và Sử dụng

### 1️⃣ Chuẩn bị môi trường
Clone hoặc tải project về máy, sau đó di chuyển vào thư mục dự án:
```bash
cd Ten_Thu_Muc_Project
```

### 2️⃣ Cài đặt thư viện (thủ công nếu cần)
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib xgboost
```

### 3️⃣ Quy trình thực hiện

#### Bước 1 – Tiền xử lý dữ liệu
Chạy notebook:
```plaintext
Notebooks/Data_preprocessing.ipynb
```
- Chuẩn hóa dữ liệu
- Chia tập Train / Test
- Lưu kết quả vào thư mục `Data/`

#### Bước 2 – Huấn luyện mô hình
Chạy lần lượt các notebook trong thư mục `Notebooks/`:
- AdaBoost
- Random Forest
- Gradient Boosting
- OSEL

👉 Các mô hình sau khi huấn luyện được lưu tại:
```plaintext
Notebooks/models/
```

#### Bước 3 – So sánh kết quả
```bash
python Compare_models.py
```

---

## 📊 Các mô hình triển khai

### AdaBoost
- Tập trung cải thiện các mẫu khó phân loại

### Random Forest
- Giảm overfitting bằng cách kết hợp nhiều cây quyết định

### Gradient Boosting
- Tối ưu sai số thông qua quá trình Gradient Descent

### OSEL (Optimal Stacking Ensemble Learning)
- Kiến trúc Stacking với **Meta-learner (XGBoost)**
- Tận dụng ưu điểm của các mô hình cơ sở

---

## 📈 Kết quả
Sau khi chạy `Compare_models.py`, các biểu đồ:
- **Accuracy**
- **F1-Score**
- **Recall**

sẽ được tự động lưu trong thư mục:
```plaintext
Results/
```

📌 Kết quả thực nghiệm cho thấy mô hình **OSEL** thường đạt hiệu suất **ổn định và cao nhất** trên tập dữ liệu thử nghiệm.
