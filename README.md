# 🩺 Breast Cancer Prediction  
**Dự đoán Ung thư vú bằng Machine Learning**

---

## 📋 Giới thiệu Dự án
Dự án tập trung vào việc xây dựng, huấn luyện và so sánh các mô hình Machine Learning nhằm dự đoán khả năng mắc ung thư vú dựa trên các đặc trưng lâm sàng từ bộ dữ liệu Wisconsin.

Mục tiêu chính:
- Tối ưu độ chính xác chẩn đoán thông qua kiến trúc Ensemble Learning
- Đặc biệt chú trọng các chỉ số F1-Score, Recall và Accuracy để giảm thiểu sai sót y tế.
- Triển khai mô hình OSEL và đánh giá hiệu suất so với các thuật toán truyền thống

- 🧬 Mô hình OSEL (Optimized Stacking Ensemble Learning)
Mô hình OSEL là điểm nhấn kỹ thuật của dự án, được phát triển dựa trên nghiên cứu tối ưu hóa quá trình học tập kết hợp (Stacking).

1. Kiến trúc Stacking
- OSEL không sử dụng một bộ phân loại duy nhất mà kết hợp hai tầng dữ liệu:
  - Base-classifiers (Tầng cơ sở): Sử dụng các thuật toán mạnh mẽ như SVM, KNN, Random Forest, và Decision Tree để tạo ra các dự đoán ban đầu (meta-data).
  - Meta-classifier (Bộ siêu phân loại): Tiếp nhận kết quả từ tầng cơ sở dưới dạng ma trận đặc trưng mới để đưa ra dự đoán cuối cùng, giúp giảm thiểu sai số và độ lệch.
  - 
2. Tối ưu hóa bằng Thuật toán Di truyền (Genetic Algorithm)
Điểm khác biệt của OSEL là việc tích hợp Thuật toán Di truyền (GA) để tự động hóa việc lựa chọn tổ hợp các bộ phân loại cơ sở tối ưu nhất.
GA giúp tìm kiếm trong không gian các mô hình để xác định các mô hình tốt nhất cho kết quả tốt nhất thay vì chọn lựa thủ công.
Sử dụng độ chính xác làm hàm thích nghi để tinh chỉnh các tổ hợp mô hình trong "hộp trắng" của kiến trúc Stacking.

---

## 📂 Cấu trúc Thư mục
```plaintext
.
├── Data/                   # Dữ liệu thô & dữ liệu đã xử lý (.npy)
├── Notebooks/              # Các Jupyter Notebook thực nghiệm
│   ├── models/             # Lưu trữ các mô hình đã huấn luyện (.pkl)
│   │   ├── adaboost.pkl
│   │   ├── decision_tree.pkl
│   │   ├── gradient_boosting.pkl
│   │   ├── knn.pkl
│   │   ├── logistic_regression.pkl
│   │   ├── osel.pkl
│   │   ├── random_forest.pkl
│   │   ├── svm.pkl
│   │   └── xgboost.pkl
│   ├── AdaBoost.ipynb
│   ├── Decision_tree.ipynb
│   ├── Data_preprocessing.ipynb
│   ├── Gradient_boosting.ipynb
│   ├── KNN.ipynb
│   ├── Logistic_Regression.ipynb
│   ├── OSEL.ipynb          
│   ├── Random_Forest.ipynb
│   ├── SVM.ipynb
│   └── XGBoost.ipynb
├── Results/                # Kết quả đánh giá mô hình
│   ├── Compare_models.py   # So sánh tổng thể các mô hình
│   ├── comparison_accuracy.png
│   ├── comparison_f1-score.png
│   └── comparison_recall.png
├── src/                    # Mã nguồn hỗ trợ
│   ├── models.py           # Định nghĩa cấu trúc mô hình
│   └── utils.py            # Các hàm bổ trợ (xử lý dữ liệu, vẽ biểu đồ)
├── requirements.txt        # Danh sách thư viện
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
Để đảm bảo dự án chạy ổn định và không xung đột thư viện, hãy làm theo các bước sau:
### 1️⃣ Khởi tạo Môi trường ảo (Khuyến nghị)
Sử dụng môi trường ảo giúp quản lý thư viện độc lập:
```bash
# Tạo môi trường ảo tên là 'env'
python -m venv venv

# Kích hoạt môi trường (Windows)
.\venv\Scripts\activate

# Kích hoạt môi trường (macOS/Linux)
source venv/bin/activate
```

### 2️⃣ Cài đặt Thư viện
Cài đặt tất cả các phụ thuộc chỉ với một lệnh:
```bash
pip install -r requirements.txt
```
Các thư viện chính bao gồm: pandas, numpy, scikit-learn, xgboost, catboost, matplotlib, seaborn, joblib

### 3️⃣ Quy trình thực hiện

#### Bước 1 – Tiền xử lý dữ liệu
Chạy notebook:
```plaintext
Data/Data_preprocessing.ipynb
```
- Chuẩn hóa dữ liệu
- Chia tập Train / Test
- Lưu kết quả vào thư mục `Data/`

#### Bước 2 – Huấn luyện mô hình và tinh chỉnh tham số
Chạy lần lượt các notebook trong thư mục `Notebooks/`:
- AdaBoost
- Decision Tree
- Gradient Boosting
- KNN
- Logistic Regression
- OSEL
- Random Forest
- SVM
- XGBoost

👉 Các mô hình sau khi huấn luyện và tinh chỉnh tham số được lưu tại:
```plaintext
Notebooks/models/
```

#### Bước 3 – So sánh kết quả
Chạy file Compare_models.py trong thư mục Results để hiển thị các kết quả so sánh hoặc chạy trong Terminals như sau: 
```bash
python Compare_models.py
```

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
