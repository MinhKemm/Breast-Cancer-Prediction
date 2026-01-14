# 🩺 Breast Cancer Prediction  
**Dự đoán Ung thư vú bằng Machine Learning**

---

## 📋 Giới thiệu Dự án
Dự án tập trung vào việc xây dựng, huấn luyện và so sánh các mô hình Machine Learning nhằm dự đoán khả năng mắc ung thư vú dựa trên các đặc trưng lâm sàng từ bộ dữ liệu Wisconsin.

---

## 📊 Dataset
Dự án sử dụng bộ dữ liệu chuẩn trong nghiên cứu y sinh để đảm bảo tính khách quan và chính xác:
- Nguồn dữ liệu: Sử dụng bộ dữ liệu Breast Cancer Wisconsin (Diagnostic) từ kho lưu trữ UCI Machine Learning Repository.
- Quy mô mẫu: Tổng cộng 569 mẫu dữ liệu, bao gồm: 
  - Lành tính (Benign): 357 mẫu (chiếm 62.7%).
  - Ác tính (Malignant): 212 mẫu (chiếm 37.3%).
- Đặc trưng lâm sàng: Mỗi mẫu được mô tả bởi 30 thuộc tính số thực rút gọn từ hình ảnh số hóa của các khối u ở vú. Các đặc tính tiêu biểu bao gồm:
  - Kích thước: Bán kính (radius), chu vi (perimeter), diện tích (area).
  - Hình dạng: Độ nhẵn (smoothness), độ gọn (compactness), độ lõm (concavity).
  - Chi tiết khác: Kết cấu (texture), các điểm lõm (concave points), tính đối xứng (symmetry) và kích thước fractal.
  - Dữ liệu thiếu: Bộ dữ liệu hoàn chỉnh, không có giá trị bị thiếu (missing values), giúp tăng độ tin cậy cho quá trình huấn luyện mô hình.
  
---

## 🎯 Mục tiêu
Mục tiêu chính:
- Tối ưu độ chính xác chẩn đoán thông qua kiến trúc Ensemble Learning
- Đặc biệt chú trọng các chỉ số F1-Score, Recall và Accuracy để giảm thiểu sai sót y tế.
- Triển khai mô hình OSEL và đánh giá hiệu suất so với các thuật toán truyền thống

---

## 🧬 Áp dụng mô hình OSEL kết hợp thuật toán di truyền
🧬 Mô hình OSEL (Optimized Stacking Ensemble Learning)
Mô hình OSEL là điểm nhấn kỹ thuật của dự án, được phát triển dựa trên nghiên cứu tối ưu hóa quá trình học tập kết hợp (Stacking).

1. Kiến trúc Stacking
- OSEL không sử dụng một bộ phân loại duy nhất mà kết hợp hai tầng dữ liệu:
  - Base-classifiers (Tầng cơ sở): Sử dụng các thuật toán mạnh mẽ như SVM, KNN, Random Forest, và Decision Tree để tạo ra các dự đoán ban đầu (meta-data).
  - Meta-classifier (Bộ siêu phân loại): Tiếp nhận kết quả từ tầng cơ sở dưới dạng ma trận đặc trưng mới để đưa ra dự đoán cuối cùng, giúp giảm thiểu sai số và độ lệch.
 
2. Tối ưu hóa bằng Thuật toán Di truyền (Genetic Algorithm)
Điểm khác biệt của OSEL là việc tích hợp Thuật toán Di truyền (GA) để tự động hóa việc lựa chọn tổ hợp các bộ phân loại cơ sở tối ưu nhất.
GA giúp tìm kiếm trong không gian các mô hình để xác định các mô hình tốt nhất cho kết quả tốt nhất thay vì chọn lựa thủ công.
Sử dụng độ chính xác làm hàm thích nghi để tinh chỉnh các tổ hợp mô hình trong "hộp trắng" của kiến trúc Stacking.

---

## 📂 Cấu trúc Thư mục
```plaintext
.
├── Data/                   # Dữ liệu thô & dữ liệu đã xử lý (.npy)
│   ├── Data_preprocessing.ipynb
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

## 🚀 Hướng dẫn Huấn luyện mô hình
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

1. Phân tích Hiệu suất
- Kết quả thu được từ quá trình chạy mã nguồn Compare_models.py cho thấy sự phân hóa rõ rệt giữa các nhóm thuật toán:
- Độ chính xác tổng thể (Accuracy):
  - Mô hình SVM dẫn đầu thực nghiệm với độ chính xác đạt 99.12%.
  - Mô hình đề xuất OSEL cùng với AdaBoost và Logistic Regression cho thấy sự ổn định cao với cùng mức 98.25%.
  - Các thuật toán đơn lẻ như KNN (93.86%) và Decision Tree (96.49%) có hiệu suất thấp hơn rõ rệt.
- Chỉ số F1 và Độ nhạy (Recall):
  - Trong bài toán chẩn đoán ung thư, Recall là chỉ số sống còn giúp giảm thiểu tỷ lệ bỏ sót bệnh nhân (False Negative).
  - Mô hình OSEL đạt mức Recall ấn tượng 98.25%, chứng minh khả năng phát hiện tế bào ác tính cực kỳ hiệu quả và tin cậy.
  - Chỉ số F1-Score của OSEL đạt 0.9824, cho thấy sự cân bằng hoàn hảo giữa độ chính xác và khả năng thu hồi, không gây ra tình trạng chẩn đoán nhầm quá mức (False Positive).

2. Đánh giá Mô hình Đề xuất OSEL
- Mô hình OSEL (Optimized Stacking Ensemble Learning) không chỉ đạt con số ấn tượng mà còn mang lại những ưu thế kỹ thuật so với các phương pháp truyền thống:
- Sức mạnh từ sự kết hợp (Stacking): Bằng cách kết hợp dự đoán từ các "chuyên gia" như SVM, Random Forest và Logistic Regression, OSEL tận dụng được thế mạnh của từng thuật toán để đưa ra quyết định cuối cùng.
- Tối ưu hóa bằng Genetic Algorithm (GA): Thay vì lựa chọn mô hình thủ công, dự án đã triển khai GA để tự động hóa việc tìm kiếm tổ hợp các bộ phân loại cơ sở (base-classifiers) tối ưu nhất, giúp mô hình thích nghi tốt hơn với dữ liệu.
- Tính ổn định cao: Dù kết quả SVM trong thực nghiệm này đạt con số cao nhất, OSEL rất có khả năng đạt tới độ chính xác cao hơn nếu được tinh chỉnh đầy đủ các siêu tham số.

## Hướng dẫn sử dụng công cụ trực quan hóa (Streamlit)
### Bước 1: Nhập lệnh trong Terminal
```plaintexts
py -m streamlit run app.py
```

Sau khi chạy, một địa chỉ local (thường là `http://localhost:8501`) sẽ hiện ra.

### Bước 2: Nhập dữ liệu chẩn đoán
Giao diện cung cấp cho bạn 2 lựa chọn nhập liệu linh hoạt:

Cách 1 - Nhập thủ công: Điền chính xác 30 chỉ số xét nghiệm tế bào vào các ô tương ứng. Cách này phù hợp khi bạn muốn kiểm tra thay đổi của từng chỉ số đơn lẻ.

Cách 2 - Nhập nhanh: Copy một dòng dữ liệu từ file CSV (ví dụ: 17.99, 10.38, 122.8, ...) và dán vào ô văn bản. Hệ thống sẽ tự động tách 30 tham số dựa trên dấu phẩy.

### Bước 3: Chọn mô hình và xem kết quả
Tại thanh bên (Sidebar), chọn một trong các mô hình đã huấn luyện (Random Forest, SVM, XGBoost, OSEL...).

### Bước 4: Nhấn nút "Chẩn đoán".

Hệ thống sẽ hiển thị kết quả:

Ác tính (M): Hiển thị màu đỏ kèm cảnh báo.

Lành tính (B): Hiển thị màu xanh an toàn.

Độ tin cậy (%): Hiển thị xác suất mô hình tin tưởng vào quyết định đó (đối với các mô hình hỗ trợ predict_proba).

## 📚 Tham khảo
Dự án được thực hiện dựa trên phương pháp nghiên cứu của bài báo:
Kumar, M., Singhal, S., Shekhar, S., Sharma, B., & Srivastava, G. (2022). "Optimized Stacking Ensemble Learning Model for Breast Cancer Detection and Classification Using Machine Learning". Sustainability, 14(21), 13998.
