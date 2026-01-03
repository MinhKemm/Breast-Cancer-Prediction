Dự đoán Ung thư vú (Breast Cancer Prediction)
📋 Giới thiệu Dự án
Dự án này tập trung vào việc xây dựng và so sánh các mô hình học máy (Machine Learning) để dự đoán khả năng mắc ung thư vú dựa trên các đặc trưng lâm sàng. Mục tiêu là tìm ra mô hình có độ chính xác cao nhất và đáng tin cậy nhất (F1-Score, Recall) để hỗ trợ chẩn đoán y tế.

Dự án triển khai nhiều thuật toán Ensemble Learning như AdaBoost, Random Forest, Gradient Boosting và đặc biệt là mô hình OSEL (Optimal Stacking Ensemble Learning).

📂 Cấu trúc Thư mục
Dự án được tổ chức theo cấu trúc sau để đảm bảo tính logic và dễ quản lý:

Plaintext
.
├── Data/                   # Chứa dữ liệu thô và dữ liệu đã qua xử lý (.npy)
├── Notebooks/              # Chứa các file thực nghiệm Jupyter Notebook
│   ├── Data_preprocessing.ipynb
│   ├── AdaBoost.ipynb
│   ├── Random_Forest.ipynb
│   ├── Gradient_boosting.ipynb
│   ├── OSEL.ipynb          # Mô hình Stacking Ensemble
│   └── models/             # Lưu trữ các mô hình đã training (.pkl)
├── Results/                # Chứa các biểu đồ so sánh kết quả (Accuracy, F1, Recall)
├── Compare_models.py       # Script Python để đánh giá tổng thể các mô hình
├── requirements.txt        # Các thư viện cần thiết
└── README.md               # Hướng dẫn dự án
🛠 Yêu cầu Hệ thống (Requirements)
Để chạy được project này, bạn cần cài đặt Python (phiên bản >= 3.9) và các thư viện sau:

Ngôn ngữ: Python

Thư viện xử lý dữ liệu: pandas, numpy

Thư viện học máy: scikit-learn, xgboost, joblib

Thư viện trực quan hóa: matplotlib, seaborn

[!TIP] Bạn có thể cài đặt nhanh bằng lệnh: pip install -r requirements.txt

🚀 Hướng dẫn Cài đặt và Sử dụng
1. Chuẩn bị môi trường

Tải thư mục dự án về máy và di chuyển vào thư mục đó:

Bash
cd [Ten_Thu_Muc_Project]
2. Cài đặt thư viện

Bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib xgboost
3. Quy trình thực hiện

Bước 1 - Tiền xử lý: Chạy file Notebooks/Data_preprocessing.ipynb để chuẩn hóa dữ liệu và chia tập Train/Test. Kết quả sẽ được lưu vào thư mục Data/.

Bước 2 - Huấn luyện: Chạy các file notebook trong Notebooks/ để huấn luyện từng mô hình. Các mô hình sau khi train sẽ được lưu tại Notebooks/models/.

Bước 3 - So sánh kết quả: Chạy script so sánh để xuất biểu đồ đánh giá:

Bash
python Compare_models.py
📊 Các mô hình triển khai
Dự án thực hiện đánh giá trên 5+ thuật toán, tiêu biểu gồm:

AdaBoost: Tập trung vào các mẫu khó phân loại.

Random Forest: Giảm thiểu overfitting bằng cách kết hợp nhiều cây quyết định.

Gradient Boosting: Tối ưu hóa sai số thông qua Gradient Descent.

OSEL (Optimal Stacking Ensemble Learning): Sử dụng kiến trúc Stacking với Meta-learner (XGBoost) để kết hợp ưu điểm của các mô hình cơ sở.

📈 Kết quả
Sau khi chạy script Compare_models.py, các biểu đồ so sánh về Accuracy, F1-Score và Recall sẽ được tự động lưu trong thư mục Results/. Dựa vào đây, thầy có thể thấy mô hình OSEL thường cho hiệu suất ổn định và cao nhất trên tập dữ liệu thử nghiệm.
