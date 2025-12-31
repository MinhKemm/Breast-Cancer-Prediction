import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.ticker as ticker

def compare_models(model_paths):

    # Tạo thư mục Results
    output_dir = 'Results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Tải dữ liệu 
    print("--- Đang tải dữ liệu ---")
    X_test = np.load('/Users/binhminh/Documents/Intro ML code /Data/X_test.npy')
    y_test = np.load('/Users/binhminh/Documents/Intro ML code /Data/y_test.npy')
    
    results = []

    # Đánh giá các mô hình
    print("--- Đang đánh giá các mô hình ---")
    for name, path in model_paths.items():
        try:
            model = joblib.load(path)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            results.append({'Model': name, 'Accuracy': acc, 'F1-Score': f1})
        except Exception as e:
            print(f"Lỗi khi load {name}: {e}")

    # Sắp xếp tăng dần
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by='Accuracy', ascending=True).reset_index(drop=True)
    
    # Thiết lập màu sắc
    colors = ['skyblue' if model != 'OSEL' else 'gold' for model in df_results['Model']]

    # Vẽ biểu đồ với thang đo chi tiết
    plt.figure(figsize=(15, 9))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(x='Model', y='Accuracy', data=df_results, palette=colors)
    
    # Lấy giá trị thấp nhất trừ đi một khoảng nhỏ để làm mốc bắt đầu
    min_acc = df_results['Accuracy'].min()
    y_bottom = floor(min_acc * 100 - 1) / 100 if min_acc > 0.1 else 0 # Ví dụ: 0.92
    plt.ylim(y_bottom, 1.0) 
    
    # Chia nhỏ các vạch trên trục Y (cách nhau 0.01)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))

    # Thêm số liệu trên đầu cột
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.4f'), 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 12), 
                    textcoords = 'offset points',
                    fontsize=12, fontweight='bold', color='black')

    # Chú thích
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', label='Thuật toán cơ bản'),
        Patch(facecolor='gold', label='Thuật toán OSEL (Đề xuất)')
    ]
    plt.legend(handles=legend_elements, loc='upper left', fontsize=12, frameon=True)

    plt.title('So sánh chi tiết Accuracy giữa các mô hình', fontsize=20, pad=25)
    plt.ylabel('Độ chính xác (Accuracy)', fontsize=14)
    plt.xlabel('Mô hình', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=11)

    # Lưu kết quả
    output_path = os.path.join(output_dir, 'models_comparison.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    
    print(f"\n--- Hoàn tất! Biểu đồ chi tiết đã được lưu tại '{output_path}' ---")

def floor(x):
    return int(x) if x >= 0 else int(x) - 1

if __name__ == "__main__":
    MODELS = {
        'AdaBoost': '/Users/binhminh/Documents/Intro ML code /Notebooks/models/adaboost.pkl',
        'Decision Tree' : '/Users/binhminh/Documents/Intro ML code /Notebooks/models/decision_tree.pkl',
        'Gradient Boosting' : '/Users/binhminh/Documents/Intro ML code /Notebooks/models/gradient_boosting.pkl',
        'KNN': '/Users/binhminh/Documents/Intro ML code /Notebooks/models/knn.pkl',
        'Logistic Regression': '/Users/binhminh/Documents/Intro ML code /Notebooks/models/logistic_regression.pkl',
        'Random Forest': '/Users/binhminh/Documents/Intro ML code /Notebooks/models/random_forest.pkl',
        'SVM' : '/Users/binhminh/Documents/Intro ML code /Notebooks/models/svm.pkl',
        'XGBoost': '/Users/binhminh/Documents/Intro ML code /Notebooks/models/xgboost.pkl',
        'OSEL' : '/Users/binhminh/Documents/Intro ML code /Notebooks/models/osel.pkl'
    }
    compare_models(MODELS)