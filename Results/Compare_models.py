import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, f1_score, recall_score
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

def floor(x):
    return int(x) if x >= 0 else int(x) - 1

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
            recall = recall_score(y_test, y_pred, average='weighted')

            results.append({
                'Model': name, 
                'Accuracy': acc, 
                'F1-Score': f1, 
                'Recall': recall
            })
        except Exception as e:
            print(f"Lỗi khi load {name}: {e}")

    df_results = pd.DataFrame(results)

    # Danh sách các metric
    metrics = {
        'Accuracy': 'Độ chính xác (Accuracy)',
        'F1-Score': 'Chỉ số F1 (F1-Score)',
        'Recall': 'Độ nhạy (Recall)'
    }

    sns.set_style("whitegrid")

    for metric_col, metric_name in metrics.items():
        # Sắp xếp tăng dần theo metric 
        df_sorted = df_results.sort_values(by=metric_col, ascending=True).reset_index(drop=True)
        
        # Thiết lập màu sắc 
        colors = ['skyblue' if model != 'OSEL' else 'gold' for model in df_sorted['Model']]

        plt.figure(figsize=(15, 9))
        ax = sns.barplot(x='Model', y=metric_col, data=df_sorted, palette=colors)
        
        # Tùy chỉnh trục Y động theo giá trị của từng metric
        min_val = df_sorted[metric_col].min()
        y_bottom = floor(min_val * 100 - 1) / 100 if min_val > 0.1 else 0
        plt.ylim(y_bottom, 1.0) 
        
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
        legend_elements = [
            Patch(facecolor='skyblue', label='Thuật toán cơ bản'),
            Patch(facecolor='gold', label='Thuật toán OSEL (Đề xuất)')
        ]
        plt.legend(handles=legend_elements, loc='upper left', fontsize=12, frameon=True)

        plt.title(f'So sánh chi tiết {metric_col} giữa các mô hình', fontsize=20, pad=25)
        plt.ylabel(metric_name, fontsize=14)
        plt.xlabel('Mô hình', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=11)

        # Lưu kết quả với tên file tương ứng
        file_name = f'comparison_{metric_col.lower()}.png'
        output_path = os.path.join(output_dir, file_name)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close() 
        
        print(f"--- Đã lưu biểu đồ {metric_col} tại '{output_path}' ---")

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