import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

# IO
def load_data():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent  # nếu file trong src/
    DATA_DIR = PROJECT_ROOT / "data" / "processed"
    X = pd.read_csv(DATA_DIR / "X_train.csv")
    y = pd.read_csv(DATA_DIR / "y_train.csv").values.ravel()
    return X, y

def save_model(model, model_name, folder='models'):
    """Lưu model vào thư mục chỉ định"""
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Chuyển tên 'Logistic Regression' -> 'logistic_regression.pkl' cho chuẩn hóa
    file_name = f"{model_name.lower().replace(' ', '_')}.pkl"
    file_path = os.path.join(folder, file_name)
    
    joblib.dump(model, file_path)
    return file_path

def save_metrics(name, metrics_dict):
    try:
        with open('results/metrics.json', 'r') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    data[name] = metrics_dict
    with open('results/metrics.json', 'w') as f:
        json.dump(data, f, indent=4)

# Metric
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Chuyển sang float chuẩn Python
    return {'rmse': float(rmse), 'mae': float(mae)}

# OOF (Out-of-Fold) cho Stacking
def get_oof_preds(model, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict(X_val)
        
    return oof_preds


def save_figure(fig, filename):
    """
    Lưu biểu đồ (matplotlib figure) vào thư mục results/figures/.
    Tạo thư mục nếu nó chưa tồn tại.
    """
    # Đường dẫn tuyệt đối đến thư mục figures
    FIGURES_PATH = Path("results/figures")
    
    # Đảm bảo thư mục tồn tại (tạo nếu chưa có)
    FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    
    # Lưu hình ảnh
    fig.savefig(FIGURES_PATH / filename, bbox_inches='tight')
    print(f"Hình ảnh đã lưu thành công tại: {FIGURES_PATH / filename}")
