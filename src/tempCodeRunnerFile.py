from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb

def get_model(model_name, params=None):
    if params is None:
        params = {}
        
    if model_name == 'linear':
        return LinearRegression(**params)
    elif model_name == 'ridge':
        return Ridge(**params)
    elif model_name == 'random_forest':
        return RandomForestRegressor(n_jobs=-1, random_state=42, **params)
    elif model_name == 'xgboost':
        return xgb.XGBRegressor(n_jobs=-1, random_state=42, **params)
    elif model_name == 'lightgbm':
        return lgb.LGBMRegressor(n_jobs=-1, random_state=42, verbose=-1, **params)
    elif model_name == 'Logistic Regression':
        return LogisticRegression(**params)
    else:
        raise ValueError(f"Model {model_name} not found")