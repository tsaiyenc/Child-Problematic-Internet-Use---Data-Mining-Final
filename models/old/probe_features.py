
import joblib
import os
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

model_dir = 'outputs/regression_ver1'
models = {
    'XGBoost': 'XGBoost_fold1.pkl',
    'LightGBM': 'LightGBM_fold1.pkl',
    'CatBoost': 'CatBoost_fold1.pkl',
    'RandomForest': 'RandomForest_fold1.pkl',
    'HistGradientBoosting': 'HistGradientBoosting_fold1.pkl'
}

for name, filename in models.items():
    path = os.path.join(model_dir, filename)
    if os.path.exists(path):
        try:
            model = joblib.load(path)
            print(f"--- {name} ---")
            if hasattr(model, 'feature_names_in_'):
                print(f"Has feature_names_in_: {model.feature_names_in_[:5]}...")
            elif hasattr(model, 'feature_name_'):
                print(f"Has feature_name_: {model.feature_name_[:5]}...")
            elif hasattr(model, 'feature_names_'):
                print(f"Has feature_names_: {model.feature_names_[:5]}...")
            else:
                print("No obvious feature names attribute.")
        except Exception as e:
            print(f"Error loading {name}: {e}")
    else:
        print(f"{name} file not found at {path}")
