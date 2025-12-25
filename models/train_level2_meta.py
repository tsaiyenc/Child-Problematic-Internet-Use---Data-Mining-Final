import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_squared_error
from sklearn.inspection import permutation_importance
from scipy.optimize import minimize
from functools import partial
import os
import argparse
import joblib
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="Train Level 2 Meta Learner for Stacking.")
    parser.add_argument('-i', '--input_dir', type=str, default='outputs/regression_stacking_level1', help='Directory containing oof_predictions.csv')
    parser.add_argument('-o', '--output_dir', type=str, default=None, help='Directory to save meta-model outputs (default: input_dir)')
    parser.add_argument('--meta-model', type=str, default='XGBoost', 
                        choices=['XGBoost', 'LightGBM', 'CatBoost', 'HistGradientBoosting', 'RandomForest', 'Ridge', 'Lasso', 'LinearRegression'],
                        help='Meta-learner model architecture')
    return parser.parse_args()

class OptimizedRounder:
    def __init__(self):
        self.coef_ = [0.5, 1.5, 2.5]

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            else:
                X_p[i] = 3

        ll = cohen_kappa_score(y, X_p, weights='quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5]
        self.coef_ = minimize(loss_partial, initial_coef, method='nelder-mead').x

    def predict(self, X, coef=None):
        if coef is None:
            coef = self.coef_
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            else:
                X_p[i] = 3
        return X_p.astype(int)

def optimize_predictions(y_pred, coef=[0.5, 1.5, 2.5]):
    rounder = OptimizedRounder()
    return rounder.predict(y_pred, coef)

def evaluate_model(y_true, y_pred, coef=[0.5, 1.5, 2.5]):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    y_pred_rounded = optimize_predictions(y_pred, coef)
    y_true_int = y_true.astype(int)
    acc = accuracy_score(y_true_int, y_pred_rounded)
    qwk = cohen_kappa_score(y_true_int, y_pred_rounded, weights='quadratic')
    return {'RMSE': rmse, 'Accuracy': acc, 'QWK': qwk}

def get_meta_model(name):
    if name == 'XGBoost':
        return xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100, learning_rate=0.05, max_depth=3,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=4
        )
    elif name == 'LightGBM':
        return lgb.LGBMRegressor(
             objective='regression', n_estimators=100, learning_rate=0.05,
             num_leaves=10, max_depth=3, random_state=42, verbosity=-1
        )
    elif name == 'CatBoost':
        return CatBoostRegressor(
            loss_function='RMSE', iterations=200, learning_rate=0.05,
            depth=4, random_seed=42, verbose=False, allow_writing_files=False
        )
    elif name == 'HistGradientBoosting':
        return HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=100, max_depth=3, 
            l2_regularization=1.0, random_state=42
        )
    elif name == 'RandomForest':
        return RandomForestRegressor(
            n_estimators=100, max_depth=5, random_state=42, n_jobs=4
        )
    elif name == 'Ridge':
        return Ridge(alpha=1.0)
    elif name == 'Lasso':
        return Lasso(alpha=0.01)
    elif name == 'LinearRegression':
        return LinearRegression()
    else:
        raise ValueError(f"Unknown meta-model: {name}")

def plot_feature_importance(model, model_name, feature_names, output_dir):
    try:
        importances = None
        if model_name in ['HistGradientBoosting', 'Ridge', 'Lasso', 'LinearRegression']:
             # For these, we might not get simple feature importances easily suitable for this plot function 
             # without permutation importance on a validation set. 
             # But for simplicity, we skip or handle basic cases.
             if hasattr(model, 'coef_'):
                 importances = np.abs(model.coef_)
             else:
                 return # Skip for HGB without validation data handy here
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'feature_importance'): # LightGBM
            importances = model.feature_importance()
        
        if importances is not None:
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 6))
            plt.title(f"Feature Importances: {model_name}")
            plt.barh(range(len(indices)), importances[indices], align="center")
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'feature_importance_meta_{model_name}.png'))
            plt.close()
    except Exception as e:
        print(f"Could not plot feature importance: {e}")

def train_meta_learner(input_dir, output_dir, meta_model_name):
    oof_path = os.path.join(input_dir, 'oof_predictions.csv')
    if not os.path.exists(oof_path):
        print(f"Error: {oof_path} not found. Run Level 1 training first.")
        return

    print(f"Loading OOF predictions from {oof_path}...")
    oof_df = pd.read_csv(oof_path)
    
    if 'sii' not in oof_df.columns:
        print("Error: 'sii' column not found in OOF predictions. Please re-run Level 1 training.")
        return
        
    y = oof_df['sii']
    X = oof_df.drop(columns=['sii'])
    
    # Feature Engineering
    print("Generating Meta-Features (Mean, Std, Min, Max)...")
    X['mean'] = X.mean(axis=1)
    X['std'] = X.std(axis=1)
    X['min'] = X.min(axis=1)
    X['max'] = X.max(axis=1)
    
    print(f"Training Meta Learner: {meta_model_name}...")
    model = get_meta_model(meta_model_name)
    model.fit(X, y)
    
    # Save Model
    joblib.dump(model, os.path.join(output_dir, f'meta_model_{meta_model_name}.pkl'))
    
    # Evaluate on Training Data (OOF) - Note: This is somewhat optimistic but checks fit
    pred = model.predict(X)
    
    # Optimize Rounder
    rounder = OptimizedRounder()
    rounder.fit(pred, y)
    joblib.dump(rounder, os.path.join(output_dir, f'meta_rounder_{meta_model_name}.pkl'))
    
    scores = evaluate_model(y, pred, rounder.coef_)
    
    print("\n=== Meta Model Performance (on OOF features) ===")
    print(f"Rounder Coefficients: {rounder.coef_}")
    print(f"RMSE: {scores['RMSE']:.4f}")
    print(f"Accuracy: {scores['Accuracy']:.4f}")
    print(f"QWK: {scores['QWK']:.4f}")
    
    # Save Scores
    with open(os.path.join(output_dir, f'meta_scores_{meta_model_name}.json'), 'w') as f:
        json.dump(scores, f, indent=4)
        
    plot_feature_importance(model, meta_model_name, X.columns.tolist(), output_dir)

def main():
    args = parse_args()
    output_dir = args.output_dir if args.output_dir else args.input_dir
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    train_meta_learner(args.input_dir, output_dir, args.meta_model)

if __name__ == "__main__":
    main()
