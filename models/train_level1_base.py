import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.utils.class_weight import compute_sample_weight
from scipy.optimize import minimize
from functools import partial
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.tree import plot_tree
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
    parser = argparse.ArgumentParser(description="Train Level 1 Base Models for Stacking.")
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/regression_stacking_level1', help='Directory to save outputs')
    parser.add_argument('-d', '--data_path', type=str, default='data/train_cleaned_imputed.csv', help='Path to input CSV')
    
    # Arguments for flexibility
    parser.add_argument('--models', nargs='+', type=str, 
                        default=['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'HistGradientBoosting'],
                        choices=['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'HistGradientBoosting'],
                        help='List of models to train (default: all)')
    
    return parser.parse_args()

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    if 'sii' in df.columns:
        df['sii'] = df['sii'].astype(float)
    return df

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
    # Use provided coefficients or default
    rounder = OptimizedRounder()
    return rounder.predict(y_pred, coef)

def evaluate_model(y_true, y_pred, coef=[0.5, 1.5, 2.5]):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    y_pred_rounded = optimize_predictions(y_pred, coef)
    y_true_int = y_true.astype(int)
    
    acc = accuracy_score(y_true_int, y_pred_rounded)
    qwk = cohen_kappa_score(y_true_int, y_pred_rounded, weights='quadratic')
    
    return {'RMSE': rmse, 'Accuracy': acc, 'QWK': qwk}

def plot_confusion_matrix_heatmap(y_true, y_pred, model_name, output_dir, coef=[0.5, 1.5, 2.5]):
    y_pred_rounded = optimize_predictions(y_pred, coef)
    y_true_int = y_true.astype(int)
    
    cm = confusion_matrix(y_true_int, y_pred_rounded)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Rounded): {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}.png'))
    plt.close()

def plot_feature_importance(model, model_name, feature_names, output_dir, X_val=None, y_val=None):
    try:
        importances = None
        if model_name == 'HistGradientBoosting':
            if X_val is not None and y_val is not None:
                result = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42, n_jobs=4, scoring='neg_mean_squared_error')
                importances = result.importances_mean
            else:
                return
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'feature_importance'): # LightGBM
            importances = model.feature_importance()
        
        if importances is not None:
            indices = np.argsort(importances)[::-1][:20] # Top 20
            plt.figure(figsize=(10, 8))
            plt.title(f"Feature Importances: {model_name} (Top 20)")
            plt.barh(range(len(indices)), importances[indices], align="center")
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'feature_importance_{model_name}.png'))
            plt.close()
    except Exception as e:
        print(f"Could not plot feature importance for {model_name}: {e}")

def plot_learning_curve(history, model_name, output_dir):
    try:
        plt.figure(figsize=(10, 6))
        values = []
        label = 'Validation Metric'
        
        if isinstance(history, (list, np.ndarray)):
            values = history
        elif isinstance(history, dict):
            # Helper to find list
            def find_metric_list(d):
                for k, v in d.items():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
                        if 'loss' in k.lower() or 'rmse' in k.lower(): return v
                for k, v in d.items(): 
                    if isinstance(v, dict): 
                        res = find_metric_list(v)
                        if res: return res
                return None
            values = find_metric_list(history) or []
            
        if len(values) > 0:
            plt.plot(range(len(values)), values, label='Validation Score')
            plt.title(f'{model_name} Learning Curve')
            plt.xlabel('Iterations')
            plt.ylabel('Metric')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'learning_curve_{model_name}.png'))
        plt.close()
    except Exception as e:
        print(f"Could not plot learning curve for {model_name}: {e}")

def plot_first_tree(model, model_name, feature_names, output_dir):
    try:
        if model_name == 'RandomForest':
            plt.figure(figsize=(20, 10))
            plot_tree(model.estimators_[0], feature_names=feature_names, max_depth=3, filled=True, proportion=True, rounded=True)
            plt.title(f"{model_name} - First Tree (Depth 3)")
            plt.savefig(os.path.join(output_dir, f'tree_viz_{model_name}.png'))
            plt.close()
        elif model_name == 'XGBoost':
            plt.figure(figsize=(20, 10))
            xgb.plot_tree(model, num_trees=0, rankdir='LR')
            plt.title(f"{model_name} - First Tree")
            plt.savefig(os.path.join(output_dir, f'tree_viz_{model_name}.png'))
            plt.close()
        elif model_name == 'LightGBM':
            plt.figure(figsize=(20, 10))
            lgb.plot_tree(model, tree_index=0, figsize=(20, 10), show_info=['split_gain'])
            plt.title(f"{model_name} - First Tree")
            plt.savefig(os.path.join(output_dir, f'tree_viz_{model_name}.png'))
            plt.close()
    except Exception as e:
        print(f"Could not plot tree for {model_name}: {e}")

def get_base_model(name):
    if name == 'XGBoost':
        return xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='rmse', n_jobs=4, early_stopping_rounds=50
        )
    elif name == 'LightGBM':
        return lgb.LGBMRegressor(
             objective='regression', n_estimators=500, learning_rate=0.05,
             num_leaves=20, max_depth=6, min_child_samples=30,
             random_state=42, verbosity=-1
        )
    elif name == 'CatBoost':
        return CatBoostRegressor(
            loss_function='RMSE', iterations=1000, learning_rate=0.05,
            depth=6, l2_leaf_reg=5, random_seed=42,
            verbose=False, allow_writing_files=False, early_stopping_rounds=50
        )
    elif name == 'RandomForest':
        return RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=4
        )
    elif name == 'HistGradientBoosting':
        return HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=500, max_depth=6,
            random_state=42, early_stopping=True
        )
    else:
        raise ValueError(f"Unknown model: {name}")

def train_base_models(X, y, output_dir, models_to_train=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_stratify = optimize_predictions(y)
    feature_names = X.columns.tolist()
    
    # Logging
    log_file = open(os.path.join(output_dir, 'training_log_level1.txt'), 'w')
    
    if models_to_train is None:
        base_model_names = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'HistGradientBoosting']
    else:
        base_model_names = models_to_train
        
    model_results = {name: {'RMSE': [], 'Accuracy': [], 'QWK': []} for name in base_model_names}
    
    # Validation scores for weighted averaging
    cv_scores = {name: [] for name in base_model_names}
    
    # Store OOF predictions
    oof_preds = pd.DataFrame(0.0, index=X.index, columns=base_model_names)
    
    print(f"Starting Level 1 Base Model Training...")
    log_file.write(f"Starting Level 1 Base Model Training...\n")
    
    fold = 1
    for train_index, val_index in skf.split(X, y_stratify):
        msg = f"\n=== Fold {fold}/5 ==="
        print(msg)
        log_file.write(msg + "\n")
        
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Calculate soft sample weights (sqrt of balanced)
        train_weights = compute_sample_weight(
            class_weight='balanced',
            y=np.round(y_train).astype(int)
        )
        train_weights = np.sqrt(train_weights) # Soften the weights

        
        for name in base_model_names:
            print(f"  Training {name}...")
            model = get_base_model(name)
            
            # Fit
            if name == 'XGBoost':
                model.fit(X_train, y_train, sample_weight=train_weights, eval_set=[(X_val, y_val)], verbose=False)
            elif name == 'LightGBM':
                callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)]
                model.fit(X_train, y_train, sample_weight=train_weights, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=callbacks)
            elif name == 'CatBoost':
                model.fit(X_train, y_train, sample_weight=train_weights, eval_set=(X_val, y_val))
            else:
                model.fit(X_train, y_train, sample_weight=train_weights)
            
            # Save Base Model
            joblib.dump(model, os.path.join(output_dir, f'{name}_fold{fold}.pkl'))
            
            # Predict OOF
            if name == 'CatBoost':
                pred = model.predict(X_val).flatten()
            else:
                pred = model.predict(X_val)
            
            oof_preds.loc[val_index, name] = pred
            
            # Evaluate Individual Model
            # Fit rounder on this fold for evaluation
            folder_rounder = OptimizedRounder()
            folder_rounder.fit(pred, y_val)
            scores = evaluate_model(y_val, pred, folder_rounder.coef_)
            for k, v in scores.items(): 
                model_results[name][k].append(v)
            
            # Save QWK score for weighted averaging (Confidence Score)
            cv_scores[name].append(scores['QWK'])
            
            score_str = f"    {name} Fold {fold}: RMSE={scores['RMSE']:.4f}, QWK={scores['QWK']:.4f}"
            print(score_str)
            log_file.write(score_str + "\n")
            
            # Visualizations (Only for Fold 1)
            if fold == 1:
                plot_confusion_matrix_heatmap(y_val, pred, name, output_dir, folder_rounder.coef_)
                plot_feature_importance(model, name, feature_names, output_dir, X_val, y_val)
                plot_first_tree(model, name, feature_names, output_dir)
                
                # Learning Curves
                if name == 'XGBoost':
                    plot_learning_curve(model.evals_result(), name, output_dir)
                elif name == 'LightGBM':
                    if hasattr(model, 'evals_result_'):
                        plot_learning_curve(model.evals_result_, name, output_dir)
                elif name == 'CatBoost':
                    plot_learning_curve(model.get_evals_result(), name, output_dir)
                elif name == 'HistGradientBoosting':
                    if hasattr(model, 'validation_scores_'):
                        plot_learning_curve(model.validation_scores_, name, output_dir)

        fold += 1
        
    print("\nLevel 1 Training Complete.")
    log_file.write("\nLevel 1 Training Complete.\n")
    
    # Save CV Scores for Inference
    with open(os.path.join(output_dir, 'cv_scores.json'), 'w') as f:
        json.dump(cv_scores, f, indent=4)
    print(f"Saved CV scores to {os.path.join(output_dir, 'cv_scores.json')}")

    # Save OOF predictions with ID and Target (sii)
    oof_preds['sii'] = y 
    oof_preds.to_csv(os.path.join(output_dir, 'oof_predictions.csv'), index=False)
    print(f"Saved OOF predictions (with target) to {os.path.join(output_dir, 'oof_predictions.csv')}")

    # Print Base Model Summaries
    print("\n=== Base Model Performance (CV Average) ===")
    log_file.write("\n=== Base Model Performance (CV Average) ===\n")
    for name, metrics in model_results.items():
        rmse_mean = np.mean(metrics['RMSE'])
        acc_mean = np.mean(metrics['Accuracy'])
        qwk_mean = np.mean(metrics['QWK'])
        res_str = f"{name:<20} | RMSE: {rmse_mean:.4f} | Accuracy: {acc_mean:.4f} | QWK: {qwk_mean:.4f}"
        print(res_str)
        log_file.write(res_str + "\n")
    
    log_file.close()
    return base_model_names

def main():
    args = parse_args()
    
    if args.data_path:
        data_path = args.data_path
    else:
        # Fallback to default
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, '../data/train_cleaned_imputed.csv')
        
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return
        
    df = load_data(data_path)
    df = preprocess_data(df)
    
    X = df.drop(columns=['sii'])
    y = df['sii']
    
    train_base_models(X, y, args.output_dir, models_to_train=args.models)

if __name__ == "__main__":
    main()
