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
    parser = argparse.ArgumentParser(description="Train stacking ensemble model.")
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/regression_stacking_SoftBalancedWeight', help='Directory to save outputs')
    parser.add_argument('-d', '--data_path', type=str, default='data/train_cleaned_imputed.csv', help='Path to input CSV')
    parser.add_argument('-t', '--test_path', type=str, default='data/test_cleaned_imputed.csv', help='Path to test CSV')
    
    # New arguments for flexibility
    parser.add_argument('--models', nargs='+', type=str, 
                        default=['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'HistGradientBoosting'],
                        choices=['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'HistGradientBoosting'],
                        help='List of models to train (default: all)')
    parser.add_argument('--no-ensemble', dest='ensemble', action='store_false', default=True, help='Skip Level 2 Ensemble')
    
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

def train_stacking_cv(X, y, output_dir, models_to_train=None, run_ensemble=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_stratify = optimize_predictions(y)
    feature_names = X.columns.tolist()
    
    # Logging
    log_file = open(os.path.join(output_dir, 'training_log.txt'), 'w')
    
    if models_to_train is None:
        base_model_names = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'HistGradientBoosting']
    else:
        base_model_names = models_to_train
        
    model_results = {name: {'RMSE': [], 'Accuracy': [], 'QWK': []} for name in base_model_names}
    
    # Validation scores for weighted averaging
    cv_scores = {name: [] for name in base_model_names}
    
    # Store OOF predictions
    oof_preds = pd.DataFrame(0.0, index=X.index, columns=base_model_names)
    
    print(f"Starting Level 1 Stacking CV...")
    log_file.write(f"Starting Level 1 Stacking CV...\n")
    
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

    # Save OOF predictions
    oof_preds.to_csv(os.path.join(output_dir, 'oof_predictions.csv'), index=False)

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
        print(res_str)
        log_file.write(res_str + "\n")
    
    # Global Optimized Rounder for Final Stack
    if run_ensemble:
        print("\nFitting Global Optimized Rounder on OOF...")
        final_coefficients = {}
        for name in base_model_names:
            rounder = OptimizedRounder()
            rounder.fit(oof_preds[name], y)
            final_coefficients[name] = rounder.coef_
            print(f"  {name} Coefficients: {rounder.coef_}")
            joblib.dump(rounder, os.path.join(output_dir, f'{name}_rounder.pkl'))

        # Feature Engineering for Meta Learner: Add Row-wise Statistics
        print("Generating Meta-Features (Mean, Std, Min, Max)...")
        oof_stats = oof_preds.copy()
        oof_stats['mean'] = oof_preds.mean(axis=1)
        oof_stats['std'] = oof_preds.std(axis=1)
        oof_stats['min'] = oof_preds.min(axis=1)
        oof_stats['max'] = oof_preds.max(axis=1)
        
        # --- Level 2: Meta Learner ---
        print("\nTraining Meta Learner (XGBoost)...")
        log_file.write("\nTraining Meta Learner (XGBoost)...\n")
        
        # Using XGBoost as Meta Learner
        meta_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.05,
            max_depth=3,  # Shallow depth to prevent overfitting on meta-features
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=4
        )
        meta_model.fit(oof_stats, y)
        
        joblib.dump(meta_model, os.path.join(output_dir, 'meta_model_xgb.pkl'))
        
        # Evaluate Meta Model on OOF
        meta_oof_pred = meta_model.predict(oof_stats)
        
        meta_rounder = OptimizedRounder()
        meta_rounder.fit(meta_oof_pred, y)
        joblib.dump(meta_rounder, os.path.join(output_dir, 'meta_rounder_xgb.pkl'))
        
        scores = evaluate_model(y, meta_oof_pred, meta_rounder.coef_)
        
        print("\n=== Meta Model OOF Performance ===")
        log_file.write("\n=== Meta Model OOF Performance ===\n")
        print(f"Meta Rounder Coefficients: {meta_rounder.coef_}")
        log_file.write(f"Meta Rounder Coefficients: {meta_rounder.coef_}\n")
        res_str = f"RMSE: {scores['RMSE']:.4f}\nAccuracy: {scores['Accuracy']:.4f}\nQWK: {scores['QWK']:.4f}\n"
        print(res_str)
        log_file.write(res_str)
        
        # Save Feature Importance for Meta Learner
        plot_feature_importance(meta_model, 'Meta_XGBoost', oof_stats.columns.tolist(), output_dir)
    else:
        print("\nSkipping Level 2 Ensemble (Meta Learner) as requested.")
        log_file.write("\nSkipping Level 2 Ensemble (Meta Learner) as requested.\n")
    
    log_file.close()
    return base_model_names

def generate_inference(test_path, output_dir, train_cols, base_model_names):
    print(f"\nGenerating inference on {test_path}...")
    
    if not os.path.exists(test_path):
        print(f"Test file not found: {test_path}")
        return

    df_test = pd.read_csv(test_path)
    
    if 'id' in df_test.columns:
        ids = df_test['id'].values
        df_test = df_test.drop(columns=['id'])
    else:
        ids = np.arange(len(df_test))
        
    for col in train_cols:
        if col not in df_test.columns:
            df_test[col] = 0
    df_test = df_test[train_cols]
    
    meta_features = pd.DataFrame(index=df_test.index, columns=base_model_names)
    
    # Load CV Scores (Confidence Scores)
    cv_scores_path = os.path.join(output_dir, 'cv_scores.json')
    cv_scores = {}
    if os.path.exists(cv_scores_path):
        with open(cv_scores_path, 'r') as f:
            cv_scores = json.load(f)
    
    for name in base_model_names:
        print(f"  Generating base predictions for {name}...")
        fold_preds = []
        weights = []
        
        # Get weights for this model
        model_weights = cv_scores.get(name, [1.0] * 5)
        
        for fold in range(1, 6):
            model_path = os.path.join(output_dir, f'{name}_fold{fold}.pkl')
            if not os.path.exists(model_path):
                continue
            model = joblib.load(model_path)
            p = model.predict(df_test)
            if len(p.shape) > 1: p = p.flatten()
            fold_preds.append(p)
            
            # Use QWK as weight (clip negative to 0 if any)
            w = max(0, model_weights[fold-1]) if len(model_weights) >= fold else 1.0
            weights.append(w)
            
        if fold_preds:
            # Weighted Average
            fold_preds = np.array(fold_preds)
            weights = np.array(weights)
            
            if np.sum(weights) == 0:
                print(f"    Warning: Sum of weights for {name} is 0. Using mean.")
                avg_pred = np.mean(fold_preds, axis=0)
            else:
                # Weighted average across folds (axis 0 is folds)
                avg_pred = np.average(fold_preds, axis=0, weights=weights)
                
            meta_features[name] = avg_pred
    
    print("  Generating Meta-Features (Mean, Std, Min, Max)...")
    meta_features['mean'] = meta_features.mean(axis=1)
    meta_features['std'] = meta_features.std(axis=1)
    meta_features['min'] = meta_features.min(axis=1)
    meta_features['max'] = meta_features.max(axis=1)

    print("  Predicting with Meta Learner...")
    meta_model_path = os.path.join(output_dir, 'meta_model_xgb.pkl')
    meta_rounder_path = os.path.join(output_dir, 'meta_rounder_xgb.pkl')
    if os.path.exists(meta_model_path) and os.path.exists(meta_rounder_path):
        meta_model = joblib.load(meta_model_path)
        meta_rounder = joblib.load(meta_rounder_path)

        final_pred = meta_model.predict(meta_features)
        
        results = pd.DataFrame({'id': ids, 'sii': final_pred})
        results['sii_rounded'] = meta_rounder.predict(final_pred)
        
        out_csv = os.path.join(output_dir, 'stacking_predictions.csv')
        results.to_csv(out_csv, index=False)
        print(f"Saved {out_csv}")

def main():
    args = parse_args()
    
    if args.data_path:
        data_path = args.data_path
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, '../data/train_cleaned_imputed.csv')
        
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        return
        
    df = load_data(data_path)
    df = preprocess_data(df)
    
    X = df.drop(columns=['sii'])
    y = df['sii']
    
    base_models = train_stacking_cv(X, y, args.output_dir, models_to_train=args.models, run_ensemble=args.ensemble)
    
    if args.test_path:
        generate_inference(args.test_path, args.output_dir, X.columns.tolist(), base_models)

if __name__ == "__main__":
    main()
