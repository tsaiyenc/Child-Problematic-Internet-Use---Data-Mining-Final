import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
import warnings
import json
import joblib

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="Train regression models, save them, and generate inference.")
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/regression_ver1', help='Directory to save outputs')
    parser.add_argument('-d', '--data_path', type=str, default='data/train_cleaned_imputed.csv', help='Path to input CSV')
    parser.add_argument('-t', '--test_path', type=str, default='data/test_cleaned_imputed.csv', help='Path to test CSV')
    return parser.parse_args()

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    if 'sii' in df.columns:
        df['sii'] = df['sii'].astype(float) # Target as float for regression
    return df

def optimize_predictions(y_pred):
    # Round to nearest integer and clip to range 0-3
    return np.round(np.clip(y_pred, 0, 3)).astype(int)

def evaluate_model(y_true, y_pred, model_name):
    # Calculate RMSE on raw float predictions
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate Classification metrics on rounded predictions
    y_pred_rounded = optimize_predictions(y_pred)
    y_true_int = y_true.astype(int)
    
    acc = accuracy_score(y_true_int, y_pred_rounded)
    qwk = cohen_kappa_score(y_true_int, y_pred_rounded, weights='quadratic')
    
    return {'RMSE': rmse, 'Accuracy': acc, 'QWK': qwk}

def plot_confusion_matrix_heatmap(y_true, y_pred, model_name, output_dir):
    y_pred_rounded = optimize_predictions(y_pred)
    y_true_int = y_true.astype(int)
    
    cm = confusion_matrix(y_true_int, y_pred_rounded)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Rounded): {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}.png'))
    plt.close()

def plot_actual_vs_predicted(y_true, y_pred, model_name, output_dir):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, alpha=0.3)
    
    # Perfect fit line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.xlabel('Actual SII')
    plt.ylabel('Predicted SII')
    plt.title(f'Actual vs Predicted: {model_name}')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'actual_vs_predicted_{model_name}.png'))
    plt.close()

def plot_residuals(y_true, y_pred, model_name, output_dir):
    residuals = y_pred - y_true
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.3)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Predicted SII')
    plt.ylabel('Residuals (Predicted - Actual)')
    plt.title(f'Residual Plot: {model_name}')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f'residuals_{model_name}.png'))
    plt.close()

def plot_feature_importance(model, model_name, feature_names, output_dir, X_val=None, y_val=None):
    try:
        importances = None
        if model_name == 'HistGradientBoosting':
            if X_val is not None and y_val is not None:
                result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=4, scoring='neg_mean_squared_error')
                importances = result.importances_mean
            else:
                print(f"Skipping feature importance for {model_name} (requires X_val, y_val)")
                return
        elif hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'feature_importance'): # LightGBM
            importances = model.feature_importance()
        else:
            print(f"Skipping feature importance for {model_name} (not supported)")
            return

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
        ylabel = 'Value'
        
        if isinstance(history, (list, np.ndarray)):
            # Direct list (e.g., HistGradientBoosting validation_scores_)
            values = history
            label = 'Validation Score'
            ylabel = 'Score'
        elif isinstance(history, dict):
            # Helper to find list in dict values
            def find_metric_list(d):
                # First pass: look for loss/error, rmse, mse
                for k, v in d.items():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
                        k_lower = k.lower()
                        if 'loss' in k_lower or 'error' in k_lower or 'rmse' in k_lower or 'mse' in k_lower:
                            return v, f'Validation {k}', k
                    elif isinstance(v, dict):
                        res = find_metric_list(v)
                        if res: return res
                
                # Second pass: take any numeric list
                for k, v in d.items():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
                        return v, f'Validation {k}', k
                return None

            found = find_metric_list(history)
            if found:
                values, label, ylabel = found
        
        if len(values) > 0:
            plt.plot(range(len(values)), values, label=label)
            plt.ylabel(ylabel)
            plt.title(f'{model_name} Learning Curve')
            plt.xlabel('Iterations')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'learning_curve_{model_name}.png'))
        else:
            print(f"No metric found for {model_name} in history")
            
        plt.close()
    except Exception as e:
        print(f"Could not plot learning curve for {model_name}: {e}")

def plot_first_tree(model, model_name, feature_names, output_dir):
    try:
        if model_name == 'RandomForest':
            # RandomForest uses matplotlib, usually safe
            plt.figure(figsize=(20, 10))
            plot_tree(model.estimators_[0], feature_names=feature_names, max_depth=3, filled=True, proportion=True, rounded=True)
            plt.title(f"{model_name} - First Tree (Depth 3)")
            plt.savefig(os.path.join(output_dir, f'tree_viz_{model_name}.png'))
            plt.close()
            
        elif model_name == 'XGBoost':
            try:
                plt.figure(figsize=(20, 10))
                xgb.plot_tree(model, num_trees=0, rankdir='LR')
                plt.title(f"{model_name} - First Tree")
                plt.savefig(os.path.join(output_dir, f'tree_viz_{model_name}.png'))
                plt.close()
            except Exception as e:
                print(f"Graphviz not available for {model_name} visualization. Saving text dump instead.")
                # Fallback: Save text representation
                model.get_booster().dump_model(os.path.join(output_dir, f'tree_dump_{model_name}.txt'), with_stats=True)
            
        elif model_name == 'LightGBM':
            try:
                plt.figure(figsize=(20, 10))
                lgb.plot_tree(model, tree_index=0, figsize=(20, 10), show_info=['split_gain'])
                plt.title(f"{model_name} - First Tree")
                plt.savefig(os.path.join(output_dir, f'tree_viz_{model_name}.png'))
                plt.close()
            except Exception as e:
                print(f"Graphviz not available for {model_name} visualization. Saving text dump instead.")
                # Fallback: Save text representation of the first tree
                tree_json = model.booster_.dump_model()['tree_info'][0]
                with open(os.path.join(output_dir, f'tree_dump_{model_name}.json'), 'w') as f:
                    json.dump(tree_json, f, indent=4)

    except Exception as e:
        print(f"Could not plot tree for {model_name}. Error: {e}")

def train_and_evaluate_cv(X, y, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    y_stratify = optimize_predictions(y)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    model_results = {name: {'RMSE': [], 'Accuracy': [], 'QWK': []} for name in ['XGBoost', 'LightGBM', 'CatBoost', 'HistGradientBoosting', 'RandomForest']}
    
    feature_names = X.columns.tolist()
    
    log_file = open(os.path.join(output_dir, 'training_log.txt'), 'w')
    
    print(f"Starting CV (Regression), saving to {output_dir}...")
    
    fold = 1
    for train_index, val_index in skf.split(X, y_stratify):
        print(f"\n=== Fold {fold}/5 ===")
        log_file.write(f"\n=== Fold {fold}/5 ===\n")
        log_file.flush()
        
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # --- XGBoost ---
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='rmse', n_jobs=4, early_stopping_rounds=50
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        joblib.dump(xgb_model, os.path.join(output_dir, f'XGBoost_fold{fold}.pkl'))
        
        y_pred = xgb_model.predict(X_val)
        scores = evaluate_model(y_val, y_pred, "XGBoost")
        for k, v in scores.items(): model_results['XGBoost'][k].append(v)
        
        if fold == 1:
            plot_learning_curve(xgb_model.evals_result(), 'XGBoost', output_dir)
            plot_feature_importance(xgb_model, 'XGBoost', feature_names, output_dir)
            plot_confusion_matrix_heatmap(y_val, y_pred, 'XGBoost', output_dir)
            plot_actual_vs_predicted(y_val, y_pred, 'XGBoost', output_dir)
            plot_residuals(y_val, y_pred, 'XGBoost', output_dir)
            plot_first_tree(xgb_model, 'XGBoost', feature_names, output_dir)

        # --- LightGBM ---
        lgb_model = lgb.LGBMRegressor(
             objective='regression', n_estimators=500, learning_rate=0.05,
             num_leaves=20, max_depth=6, min_child_samples=30,
             random_state=42, verbosity=-1
        )
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)]
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='rmse', callbacks=callbacks)
        joblib.dump(lgb_model, os.path.join(output_dir, f'LightGBM_fold{fold}.pkl'))
        
        y_pred = lgb_model.predict(X_val)
        scores = evaluate_model(y_val, y_pred, "LightGBM")
        for k, v in scores.items(): model_results['LightGBM'][k].append(v)
        
        if fold == 1:
            if hasattr(lgb_model, 'evals_result_'):
                 plot_learning_curve(lgb_model.evals_result_, 'LightGBM', output_dir)
            plot_feature_importance(lgb_model, 'LightGBM', feature_names, output_dir)
            plot_confusion_matrix_heatmap(y_val, y_pred, 'LightGBM', output_dir)
            plot_actual_vs_predicted(y_val, y_pred, 'LightGBM', output_dir)
            plot_residuals(y_val, y_pred, 'LightGBM', output_dir)
            plot_first_tree(lgb_model, 'LightGBM', feature_names, output_dir)

        # --- CatBoost ---
        cat_model = CatBoostRegressor(
            loss_function='RMSE', iterations=1000, learning_rate=0.05,
            depth=6, l2_leaf_reg=5, random_seed=42,
            verbose=False, allow_writing_files=False, early_stopping_rounds=50
        )
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
        # Note: CatBoost usually saves internal format, but joblib works for sklearn wrapper usually.
        # For safety/consistency with other sklearn models, joblib is fine here as it pickles the object.
        joblib.dump(cat_model, os.path.join(output_dir, f'CatBoost_fold{fold}.pkl'))
        
        y_pred = cat_model.predict(X_val).flatten()
        scores = evaluate_model(y_val, y_pred, "CatBoost")
        for k, v in scores.items(): model_results['CatBoost'][k].append(v)
        
        if fold == 1:
            plot_learning_curve(cat_model.get_evals_result(), 'CatBoost', output_dir)
            plot_feature_importance(cat_model, 'CatBoost', feature_names, output_dir)
            plot_confusion_matrix_heatmap(y_val, y_pred, 'CatBoost', output_dir)
            plot_actual_vs_predicted(y_val, y_pred, 'CatBoost', output_dir)

        # --- RandomForest ---
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=8, 
            random_state=42,
            n_jobs=4
        )
        rf_model.fit(X_train, y_train)
        joblib.dump(rf_model, os.path.join(output_dir, f'RandomForest_fold{fold}.pkl'))
        
        y_pred = rf_model.predict(X_val)
        scores = evaluate_model(y_val, y_pred, "RandomForest")
        for k, v in scores.items(): model_results['RandomForest'][k].append(v)
        
        if fold == 1:
            plot_feature_importance(rf_model, 'RandomForest', feature_names, output_dir)
            plot_confusion_matrix_heatmap(y_val, y_pred, 'RandomForest', output_dir)
            plot_actual_vs_predicted(y_val, y_pred, 'RandomForest', output_dir)
            plot_residuals(y_val, y_pred, 'RandomForest', output_dir)
            plot_first_tree(rf_model, 'RandomForest', feature_names, output_dir)

        # --- HistGradientBoosting ---
        hgb_model = HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=500, max_depth=6,
            random_state=42, early_stopping=True
        )
        hgb_model.fit(X_train, y_train)
        joblib.dump(hgb_model, os.path.join(output_dir, f'HistGradientBoosting_fold{fold}.pkl'))
        
        y_pred = hgb_model.predict(X_val)
        scores = evaluate_model(y_val, y_pred, "HistGradientBoosting")
        for k, v in scores.items(): model_results['HistGradientBoosting'][k].append(v)
        
        if fold == 1:
            plot_confusion_matrix_heatmap(y_val, y_pred, 'HistGradientBoosting', output_dir)
            if hasattr(hgb_model, 'validation_scores_'):
                plot_learning_curve(hgb_model.validation_scores_, 'HistGradientBoosting', output_dir)
            plot_feature_importance(hgb_model, 'HistGradientBoosting', feature_names, output_dir, X_val=X_val, y_val=y_val)
            plot_actual_vs_predicted(y_val, y_pred, 'HistGradientBoosting', output_dir)
            plot_residuals(y_val, y_pred, 'HistGradientBoosting', output_dir)
             
        fold += 1

    print("\n\n=== Final CV Results (Regression) ===")
    log_file.write("\n\n=== Final CV Results (Regression) ===\n")
    for model_name, metrics in model_results.items():
        rmse_mean = np.mean(metrics['RMSE'])
        rmse_std = np.std(metrics['RMSE'])
        acc_mean = np.mean(metrics['Accuracy'])
        acc_std = np.std(metrics['Accuracy'])
        qwk_mean = np.mean(metrics['QWK'])
        qwk_std = np.std(metrics['QWK'])
        
        res_str = f"{model_name :<22} | RMSE: {rmse_mean:.4f} (+/- {rmse_std:.4f}) | Accuracy: {acc_mean:.4f} (+/- {acc_std:.4f}) | QWK: {qwk_mean:.4f} (+/- {qwk_std:.4f})"
        print(res_str)
        log_file.write(res_str + "\n")
        
    log_file.close()

def calculate_confidence(y_pred):
    # Regression models don't output probabilities.
    # We define "confidence" as how close the prediction is to the nearest integer.
    # Dist 0 (e.g. 1.0) -> Conf 1.0
    # Dist 0.5 (e.g. 1.5) -> Conf 0.0
    # Formula: 1 - (2 * |pred - round(pred)|)
    nearest = np.round(y_pred)
    dist = np.abs(y_pred - nearest)
    conf = 1 - (2 * dist)
    return np.clip(conf, 0, 1)

def generate_inference(test_path, output_dir, train_cols):
    print(f"\nGenerating inference on {test_path}...")
    
    if not os.path.exists(test_path):
        print(f"Test file not found: {test_path}")
        return

    df_test = pd.read_csv(test_path)
    
    # Store IDs
    if 'id' in df_test.columns:
        ids = df_test['id'].values
        df_test = df_test.drop(columns=['id'])
    else:
        ids = np.arange(len(df_test))
        
    # Align columns
    for col in train_cols:
        if col not in df_test.columns:
            df_test[col] = 0
            
    df_test = df_test[train_cols]
    
    model_names = ['XGBoost', 'LightGBM', 'CatBoost', 'HistGradientBoosting', 'RandomForest']
    
    for model_name in model_names:
        print(f"Predicting with {model_name}...")
        
        # Dictionary to store results
        results = {'id': ids}
        fold_preds_list = []
        
        for fold in range(1, 6):
            model_path = os.path.join(output_dir, f'{model_name}_fold{fold}.pkl')
            if not os.path.exists(model_path):
                print(f"  Warning: Model file {model_path} not found. Skipping fold.")
                continue
                
            model = joblib.load(model_path)
            
            try:
                p = model.predict(df_test)
                if len(p.shape) > 1:
                    p = p.flatten()
                
                # Add to results
                results[f'fold{fold}_pred'] = p
                results[f'fold{fold}_conf'] = calculate_confidence(p)
                
                fold_preds_list.append(p)
            except Exception as e:
                print(f"  Error predicting fold {fold}: {e}")
                
        if not fold_preds_list:
            print(f"  No predictions generated for {model_name}.")
            continue
            
        # Clean up list for averaging
        fold_all = np.array(fold_preds_list)
        
        # Calculate Average
        avg_preds = np.mean(fold_all, axis=0)
        results['average_pred'] = avg_preds
        
        # Calculate Rounded Average
        results['average_rounded'] = np.round(avg_preds).astype(int)
        
        # Save to CSV
        out_csv = os.path.join(output_dir, f'{model_name}_predictions.csv')
        res_df = pd.DataFrame(results)
        res_df.to_csv(out_csv, index=False)
        print(f"  Saved {out_csv}")

def main():
    args = parse_args()
    
    # Defaults logic
    if args.data_path:
        data_path = args.data_path
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, '../data/train_cleaned_imputed.csv')
        
    if not os.path.exists(data_path):
        data_path = 'data/train_cleaned_imputed.csv'
        
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    df = load_data(data_path)
    df = preprocess_data(df)
    
    X = df.drop(columns=['sii'])
    y = df['sii']
    
    # Train
    train_and_evaluate_cv(X, y, args.output_dir)
    
    # Inference
    if args.test_path:
        generate_inference(args.test_path, args.output_dir, X.columns.tolist())

if __name__ == "__main__":
    main()
