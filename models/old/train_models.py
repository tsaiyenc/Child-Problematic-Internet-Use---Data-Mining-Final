import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix, log_loss
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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

warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="Train models and generate detailed visualizations.")
    parser.add_argument('-o', '--output_dir', type=str, default='outputs/ver2', help='Directory to save outputs')
    parser.add_argument('-d', '--data_path', type=str, default='data/train_cleaned_imputed.csv', help='Path to input CSV')
    return parser.parse_args()

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    if 'sii' in df.columns:
        df['sii'] = df['sii'].astype(int)
    return df

def evaluate_model(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    return {'Accuracy': acc, 'QWK': qwk}

def plot_confusion_matrix_heatmap(y_true, y_pred, model_name, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(output_dir, f'confusion_matrix_{model_name}.png'))
    plt.close()

def plot_feature_importance(model, model_name, feature_names, output_dir, X_val=None, y_val=None):
    try:
        if model_name == 'HistGradientBoosting':
            if X_val is not None and y_val is not None:
                result = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=42, n_jobs=4)
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
        
        # Robustly find the loss metric or score list
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
                # First pass: look for loss/error
                for k, v in d.items():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
                        if 'loss' in k.lower() or 'error' in k.lower() or 'class' in k.lower():
                            return v, 'Validation Loss', 'Loss'
                    elif isinstance(v, dict):
                        res = find_metric_list(v)
                        if res: return res
                
                # Second pass: take any numeric list if no specific loss found
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
            print(f"No metric found for {model_name} in history: {history.keys() if isinstance(history, dict) else type(history)}")
            
        plt.close()
    except Exception as e:
        print(f"Could not plot learning curve for {model_name}: {e}")

def plot_sample_learning_curve(model, X, y, model_name, output_dir):
    try:
        plt.figure(figsize=(10, 6))
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=5, n_jobs=4, 
            train_sizes=np.linspace(0.1, 1.0, 5),
            scoring='accuracy'
        )
        
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
                         
        plt.title(f'{model_name} Learning Curve (Sample Size)')
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        plt.legend(loc="best")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'learning_curve_samples_{model_name}.png'))
        plt.close()
    except Exception as e:
        print(f"Could not plot sample learning curve for {model_name}: {e}")

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
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    model_results = {name: {'Accuracy': [], 'QWK': []} for name in ['XGBoost', 'LightGBM', 'CatBoost', 'HistGradientBoosting', 'RandomForest']}
    
    feature_names = X.columns.tolist()
    
    # Open log file
    log_file = open(os.path.join(output_dir, 'training_log.txt'), 'w')
    
    print(f"Starting CV, saving to {output_dir}...")
    
    fold = 1
    for train_index, val_index in skf.split(X, y):
        print(f"\n=== Fold {fold}/5 ===")
        log_file.write(f"\n=== Fold {fold}/5 ===\n")
        log_file.flush()
        
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # --- XGBoost ---
        xgb_model = xgb.XGBClassifier(
            objective='multi:softmax', num_class=len(np.unique(y_train)),
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='mlogloss', n_jobs=4, early_stopping_rounds=50
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        y_pred = xgb_model.predict(X_val)
        scores = evaluate_model(y_val, y_pred, "XGBoost")
        model_results['XGBoost']['Accuracy'].append(scores['Accuracy'])
        model_results['XGBoost']['QWK'].append(scores['QWK'])
        
        # Plotting (Only for first fold to save time/space)
        if fold == 1:
            plot_learning_curve(xgb_model.evals_result(), 'XGBoost', output_dir)
            plot_first_tree(xgb_model, 'XGBoost', feature_names, output_dir)
            plot_feature_importance(xgb_model, 'XGBoost', feature_names, output_dir)
            plot_confusion_matrix_heatmap(y_val, y_pred, 'XGBoost', output_dir)

        # --- LightGBM ---
        lgb_model = lgb.LGBMClassifier(
            objective='multiclass', n_estimators=500, learning_rate=0.05,
            num_leaves=20, max_depth=6, min_child_samples=30,
            class_weight='balanced', random_state=42, verbosity=-1
        )
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)]
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='multi_logloss', callbacks=callbacks)
        y_pred = lgb_model.predict(X_val)
        scores = evaluate_model(y_val, y_pred, "LightGBM")
        model_results['LightGBM']['Accuracy'].append(scores['Accuracy'])
        model_results['LightGBM']['QWK'].append(scores['QWK'])
        
        if fold == 1:
             # Capture history manually as sklearn API varies
            if hasattr(lgb_model, 'evals_result_'):
                 plot_learning_curve(lgb_model.evals_result_, 'LightGBM', output_dir)
            plot_first_tree(lgb_model, 'LightGBM', feature_names, output_dir)
            plot_feature_importance(lgb_model, 'LightGBM', feature_names, output_dir)
            plot_confusion_matrix_heatmap(y_val, y_pred, 'LightGBM', output_dir)

        # --- CatBoost ---
        cat_model = CatBoostClassifier(
            loss_function='MultiClass', iterations=1000, learning_rate=0.05,
            depth=6, l2_leaf_reg=5, nan_mode='Min', random_seed=42,
            verbose=False, allow_writing_files=False, early_stopping_rounds=50
        )
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
        y_pred = cat_model.predict(X_val).flatten()
        scores = evaluate_model(y_val, y_pred, "CatBoost")
        model_results['CatBoost']['Accuracy'].append(scores['Accuracy'])
        model_results['CatBoost']['QWK'].append(scores['QWK'])
        
        if fold == 1:
            plot_learning_curve(cat_model.get_evals_result(), 'CatBoost', output_dir)
            plot_feature_importance(cat_model, 'CatBoost', feature_names, output_dir)
            plot_confusion_matrix_heatmap(y_val, y_pred, 'CatBoost', output_dir)
            # CatBoost tree plotting is complex without widget, skipping simple plot

        # --- RandomForest ---
        """
        WHY RANDOM FOREST WORKS BEST HERE (Hypothesis):
        1. Bagging vs Boosting: With high noise/missing data, Boosting can overfit to the noise (outliers).
           Bagging (Random Forest) reduces variance by averaging, which is often more robust for smaller, noisy datasets.
        2. Class Weight 'balanced': Heavily penalizes errors on minority classes (sii=2, 3).
           QWK is sensitive to these "rare but severe" cases.
           
        TUNING TIPS:
        - min_samples_leaf: Increase (e.g., 5, 10) to force smoother boundaries (reduce overfitting further).
        - max_features: 'sqrt' is standard, but try 'log2' or a float (0.5) if features are redundant.
        - criterion: 'entropy' might work better than 'gini' for some multiclass problems.
        """
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=8, 
            class_weight='balanced',
            random_state=42,
            n_jobs=4
        )
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_val)
        scores = evaluate_model(y_val, y_pred, "RandomForest")
        model_results['RandomForest']['Accuracy'].append(scores['Accuracy'])
        model_results['RandomForest']['QWK'].append(scores['QWK'])
        
        if fold == 1:
            plot_first_tree(rf_model, 'RandomForest', feature_names, output_dir)
            plot_feature_importance(rf_model, 'RandomForest', feature_names, output_dir)
            plot_confusion_matrix_heatmap(y_val, y_pred, 'RandomForest', output_dir)
            plot_sample_learning_curve(rf_model, X_train, y_train, 'RandomForest', output_dir)

        # --- HistGradientBoosting ---
        hgb_model = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=500, max_depth=6,
            random_state=42, early_stopping=True
        )
        hgb_model.fit(X_train, y_train)
        y_pred = hgb_model.predict(X_val)
        scores = evaluate_model(y_val, y_pred, "HistGradientBoosting")
        model_results['HistGradientBoosting']['Accuracy'].append(scores['Accuracy'])
        model_results['HistGradientBoosting']['QWK'].append(scores['QWK'])
        
        if fold == 1:
            plot_confusion_matrix_heatmap(y_val, y_pred, 'HistGradientBoosting', output_dir)
            if hasattr(hgb_model, 'validation_scores_'):
                plot_learning_curve(hgb_model.validation_scores_, 'HistGradientBoosting', output_dir)
            plot_feature_importance(hgb_model, 'HistGradientBoosting', feature_names, output_dir, X_val=X_val, y_val=y_val)
             
        fold += 1

    print("\n\n=== Final CV Results ===")
    log_file.write("\n\n=== Final CV Results ===\n")
    for model_name, metrics in model_results.items():
        acc_mean = np.mean(metrics['Accuracy'])
        acc_std = np.std(metrics['Accuracy'])
        qwk_mean = np.mean(metrics['QWK'])
        qwk_std = np.std(metrics['QWK'])
        
        res_str = f"{model_name :<22} | Accuracy: {acc_mean:.4f} (+/- {acc_std:.4f}) | QWK: {qwk_mean:.4f} (+/- {qwk_std:.4f})"
        print(res_str)
        log_file.write(res_str + "\n")
        
    log_file.close()

def main():
    args = parse_args()
    
    # Defaults logic
    if args.data_path:
        data_path = args.data_path
    else:
        # Auto-detect relative to script
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base_dir, '../data/train_cleaned_imputed.csv')
        
    if not os.path.exists(data_path):
        # Fallback
        data_path = 'data/train_cleaned_imputed.csv'
        
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    df = load_data(data_path)
    df = preprocess_data(df)
    
    X = df.drop(columns=['sii'])
    y = df['sii']
    
    train_and_evaluate_cv(X, y, args.output_dir)

if __name__ == "__main__":
    main()
