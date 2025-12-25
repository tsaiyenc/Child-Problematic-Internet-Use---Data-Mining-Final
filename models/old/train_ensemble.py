import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
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
    parser = argparse.ArgumentParser(description="Train ensemble models.")
    parser.add_argument('--output_dir', type=str, default='outputs/ensemble', help='Directory to save outputs')
    parser.add_argument('--data_path', type=str, default=None, help='Path to input CSV')
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

def plot_learning_curve(history, model_name, output_dir):
    try:
        plt.figure(figsize=(10, 6))
        
        # Robustly find the loss metric
        loss_values = []
        metrics_found = False
        
        # Helper to find list in dict values
        def find_loss_list(d):
            for k, v in d.items():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], (int, float)):
                    # Heuristic: if key contains 'loss' or 'error' or 'MultiClass'
                    if 'loss' in k.lower() or 'error' in k.lower() or 'class' in k.lower():
                        return v
                elif isinstance(v, dict):
                    res = find_loss_list(v)
                    if res: return res
            return None

        loss_values = find_loss_list(history)
        
        if loss_values:
            plt.plot(range(len(loss_values)), loss_values, label='Validation Loss')
            plt.ylabel('Loss')
            plt.title(f'{model_name} Learning Curve')
            plt.xlabel('Iterations')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, f'learning_curve_{model_name}.png'))
        else:
            print(f"No loss metric found for {model_name} in history keys: {history.keys()}")
            
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
            try:
                plt.figure(figsize=(20, 10))
                xgb.plot_tree(model, num_trees=0, rankdir='LR')
                plt.title(f"{model_name} - First Tree")
                plt.savefig(os.path.join(output_dir, f'tree_viz_{model_name}.png'))
                plt.close()
            except Exception as e:
                print(f"Graphviz missing for {model_name}, saving text dump.")
                model.get_booster().dump_model(os.path.join(output_dir, f'tree_dump_{model_name}.txt'), with_stats=True)
            
        elif model_name == 'LightGBM':
            try:
                plt.figure(figsize=(20, 10))
                lgb.plot_tree(model, tree_index=0, figsize=(20, 10), show_info=['split_gain'])
                plt.title(f"{model_name} - First Tree")
                plt.savefig(os.path.join(output_dir, f'tree_viz_{model_name}.png'))
                plt.close()
            except Exception as e:
                print(f"Graphviz missing for {model_name}, saving text dump.")
                tree_json = model.booster_.dump_model()['tree_info'][0]
                with open(os.path.join(output_dir, f'tree_dump_{model_name}.json'), 'w') as f:
                    json.dump(tree_json, f, indent=4)

    except Exception as e:
        print(f"Could not plot tree for {model_name}: {e}")

def train_and_evaluate_cv(X, y, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Track results
    model_names = ['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'VotingEnsemble']
    model_results = {name: {'Accuracy': [], 'QWK': []} for name in model_names}
    
    feature_names = X.columns.tolist()
    log_file = open(os.path.join(output_dir, 'training_log.txt'), 'w')
    
    print(f"Starting Ensemble CV, saving to {output_dir}...")
    
    fold = 1
    for train_index, val_index in skf.split(X, y):
        print(f"\n=== Fold {fold}/5 ===")
        log_file.write(f"\n=== Fold {fold}/5 ===\n")
        log_file.flush()
        
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Generate sample weights for XGBoost to handle imbalance
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        
        # --- 1. XGBoost (Weighted) ---
        xgb_model = xgb.XGBClassifier(
            objective='multi:softmax', num_class=len(np.unique(y_train)),
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            eval_metric='mlogloss', n_jobs=4, early_stopping_rounds=50
        )
        xgb_model.fit(X_train, y_train, sample_weight=sample_weights, eval_set=[(X_val, y_val)], verbose=False)
        y_pred_xgb = xgb_model.predict(X_val)
        y_prob_xgb = xgb_model.predict_proba(X_val)
        
        scores = evaluate_model(y_val, y_pred_xgb, "XGBoost")
        model_results['XGBoost']['Accuracy'].append(scores['Accuracy'])
        model_results['XGBoost']['QWK'].append(scores['QWK'])
        
        if fold == 1:
            plot_learning_curve(xgb_model.evals_result(), 'XGBoost', output_dir)
            plot_first_tree(xgb_model, 'XGBoost', feature_names, output_dir)

        # --- 2. LightGBM (Weighted) ---
        lgb_model = lgb.LGBMClassifier(
            objective='multiclass', n_estimators=500, learning_rate=0.05,
            num_leaves=20, max_depth=6, min_child_samples=30,
            class_weight='balanced', random_state=42, verbosity=-1
        )
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=0)]
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='multi_logloss', callbacks=callbacks)
        y_pred_lgb = lgb_model.predict(X_val)
        y_prob_lgb = lgb_model.predict_proba(X_val)
        
        scores = evaluate_model(y_val, y_pred_lgb, "LightGBM")
        model_results['LightGBM']['Accuracy'].append(scores['Accuracy'])
        model_results['LightGBM']['QWK'].append(scores['QWK'])
        
        if fold == 1:
             if hasattr(lgb_model, 'evals_result_'):
                 plot_learning_curve(lgb_model.evals_result_, 'LightGBM', output_dir)
             plot_first_tree(lgb_model, 'LightGBM', feature_names, output_dir)

        # --- 3. CatBoost (Weighted) ---
        # auto_class_weights='Balanced' automatically handles imbalance
        cat_model = CatBoostClassifier(
            loss_function='MultiClass', iterations=1000, learning_rate=0.05,
            depth=6, l2_leaf_reg=5, nan_mode='Min', random_seed=42,
            verbose=False, allow_writing_files=False, early_stopping_rounds=50,
            auto_class_weights='Balanced' 
        )
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
        y_pred_cat = cat_model.predict(X_val).flatten()
        y_prob_cat = cat_model.predict_proba(X_val)
        
        scores = evaluate_model(y_val, y_pred_cat, "CatBoost")
        model_results['CatBoost']['Accuracy'].append(scores['Accuracy'])
        model_results['CatBoost']['QWK'].append(scores['QWK'])
        
        if fold == 1:
            plot_learning_curve(cat_model.get_evals_result(), 'CatBoost', output_dir)

        # --- 4. RandomForest (Weighted) ---
        rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight='balanced',
            random_state=42, n_jobs=4
        )
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_val)
        y_prob_rf = rf_model.predict_proba(X_val)
        
        scores = evaluate_model(y_val, y_pred_rf, "RandomForest")
        model_results['RandomForest']['Accuracy'].append(scores['Accuracy'])
        model_results['RandomForest']['QWK'].append(scores['QWK'])
        
        if fold == 1:
            plot_first_tree(rf_model, 'RandomForest', feature_names, output_dir)

        # --- 5. Voting Ensemble (Soft Vote) ---
        # Average probabilities
        ensemble_prob = (y_prob_xgb + y_prob_lgb + y_prob_cat + y_prob_rf) / 4.0
        ensemble_pred = np.argmax(ensemble_prob, axis=1)
        
        scores = evaluate_model(y_val, ensemble_pred, "VotingEnsemble")
        model_results['VotingEnsemble']['Accuracy'].append(scores['Accuracy'])
        model_results['VotingEnsemble']['QWK'].append(scores['QWK'])
        
        if fold == 1:
            plot_confusion_matrix_heatmap(y_val, ensemble_pred, 'VotingEnsemble', output_dir)

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
    
    train_and_evaluate_cv(X, y, args.output_dir)

if __name__ == "__main__":
    main()
