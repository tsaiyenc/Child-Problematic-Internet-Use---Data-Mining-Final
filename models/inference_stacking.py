import pandas as pd
import numpy as np
import joblib
import os
import argparse
import json
import warnings
from scipy.optimize import minimize
from functools import partial
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings('ignore')

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

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for Stacking Ensemble.")
    parser.add_argument('-t', '--test_path', type=str, default='data/test_cleaned_imputed.csv', help='Path to test CSV')
    parser.add_argument('-m', '--model_dir', type=str, default='outputs/regression_stacking_level1', help='Directory containing trained models (Level 1 and Level 2)')
    parser.add_argument('--meta-model', type=str, default='XGBoost', help='Name of the meta-learner used during training (e.g., XGBoost, HistGradientBoosting)')
    parser.add_argument('--base-models', nargs='+', type=str, 
                        default=['XGBoost', 'LightGBM', 'CatBoost', 'RandomForest', 'HistGradientBoosting'],
                        help='List of base models to use')
    return parser.parse_args()

def generate_inference(test_path, model_dir, base_model_names, meta_model_name):
    print(f"\nGenerating inference on {test_path}...")
    
    if not os.path.exists(test_path):
        print(f"Test file not found: {test_path}")
        return

    df_test = pd.read_csv(test_path)
    
    # Handle ID column
    if 'id' in df_test.columns:
        ids = df_test['id'].values
        # Keep features only
        # We need to know which columns were used for training. 
        # Ideally, we should save feature names during training. 
        # For now, we assume commonly dropped columns or just drop 'id'.
        # A robust way is to load one model and check features, or rely on consistency.
        X_test = df_test.drop(columns=['id'])
    else:
        ids = np.arange(len(df_test))
        X_test = df_test.copy()
        
    # Ensure columns match (simple validation)
    # in a real pipeline, we'd align columns strictly.
    
    meta_features = pd.DataFrame(index=X_test.index, columns=base_model_names)
    
    # Load CV Scores (Confidence Scores) for weighting
    cv_scores_path = os.path.join(model_dir, 'cv_scores.json')
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
            model_path = os.path.join(model_dir, f'{name}_fold{fold}.pkl')
            if not os.path.exists(model_path):
                # print(f"    Warning: Model file {model_path} not found.")
                continue
            
            try:
                model = joblib.load(model_path)
                # Ensure input features match
                # Some models (like LGBM) might be picky about column names if preserved
                p = model.predict(X_test)
                if len(p.shape) > 1: p = p.flatten()
                fold_preds.append(p)
                
                # Use QWK as weight (clip negative to 0 if any)
                w = max(0, model_weights[fold-1]) if len(model_weights) >= fold else 1.0
                weights.append(w)
            except Exception as e:
                print(f"    Error predicting with {name} fold {fold}: {e}")
                
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
        else:
             print(f"    Warning: No predictions generated for {name}.")
    
    print("  Generating Meta-Features (Mean, Std, Min, Max)...")
    meta_features['mean'] = meta_features.mean(axis=1)
    meta_features['std'] = meta_features.std(axis=1)
    meta_features['min'] = meta_features.min(axis=1)
    meta_features['max'] = meta_features.max(axis=1)

    print(f"  Predicting with Meta Learner ({meta_model_name})...")
    meta_model_path = os.path.join(model_dir, f'meta_model_{meta_model_name}.pkl')
    meta_rounder_path = os.path.join(model_dir, f'meta_rounder_{meta_model_name}.pkl')
    
    if os.path.exists(meta_model_path) and os.path.exists(meta_rounder_path):
        meta_model = joblib.load(meta_model_path)
        meta_rounder = joblib.load(meta_rounder_path)

        final_pred = meta_model.predict(meta_features)
        
        results = pd.DataFrame({'id': ids, 'sii': final_pred})
        results['sii_rounded'] = meta_rounder.predict(final_pred)
        
        out_csv = os.path.join(model_dir, f'submission_stacking_{meta_model_name}.csv')
        results.to_csv(out_csv, index=False)
        print(f"Saved {out_csv}")
    else:
        print(f"output model or rounder not found for {meta_model_name}")

if __name__ == "__main__":
    args = parse_args()
    generate_inference(args.test_path, args.model_dir, args.base_models, args.meta_model)
