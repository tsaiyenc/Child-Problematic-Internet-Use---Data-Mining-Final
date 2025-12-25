import pandas as pd
import numpy as np
import os
import argparse
import joblib
import warnings
from functools import partial
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score

# Suppress warnings
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using a single base model.")
    parser.add_argument('-m', '--model_name', type=str, required=True, help='Name of the model (e.g., XGBoost, LightGBM)')
    parser.add_argument('-d', '--model_dir', type=str, default='outputs/regression_stacking_SoftBalancedWeight', help='Directory containing trained models')
    parser.add_argument('-t', '--test_path', type=str, default='data/test_cleaned_imputed.csv', help='Path to test CSV')
    parser.add_argument('-o', '--output_path', type=str, help='Path to save predictions (default: submission_{model_name}.csv)')
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

def main():
    args = parse_args()
    model_name = args.model_name
    
    # Default output path
    output_path = args.output_path
    if not output_path:
        output_path = f"submission_{model_name}.csv"

    print(f"Loading predictions for {model_name} from {args.model_dir}...")
    print(f"Reading test data from {args.test_path}...")
    
    if not os.path.exists(args.test_path):
        print(f"Error: Test file not found at {args.test_path}")
        return
        
    df_test = pd.read_csv(args.test_path)
    
    # Handle ID column
    if 'id' in df_test.columns:
        ids = df_test['id'].values
        df_test = df_test.drop(columns=['id'])
    else:
        ids = np.arange(len(df_test))
        
    # Attempt to load feature names from a saved Fold 1 model
    # This ensures we use the exact same columns as training
    fold1_path = os.path.join(args.model_dir, f'{model_name}_fold1.pkl')
    train_cols = None
    if os.path.exists(fold1_path):
        try:
            model_f1 = joblib.load(fold1_path)
            if hasattr(model_f1, 'feature_names_in_'):
                train_cols = model_f1.feature_names_in_.tolist()
                print(f"Loaded {len(train_cols)} feature names from {model_name} Fold 1.")
            elif hasattr(model_f1, 'feature_name_'): # LightGBM
                train_cols = model_f1.feature_name_
                print(f"Loaded {len(train_cols)} feature names from {model_name} Fold 1.")
        except Exception as e:
            print(f"Warning: Could not load feature names from model: {e}")
            
    if train_cols:
        # Align test data columns
        for col in train_cols:
            if col not in df_test.columns:
                df_test[col] = 0
        df_test = df_test[train_cols]
    else:
        print("Warning: Could not determine training columns. Assuming test file columns match training (excluding 'sii').")
        if 'sii' in df_test.columns:
            df_test = df_test.drop(columns=['sii'])
            
    # Load model folds and predict
    print(f"Generating predictions for {model_name}...")
    fold_preds = []
    
    for fold in range(1, 6):
        model_path = os.path.join(args.model_dir, f'{model_name}_fold{fold}.pkl')
        if not os.path.exists(model_path):
            print(f"  Warning: {model_path} not found. Skipping fold.")
            continue
        
        try:
            model = joblib.load(model_path)
            p = model.predict(df_test)
            if len(p.shape) > 1:
                p = p.flatten()
            fold_preds.append(p)
            print(f"  Fold {fold} prediction complete.")
        except Exception as e:
            print(f"  Error with {model_name} fold {fold}: {e}")
            
    if not fold_preds:
        print(f"Error: No successful predictions for {model_name}")
        return

    # Average predictions
    avg_pred = np.mean(fold_preds, axis=0)
    # Standard deviation (Raw Confidence/Uncertainty)
    std_pred = np.std(fold_preds, axis=0)
    
    # Apply Rounding
    # Try to load OptimizedRounder specific to this model
    rounder_path = os.path.join(args.model_dir, f'{model_name}_rounder.pkl')
    if os.path.exists(rounder_path):
        print(f"Applying OptimizedRounder from {rounder_path}...")
        rounder = joblib.load(rounder_path)
        final_class = rounder.predict(avg_pred)
        coef = rounder.coef_
    else:
        print("Warning: Model-specific rounder not found. Using default rounding options (OptimizedRounder with default coefs).")
        rounder = OptimizedRounder()
        final_class = rounder.predict(avg_pred) # Uses default [0.5, 1.5, 2.5]
        coef = rounder.coef_
        
    # Calculate Confidence for Rounded Prediction (Distance to nearest threshold)
    # Higher value = more confident (farther from boundary)
    # If < coef[0], distance is coef[0] - pred
    # If between coef[0] and coef[1], distance is min(pred - coef[0], coef[1] - pred)
    # ...
    
    confidences = []
    for p in avg_pred:
        dists = [abs(p - c) for c in coef]
        confidences.append(min(dists))
    
    # Create submission dataframe
    submission = pd.DataFrame({
        'id': ids,
        'sii': final_class,
        'sii_raw': avg_pred,
        'sii_std': std_pred,
        'sii_confidence': confidences
    })
    
    # Ensure output dir exists
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # Save
    submission.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")

if __name__ == "__main__":
    main()
