
import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
import sklearn

def calculate_confidence(y_pred):
    """
    Calculate confidence score based on distance to nearest integer.
    Score 1.0 = exact integer match.
    Score 0.0 = exactly halfway (x.5).
    """
    nearest = np.round(y_pred)
    dist = np.abs(y_pred - nearest)
    conf = 1 - (2 * dist)
    return np.clip(conf, 0, 1)

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_path = os.path.join(base_dir, 'data', 'test_cleaned_imputed.csv')
    train_path = os.path.join(base_dir, 'data', 'train_cleaned_imputed.csv')
    output_dir = os.path.join(base_dir, 'outputs', 'regression_ver1')
    
    print(f"Test Data: {test_path}")
    print(f"Model Dir: {output_dir}")

    if not os.path.exists(test_path):
        print(f"Error: Test file not found at {test_path}")
        return

    if not os.path.exists(output_dir):
        print(f"Error: Output directory not found at {output_dir}")
        return

    # 1. Determine Feature Columns
    train_cols = []
    if os.path.exists(train_path):
        print("Loading feature names from train data...")
        df_train = pd.read_csv(train_path, nrows=1)
        # Exclude 'id' and 'sii' (target)
        train_cols = [c for c in df_train.columns if c not in ['id', 'sii']]
    else:
        print("Warning: Train file not found. Inferring features from test data (excluding 'id').")
        df_test_temp = pd.read_csv(test_path, nrows=1)
        train_cols = [c for c in df_test_temp.columns if c != 'id']

    print(f"Total Features: {len(train_cols)}")

    # 2. Load and Prepare Test Data
    print("Loading test data...")
    df_test = pd.read_csv(test_path)
    
    # Handle ID column
    if 'id' in df_test.columns:
        ids = df_test['id'].values
    else:
        ids = np.arange(len(df_test))
    
    # Align Columns: Ensure X_test has exactly the train_cols in order
    X_test = df_test.copy()
    for col in train_cols:
        if col not in X_test.columns:
            X_test[col] = 0
            
    X_test = X_test[train_cols]
    print(f"Test Data Shape: {X_test.shape}")

    # 3. Model Inference
    model_names = ['XGBoost', 'LightGBM', 'CatBoost', 'HistGradientBoosting', 'RandomForest']
    
    for model_name in model_names:
        print(f"\n--- Processing {model_name} ---")
        
        # Prepare results dictionary
        results = {'id': ids}
        fold_preds_list = []
        folds_found = 0
        
        for fold in range(1, 6):
            model_file = os.path.join(output_dir, f'{model_name}_fold{fold}.pkl')
            
            if not os.path.exists(model_file):
                print(f"  Warning: {model_file} not found. Skipping Fold {fold}.")
                continue
            
            try:
                # Load Model
                model = joblib.load(model_file)
                
                # Predict
                pred = model.predict(X_test)
                
                # Handle possible shape issues (e.g. (N,1) vs (N,))
                if isinstance(pred, pd.Series):
                    pred = pred.values
                if len(pred.shape) > 1:
                    pred = pred.flatten()
                    
                # Calculate Confidence
                conf = calculate_confidence(pred)
                
                # Store
                results[f'fold{fold}_pred'] = pred
                results[f'fold{fold}_conf'] = conf
                
                fold_preds_list.append(pred)
                folds_found += 1
                print(f"  Fold {fold}: Processed")
                
            except Exception as e:
                print(f"  Error processing Fold {fold} for {model_name}: {e}")

        # 4. Aggregate Results
        if folds_found > 0:
            fold_preds_arr = np.array(fold_preds_list)
            
            # Average
            avg_pred = np.mean(fold_preds_arr, axis=0)
            results['average'] = avg_pred
            
            # Round
            results['round'] = np.round(avg_pred).astype(int)
            
            # Create DataFrame
            out_df = pd.DataFrame(results)
            
            # Reorder columns to match request: id, fold1.., fold2.., average, round
            # Standard dict order is already insertion order in 3.7+, but let's be explicit if needed
            # The insertion order: id, f1_p, f1_c, f2_p, f2_c... avg, rnd. This is exactly what is requested.
            
            output_filename = f'{model_name}_inference_detailed.csv'
            output_path = os.path.join(output_dir, output_filename)
            out_df.to_csv(output_path, index=False)
            print(f"Successfully saved results to: {output_path}")
        else:
            print(f"No valid predictions generated for {model_name}.")

    print("\nInference Complete.")

if __name__ == "__main__":
    main()
