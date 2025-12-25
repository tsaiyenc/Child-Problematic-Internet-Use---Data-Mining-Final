import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.linear_model import LinearRegression

def visualize_features(input_file, output_dir):
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    if 'sii' not in df.columns:
        print("Error: 'sii' column not found in data.")
        return

    # # Identify feature columns (exclude 'id' and 'sii')
    # feature_cols = [col for col in df.columns if col not in ['id', 'sii']]
    # print(f"Found {len(feature_cols)} features to visualize.")

    # Identify series features
    series_feature = ['Indoor_Sedentary_Ratio','Indoor_Sedentary_Minutes','Outdoor_Active_Minutes','Night_Screen_Proxy_Minutes','Gaming_Micro_Movement_Ratio','Screen_Obsession_Index']
    print(f"Found {len(series_feature)} series features to visualize.")

    
    for i, feature in enumerate(series_feature):
        print(f"[{i+1}/{len(series_feature)}] Plotting {feature} vs sii...")
        
        plt.figure(figsize=(10, 6))
        
        try:
            # Drop NaNs for plotting
            plot_df = df[[feature, 'sii']].dropna()
            
            if plot_df.empty:
                print(f"Skipping {feature} due to missing data.")
                plt.close()
                continue

            # Check if feature is numeric
            if pd.api.types.is_numeric_dtype(plot_df[feature]):
                # Binning strategy
                unique_vals = plot_df[feature].nunique()
                
                if unique_vals > 50:
                    # For continuous data, use binning
                    # Create 50 bins
                    plot_df['bin'] = pd.cut(plot_df[feature], bins=50)
                    # Calculate mean of feature in each bin for x-axis
                    # Calculate mean of sii in each bin for y-axis
                    # Calculate count for size
                    agg_df = plot_df.groupby('bin', observed=True).agg({
                        feature: 'mean',
                        'sii': ['mean', 'count']
                    }).reset_index()
                    # Flatten columns
                    agg_df.columns = ['bin', feature, 'sii_mean', 'count']
                else:
                    # For discrete data with few values, group by value directly
                    agg_df = plot_df.groupby(feature).agg({
                        'sii': ['mean', 'count']
                    }).reset_index()
                    agg_df.columns = [feature, 'sii_mean', 'count']
                
                # Drop bins with no data if any resulted in NaN means
                agg_df = agg_df.dropna()

                # Plot aggregated points
                # x: feature value (or bin center), y: mean sii, size: count, color: count
                sns.scatterplot(
                    data=agg_df, 
                    x=feature, 
                    y='sii_mean', 
                    size='count', 
                    hue='count',
                    sizes=(20, 200),
                    alpha=0.7,
                    palette='viridis'
                )
                
                # Add Linear Regression Line (calculated on original data)
                X = plot_df[[feature]].values
                y = plot_df['sii'].values
                reg = LinearRegression().fit(X, y)
                
                # Generate line points
                x_range = np.linspace(plot_df[feature].min(), plot_df[feature].max(), 100).reshape(-1, 1)
                y_pred = reg.predict(x_range)
                
                plt.plot(x_range, y_pred, color='red', linestyle='--', linewidth=2, label=f'Regression (R2={reg.score(X, y):.2f})')
                plt.legend()
                plt.title(f'Aggregated: sii (Mean) vs {feature} with Regression')
                
            else:
                # For non-numeric features, fallback to standard plots but maybe aggregating means
                # Just bar plot of means?
                # User asked specifically for regression lines which implies numeric x usually.
                # If categorical, we can still show mean sii per category.
                agg_df = plot_df.groupby(feature)['sii'].mean().reset_index()
                sns.barplot(data=agg_df, x=feature, y='sii')
                plt.title(f'Mean sii vs {feature}')
                plt.xticks(rotation=45)

        except Exception as e:
            print(f"Could not plot {feature}: {e}")
            plt.close()
            continue

        plt.xlabel(feature)
        plt.ylabel('Mean sii')
        plt.tight_layout()
        
        # Sanitize filename
        safe_feature_name = feature.replace('/', '_').replace(' ', '_')
        plot_path = os.path.join(output_dir, f"sii_vs_{safe_feature_name}.png")
        plt.savefig(plot_path)
        plt.close()
        
    print(f"All plots saved to {output_dir}")

if __name__ == "__main__":
    current_dir = os.getcwd()
    # Handle running from parsers or analysis subdir or root
    if os.path.basename(current_dir) in ['parsers', 'analysis']:
        base_dir = os.path.dirname(current_dir)
    else:
        base_dir = current_dir
        
    input_path = os.path.join(base_dir, "data", "train_with_features.csv")
    # input_path = os.path.join(base_dir, "data", "train_cleaned.csv")
    output_dir = os.path.join(base_dir, "analysis", "plots")
    
    if os.path.exists(input_path):
        visualize_features(input_path, output_dir)
    else:
        print(f"Error: Input file not found at {input_path}")
