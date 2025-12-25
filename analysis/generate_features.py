# -*- coding: utf-8 -*-
import os
import glob
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import argparse
from tqdm import tqdm

def create_advanced_features(df):
    """
    輸入原始 DataFrame，輸出每個 ID 的進階生理特徵
    """
    # 確保有 hour 欄位
    if 'hour' not in df.columns:
        # time_of_day is in nanoseconds. Convert to hours.
        df['hour'] = (df['time_of_day'] / 1e9) / 3600

    def get_id_features(group):
        # 0. 預處理：只取有配戴的期間
        # non-wear_flag: 0 means wear, 1 means non-wear (usually) - wait, based on user code "group['non-wear_flag'] == 0" means valid/wear.
        valid = group[group['non-wear_flag'] == 0].copy()
        if valid.empty: return pd.Series()
        
        # --- Group A: L5 & M10 (經典節律指標) ---
        # 技巧：將數據按「小時」聚合，形成一個「平均的一天」
        # 這能有效解決數據中間有斷層(Missing Days)的問題
        hourly_profile = valid.groupby(valid['hour'].astype(int))['enmo'].mean()
        
        # 補齊 0-23 小時 (如果有缺漏)
        hourly_profile = hourly_profile.reindex(range(24), fill_value=0)
        
        # 為了計算跨午夜的窗口 (例如 22:00 到 03:00)，我們將數據串接 3 次 (48小時+24小時)
        # 這樣 rolling window 就能順利滑過午夜
        extended_profile = pd.concat([hourly_profile, hourly_profile, hourly_profile])
        
        # L5: 5小時滑動平均的最小值
        l5_val = extended_profile.rolling(5).mean().min()
        
        # M10: 10小時滑動平均的最大值
        m10_val = extended_profile.rolling(10).mean().max()
        
        # RA: 相對振幅 (Relative Amplitude)
        if (m10_val + l5_val) > 0:
            ra_val = (m10_val - l5_val) / (m10_val + l5_val)
        else:
            ra_val = 0
            
        # --- Group B: 夜間躁動 (Sleep Quality Proxy) ---
        # 定義夜間窗口: 22:00 - 06:00
        # 注意: 這裡的 hour 是 0-24 的 float
        night_mask = (valid['hour'] >= 22) | (valid['hour'] < 6)
        night_data = valid[night_mask]
        
        if not night_data.empty:
            # 夜間 AngleZ 的波動程度 (翻身頻率)
            anglez_std_night = night_data['anglez'].std()
            # 夜間平均活動量
            enmo_mean_night = night_data['enmo'].mean()
        else:
            anglez_std_night = np.nan
            enmo_mean_night = np.nan

        # --- Group C: 活動分佈統計 (Statistics) ---
        # 全天 ENMO 的偏態 (Skewness) 與 峰度 (Kurtosis)
        enmo_skew = skew(valid['enmo'])
        enmo_kurt = kurtosis(valid['enmo'])
        
        # --- Group D: 週末效應 (Social Jetlag Proxy) ---
        # 比較 平日(1-5) vs 週末(6-7) 的平均活動量差異
        # Assume 'weekday' column exists. 1=Monday, 7=Sunday? Or 0=Monday? 
        # Standard pandas dt.weekday is 0=Mon, 6=Sun. 
        # But user code says "weekday.isin([1,2,3,4,5])" for weekdays and "[6,7]" for weekend.
        # This implies 1-based indexing where 1=Mon, 7=Sun (ISO standard).
        # We will assume column 'weekday' follows this convention or check if we need to adjust.
        # If 'weekday' is not present, this will fail. We'll handle it outside or let it fail.
        
        if 'weekday' in valid.columns:
            weekday_enmo = valid[valid['weekday'].isin([1,2,3,4,5])]['enmo'].mean()
            weekend_enmo = valid[valid['weekday'].isin([6,7])]['enmo'].mean()
            weekend_diff = weekend_enmo - weekday_enmo # 正值代表週末動更多
        else:
            weekend_diff = np.nan

        return pd.Series({
            'L5': l5_val,
            'M10': m10_val,
            'RA': ra_val,
            'Night_AngleZ_Std': anglez_std_night,
            'Night_ENMO_Mean': enmo_mean_night,
            'ENMO_Skewness': enmo_skew,
            'ENMO_Kurtosis': enmo_kurt,
            'Weekend_Activity_Diff': weekend_diff
        })

    # 執行 GroupBy
    features_df = df.groupby('id').apply(get_id_features).reset_index()
    return features_df

def create_internet_addiction_features(df):
    """
    輸入原始 DataFrame，輸出針對「網路成癮預測」的結合特徵
    """
    # 確保有 hour 欄位
    if 'hour' not in df.columns:
        df['hour'] = (df['time_of_day'] / 1e9) / 3600

    def get_behavioral_features(group):
        # 0. 預處理：只取有配戴的期間
        valid = group[group['non-wear_flag'] == 0].copy()
        total_valid_points = len(valid)
        if total_valid_points == 0: return pd.Series()
        
        # --- 定義閾值 (Thresholds) ---
        # Light
        OUTDOOR_LUX = 1000  # 戶外光線門檻
        INDOOR_LUX_MAX = 500 # 室內光線上限
        DARK_LUX = 5        # 黑暗/睡眠門檻
        
        # ENMO (g)
        SLEEP_ENMO = 0.01   # 低於此視為靜止/睡眠
        SEDENTARY_ENMO = 0.2 # 低於此視為久坐/微動
        ACTIVE_ENMO = 0.5   # 高於此視為運動
        
        # --- Feature 1: Indoor_Sedentary_Ratio (室內久坐/疑似螢幕) ---
        # 邏輯：光線是室內的，動作是微小的 (排除睡覺)
        indoor_mask = (valid['light'] >= DARK_LUX) & (valid['light'] < INDOOR_LUX_MAX)
        sedentary_mask = (valid['enmo'] > SLEEP_ENMO) & (valid['enmo'] < SEDENTARY_ENMO)
        
        indoor_sedentary_points = len(valid[indoor_mask & sedentary_mask])
        indoor_sedentary_ratio = indoor_sedentary_points / total_valid_points
        indoor_sedentary_min = indoor_sedentary_points * 5 / 60 # 換算成分鐘數
        
        # --- Feature 2: Outdoor_Active_Duration (戶外運動) ---
        # 邏輯：光線很亮，且動作很大
        outdoor_mask = (valid['light'] >= OUTDOOR_LUX)
        active_mask = (valid['enmo'] >= ACTIVE_ENMO)
        
        outdoor_active_points = len(valid[outdoor_mask & active_mask])
        outdoor_active_min = outdoor_active_points * 5 / 60
        
        # --- Feature 3: Night_Screen_Proxy (夜間螢幕代理) ---
        # 邏輯：時間是半夜，有光(非全黑)，且沒在睡覺
        night_time_mask = (valid['hour'] >= 22) | (valid['hour'] < 5)
        # 注意：這裡 light > DARK_LUX (5) 是關鍵，全黑(0)通常是睡覺，大於0可能是手機光
        screen_light_mask = (valid['light'] > DARK_LUX) 
        awake_mask = (valid['enmo'] > SLEEP_ENMO)
        
        night_screen_points = len(valid[night_time_mask & screen_light_mask & awake_mask])
        night_screen_min = night_screen_points * 5 / 60
        
        # --- Feature 4: Gaming_Micro_Movement (電玩微動佔比) ---
        # 邏輯：在室內環境下，ENMO 落在 "0.01~0.1" 這個極窄區間的比例
        # 這個區間通常對應：手指動、身體微調、但不走動
        indoor_data = valid[indoor_mask]
        if len(indoor_data) > 0:
            micro_move_points = len(indoor_data[(indoor_data['enmo'] > 0.01) & (indoor_data['enmo'] < 0.1)])
            gaming_micro_ratio = micro_move_points / len(indoor_data)
        else:
            gaming_micro_ratio = 0
            
        # --- Feature 5: Screen_Obsession_Index (成癮指數) ---
        # 避免分母為 0，加 1
        obsession_index = indoor_sedentary_min / (outdoor_active_min + 1)

        return pd.Series({
            'Indoor_Sedentary_Ratio': indoor_sedentary_ratio,
            'Indoor_Sedentary_Minutes': indoor_sedentary_min,
            'Outdoor_Active_Minutes': outdoor_active_min,
            'Night_Screen_Proxy_Minutes': night_screen_min,
            'Gaming_Micro_Movement_Ratio': gaming_micro_ratio,
            'Screen_Obsession_Index': obsession_index
        })

    features_df = df.groupby('id').apply(get_behavioral_features).reset_index()
    return features_df

def main():
    parser = argparse.ArgumentParser(description="Generate advanced features from time series data.")
    parser.add_argument('--data_dir', type=str, default='data/series_train.parquet', help='Directory containing id folders')
    parser.add_argument('--output', type=str, default='advanced_features.csv', help='Output CSV file path')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of files to process (for testing)')
    args = parser.parse_args()

    # Search pattern: data_dir/id=*/data.csv
    search_pattern = os.path.join(args.data_dir, 'id=*', 'data.csv')
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"No files found matching {search_pattern}")
        return

    print(f"Found {len(files)} files.")
    
    if args.limit:
        files = files[:args.limit]
        print(f"Processing first {args.limit} files...")

    all_features = []
    
    for file_path in tqdm(files):
        try:
            # Extract ID from path
            # Path structure: .../id=XXXX/data.csv
            parent_dir = os.path.basename(os.path.dirname(file_path)) # id=XXXX
            if parent_dir.startswith('id='):
                subject_id = parent_dir.split('=')[1]
            else:
                # Fallback if structure is different
                subject_id = parent_dir
            
            df = pd.read_csv(file_path)
            df['id'] = subject_id
            
            # Run existing feature creation
            features1 = create_advanced_features(df)
            
            # Run new internet addiction feature creation
            features2 = create_internet_addiction_features(df)
            
            # Merge features for this ID
            # Both features1 and features2 should be 1-row DataFrames with 'id' column
            merged_id_features = pd.merge(features1, features2, on='id')
            
            all_features.append(merged_id_features)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    if all_features:
        final_df = pd.concat(all_features, ignore_index=True)
        # Ensure 'id' is the first column logic (it already should be due to reset_index in create_advanced_features)
        
        # rename columns if needed or just save
        final_df.to_csv(args.output, index=False)
        print(f"Saved features for {len(final_df)} IDs to {args.output}")
        print(final_df.head())
    else:
        print("No features generated.")

if __name__ == "__main__":
    main()
