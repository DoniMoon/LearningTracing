import os
import glob
import pandas as pd
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

DATA_ROOT = 'data'
TARGET_FILE = 'preprocessed_data_test_saint.csv'

feat_baseline = 'BASELINE'  
feat_pfa = 'LR_sscwa'     

target_models = ['SAINT', 'SAKT', 'DKT2', 'LR_i', 'LR_isscwatw']
modelid2name = {
    'SAINT': 'SAINT',
    'SAKT': 'SAKT',
    'DKT2': 'DKT',
    'LR_i': 'IRT',
    'LR_isscwatw': 'DAS3H',
}

def safe_logit(x, eps=1e-4):
    x = np.clip(x, eps, 1 - eps)
    return logit(x)

def run_regression(df, dataset_name="Combined"):
    results = []
    
    if feat_baseline not in df.columns or feat_pfa not in df.columns:
        print(f"[Skip] {dataset_name}: Baseline or PFA column missing.")
        return []

    logit_df = pd.DataFrame()
    logit_df['X_base'] = safe_logit(df[feat_baseline].values)
    logit_df['Y_pfa'] = safe_logit(df[feat_pfa].values)
    
    available_targets = [t for t in target_models if t in df.columns]
    for t in available_targets:
        logit_df[f'Z_{t}'] = safe_logit(df[t].values)

    scaler = StandardScaler()
    
    print(f"\n--- Analysis for: {dataset_name} (Samples: {len(df)}) ---")
    print(f"{'Target Model':<15} | {'Beta_Base(a)':<12} | {'Beta_PFA(b)':<12} | {'R^2':<8} | {'Depend_Ratio':<12}")
    print("-" * 75)

    local_res = []
    for t in available_targets:
        if t == 'LR_isscwatw' and dataset_name in ['assistments09', 'assistments15', 'spanish', 'statics']:
            continue
              subset = logit_df[['X_base', 'Y_pfa', f'Z_{t}']].dropna()
        
        if len(subset) < 10: 
            continue

        X_sub = subset[['X_base', 'Y_pfa']].values
        y_sub = subset[f'Z_{t}'].values.reshape(-1, 1)
        
        X_scaled = scaler.fit_transform(X_sub)
        y_scaled = scaler.fit_transform(y_sub).ravel()
        
        reg = LinearRegression(fit_intercept=False) 
        reg.fit(X_scaled, y_scaled)
        
        a = reg.coef_[0]
        b = reg.coef_[1]
        r2 = reg.score(X_scaled, y_scaled)
        
        denom = abs(a) + abs(b)
        ratio = abs(a) / denom if denom > 0 else 0.0
        
        print(f"{modelid2name[t]:<15} | {a:.2f}       | {b:.2f}       | {r2:.2f}   | {ratio:.2f}")
        
        local_res.append({
            'dataset': dataset_name,
            'model': t,
            'beta_baseline': a,
            'beta_pfa': b,
            'r2': r2,
            'dependency_ratio': ratio
        })
        
    return local_res

def main():
    all_files = glob.glob(os.path.join(DATA_ROOT, '*', TARGET_FILE))
    if not all_files:
        print(f"Wrong File path: {TARGET_FILE}")
        return

    all_data_frames = []
    
    all_regression_results = []

    print(f"Found {len(all_files)} datasets.")

    for fpath in all_files:
        dataset_name = os.path.basename(os.path.dirname(fpath))
        try:
            df = pd.read_csv(fpath, sep='\t') 
            if len(df.columns) < 2: 
                 df = pd.read_csv(fpath, sep=',')
            
            local_results = run_regression(df, dataset_name)
            all_regression_results.extend(local_results)
            
            cols_to_keep = [feat_baseline, feat_pfa] + [c for c in target_models if c in df.columns]
            all_data_frames.append(df[cols_to_keep].copy())
            
        except Exception as e:
            print(f"[Error] Failed to process {dataset_name}: {e}")

    if all_data_frames:
        combined_df = pd.concat(all_data_frames, ignore_index=True)
        print("\n" + "="*80)
        print(f" GLOBAL ANALYSIS (Total Samples: {len(combined_df)})")
        print("="*80)
        
        global_results = run_regression(combined_df, "ALL_COMBINED")
        all_regression_results.extend(global_results)

    target_dkt_key = 'DKT2' 
    
    print("\n" + "#"*80)
    print(" DKT Model Specific Analysis Table") # DKT-specific analysis table
    print("#"*80)
    print(f"{'Dataset':<20} | {'Beta_Base(a)':<12} | {'Beta_PFA(b)':<12} | {'R^2':<8} | {'Depend_Ratio':<12}")
    print("-" * 75)
    
    dkt_results = [res for res in all_regression_results if res['model'] == target_dkt_key]
    
    dkt_results.sort(key=lambda x: (x['dataset'] == 'ALL_COMBINED', x['dataset']))
    
    for res in dkt_results:
        print(f"{res['dataset']:<20} | {res['beta_baseline']:.2f}       | {res['beta_pfa']:.2f}       | {res['r2']:.2f}   | {res['dependency_ratio']:.2f}")

if __name__ == "__main__":
    main()