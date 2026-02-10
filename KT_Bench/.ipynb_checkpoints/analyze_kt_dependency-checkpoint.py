import os
import glob
import pandas as pd
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# ==========================================
# 설정
# ==========================================
DATA_ROOT = 'data'
TARGET_FILE = 'preprocessed_data_test_saint.csv'

# X, Y (설명 변수)
feat_baseline = 'BASELINE'   # x: 통계적 유사도
feat_pfa = 'LR_sscwa'        # y: 누적 학습량 (PFA)

# Z (분석 대상 모델들)
# LR_i: IRT, LR_isscwatw: DAS3H, DKT2: DKT
target_models = ['SAINT', 'SAKT', 'DKT2', 'LR_i', 'LR_isscwatw']
modelid2name = {
    'SAINT': 'SAINT',
    'SAKT': 'SAKT',
    'DKT2': 'DKT',
    'LR_i': 'IRT',
    'LR_isscwatw': 'DAS3H',
}

def safe_logit(x, eps=1e-4):
    """0이나 1이 들어올 경우를 대비해 clipping 후 logit 변환"""
    x = np.clip(x, eps, 1 - eps)
    return logit(x)

def run_regression(df, dataset_name="Combined"):
    results = []
    
    # 필요한 컬럼이 다 있는지 확인 (Target 모델들은 부분적으로 있어도 됨)
    if feat_baseline not in df.columns or feat_pfa not in df.columns:
        print(f"[Skip] {dataset_name}: Baseline or PFA column missing.")
        return []

    # 1. Logit 변환
    logit_df = pd.DataFrame()
    logit_df['X_base'] = safe_logit(df[feat_baseline].values)
    logit_df['Y_pfa'] = safe_logit(df[feat_pfa].values)
    
    # 타겟 모델들은 존재하는 것만 Logit 변환 (NaN이 있으면 그대로 NaN으로 유지됨)
    available_targets = [t for t in target_models if t in df.columns]
    for t in available_targets:
        # 원본이 NaN이면 logit 변환해도 NaN임 (scipy logit 특성)
        logit_df[f'Z_{t}'] = safe_logit(df[t].values)

    # 2. 표준화 (Standardization) 준비 - 여기서 전체를 미리 하지 않음!
    scaler = StandardScaler()
    
    print(f"\n--- Analysis for: {dataset_name} (Samples: {len(df)}) ---")
    print(f"{'Target Model':<15} | {'Beta_Base(a)':<12} | {'Beta_PFA(b)':<12} | {'R^2':<8} | {'Depend_Ratio':<12}")
    print("-" * 75)

    local_res = []
    for t in available_targets:
        if t == 'LR_isscwatw' and dataset_name in ['assistments09', 'assistments15', 'spanish', 'statics']:
            continue
        # [핵심 수정] 현재 타겟 모델(t)에 대해, X와 Y 모두 NaN이 없는 행만 추출
        subset = logit_df[['X_base', 'Y_pfa', f'Z_{t}']].dropna()
        
        if len(subset) < 10:  # 데이터가 너무 적으면 스킵
            continue

        # 추출된 subset에 대해 다시 스케일링
        X_sub = subset[['X_base', 'Y_pfa']].values
        y_sub = subset[f'Z_{t}'].values.reshape(-1, 1)
        
        X_scaled = scaler.fit_transform(X_sub)
        y_scaled = scaler.fit_transform(y_sub).ravel()
        
        # 선형 회귀
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
    # 1. 개별 데이터셋 순회
    all_files = glob.glob(os.path.join(DATA_ROOT, '*', TARGET_FILE))
    if not all_files:
        print("데이터 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    all_data_frames = []
    
    # 모든 결과를 수집할 리스트 생성
    all_regression_results = []

    print(f"Found {len(all_files)} datasets.")

    for fpath in all_files:
        dataset_name = os.path.basename(os.path.dirname(fpath))
        try:
            df = pd.read_csv(fpath, sep='\t') 
            if len(df.columns) < 2: 
                 df = pd.read_csv(fpath, sep=',')
            
            # 개별 데이터셋 분석 및 결과 수집
            local_results = run_regression(df, dataset_name)
            all_regression_results.extend(local_results)
            
            cols_to_keep = [feat_baseline, feat_pfa] + [c for c in target_models if c in df.columns]
            all_data_frames.append(df[cols_to_keep].copy())
            
        except Exception as e:
            print(f"[Error] Failed to process {dataset_name}: {e}")

    # 2. 전체 데이터셋 통합 분석 (Global Fit)
    if all_data_frames:
        combined_df = pd.concat(all_data_frames, ignore_index=True)
        print("\n" + "="*80)
        print(f" GLOBAL ANALYSIS (Total Samples: {len(combined_df)})")
        print("="*80)
        
        global_results = run_regression(combined_df, "ALL_COMBINED")
        all_regression_results.extend(global_results)

    # 3. DKT 모델 전용 테이블 출력 (추가된 부분)
    target_dkt_key = 'DKT2' # 코드 내에서 DKT는 DKT2로 사용됨
    
    print("\n" + "#"*80)
    print(" DKT Model Specific Analysis Table")
    print("#"*80)
    print(f"{'Dataset':<20} | {'Beta_Base(a)':<12} | {'Beta_PFA(b)':<12} | {'R^2':<8} | {'Depend_Ratio':<12}")
    print("-" * 75)
    
    # DKT 결과만 필터링
    dkt_results = [res for res in all_regression_results if res['model'] == target_dkt_key]
    
    # 데이터셋 이름순 정렬 (ALL_COMBINED는 맨 뒤로)
    dkt_results.sort(key=lambda x: (x['dataset'] == 'ALL_COMBINED', x['dataset']))
    
    for res in dkt_results:
        print(f"{res['dataset']:<20} | {res['beta_baseline']:.2f}       | {res['beta_pfa']:.2f}       | {res['r2']:.2f}   | {res['dependency_ratio']:.2f}")

if __name__ == "__main__":
    main()