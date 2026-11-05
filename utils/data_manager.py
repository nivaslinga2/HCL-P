import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'departmental_budget_deviations.csv')

def load_initial_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df, "Uploaded File"
        except Exception as e:
            return None, f"Error: {e}"
    elif os.path.exists(DEFAULT_DATA_PATH):
        return pd.read_csv(DEFAULT_DATA_PATH), "Default Dataset"
    else:
        from data.generate_dataset import generate_budget_dataset
        df = generate_budget_dataset(n_samples=1200, random_state=42)
        os.makedirs(os.path.dirname(DEFAULT_DATA_PATH), exist_ok=True)
        df.to_csv(DEFAULT_DATA_PATH, index=False)
        return df, "Generated Dataset"

def compute_global_kpis(df):
    if df is None or len(df) == 0:
        return {
            'total_records': {'label': 'Total Records', 'value': '0', 'delta': 'No Data', 'color': '#64748b'},
            'total_budget': {'label': 'Total Budget', 'value': '$0.00', 'delta': 'N/A', 'color': '#64748b'},
            'total_spending': {'label': 'Total Spending', 'value': '$0.00', 'delta': 'N/A', 'color': '#64748b'},
            'net_deviation': {'label': 'Net Variance', 'value': '$0.00', 'delta': '0.0%', 'color': '#64748b'},
            'compliance_rate': {'label': 'Compliance Rate', 'value': '0.0%', 'delta': 'Target: >70%', 'color': '#64748b'}
        }
    
    total_records = len(df)
    budget_col = 'annual_income' if 'annual_income' in df.columns else (df.select_dtypes(include=[np.number]).columns[0] if len(df.select_dtypes(include=[np.number]).columns) > 0 else None)
    spend_col = 'spending' if 'spending' in df.columns else ('monthly_spending' if 'monthly_spending' in df.columns else None)
    target_col = 'loyal_customer' if 'loyal_customer' in df.columns else None
    
    tot_budget = df[budget_col].sum() if budget_col else 0.0
    tot_spend = df[spend_col].sum() if spend_col else 0.0
    net_dev = tot_spend - tot_budget
    dev_pct = (net_dev / tot_budget * 100) if tot_budget > 0 else 0.0
    
    def fmt_curr(val):
        if abs(val) >= 1_000_000:
            return f"${val / 1_000_000:.2f}M"
        elif abs(val) >= 1_000:
            return f"${val / 1_000:.1f}k"
        return f"${val:.2f}"
    
    if target_col and target_col in df.columns:
        comp_rate = (df[target_col].dropna() == 1).mean() * 100
    elif budget_col and spend_col:
        comp_rate = (df[spend_col] <= df[budget_col] * 1.02).mean() * 100
    else:
        comp_rate = 50.0
        
    dev_color = '#10b981' if net_dev <= 0 else ('#f59e0b' if dev_pct < 10 else '#ef4444')
    comp_color = '#10b981' if comp_rate >= 65 else ('#f59e0b' if comp_rate >= 50 else '#ef4444')
    
    return {
        'total_records': {
            'label': 'Total Projects',
            'value': f"{total_records:,}",
            'delta': 'Active projects',
            'color': '#3b82f6'
        },
        'total_budget': {
            'label': 'Allocated Budget',
            'value': fmt_curr(tot_budget),
            'delta': f"Avg: {fmt_curr(df[budget_col].mean()) if budget_col else '$0'}",
            'color': '#6366f1'
        },
        'total_spending': {
            'label': 'Actual Spending',
            'value': fmt_curr(tot_spend),
            'delta': f"Avg: {fmt_curr(df[spend_col].mean()) if spend_col else '$0'}",
            'color': '#8b5cf6'
        },
        'net_deviation': {
            'label': 'Net Variance',
            'value': fmt_curr(net_dev),
            'delta': f"{'+' if net_dev > 0 else ''}{dev_pct:.1f}% vs budget",
            'color': dev_color
        },
        'compliance_rate': {
            'label': 'Compliance Rate',
            'value': f"{comp_rate:.1f}%",
            'delta': 'Target: >= 65%',
            'color': comp_color
        }
    }

def apply_global_filters(df, selected_depts, selected_cities, selected_quarters, budget_range):
    if df is None or len(df) == 0:
        return df
    
    filtered = df.copy()
    if 'department' in filtered.columns and selected_depts and 'All' not in selected_depts:
        filtered = filtered[filtered['department'].isin(selected_depts)]
        
    if 'city' in filtered.columns and selected_cities and 'All' not in selected_cities:
        filtered = filtered[filtered['city'].isin(selected_cities)]
        
    if 'fiscal_quarter' in filtered.columns and selected_quarters and 'All' not in selected_quarters:
        filtered = filtered[filtered['fiscal_quarter'].isin(selected_quarters)]
        
    if 'annual_income' in filtered.columns and budget_range:
        filtered = filtered[(filtered['annual_income'] >= budget_range[0]) & (filtered['annual_income'] <= budget_range[1])]
        
    return filtered

def generate_missing_report(df):
    if df is None:
        return pd.DataFrame()
    
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df) * 100).round(2)
    dtypes = df.dtypes.astype(str)
    
    return pd.DataFrame({
        'Column': df.columns,
        'Data Type': [dtypes[c] for c in df.columns],
        'Missing Count': [missing_count[c] for c in df.columns],
        'Missing %': [missing_pct[c] for c in df.columns],
        'Status': ['Clean' if missing_count[c] == 0 else ('Minor' if missing_pct[c] < 5 else 'Significant') for c in df.columns]
    }).sort_values(by='Missing Count', ascending=False).reset_index(drop=True)

def detect_outliers(df, columns, method='IQR', k=1.5, z_thresh=3.0):
    if df is None or len(df) == 0 or not columns:
        return {}, pd.DataFrame()
    
    outlier_indices = {}
    summary_list = []
    
    for col in columns:
        if col not in df.columns or not np.issubdtype(df[col].dropna().dtype, np.number):
            continue
            
        series = df[col].dropna()
        if len(series) == 0:
            continue
            
        if method == 'IQR':
            q25 = series.quantile(0.25)
            q75 = series.quantile(0.75)
            iqr = q75 - q25
            lower_bound = q25 - (k * iqr)
            upper_bound = q75 + (k * iqr)
            mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            col_outliers = df[mask].index.tolist()
        else:
            mean = series.mean()
            std = series.std()
            if std == 0:
                lower_bound, upper_bound = mean, mean
                col_outliers = []
            else:
                z_scores = (df[col] - mean) / std
                mask = z_scores.abs() > z_thresh
                lower_bound = mean - (z_thresh * std)
                upper_bound = mean + (z_thresh * std)
                col_outliers = df[mask].index.tolist()
                
        outlier_indices[col] = col_outliers
        count = len(col_outliers)
        pct = round((count / len(df)) * 100, 2)
        summary_list.append({
            'Column': col,
            'Detection Method': f"{method} ({'k=' + str(k) if method == 'IQR' else '|z|>' + str(z_thresh)})",
            'Outlier Count': count,
            'Outlier %': pct,
            'Lower Threshold': round(lower_bound, 2),
            'Upper Threshold': round(upper_bound, 2)
        })
        
    summary_df = pd.DataFrame(summary_list).sort_values(by='Outlier Count', ascending=False).reset_index(drop=True)
    return outlier_indices, summary_df

def clean_data(df, missing_strategy='median', drop_dups=True, outlier_treatment='none', outlier_cols=None, iqr_k=1.5):
    if df is None:
        return None, {}
        
    df_clean = df.copy()
    rows_initial = len(df_clean)
    missing_initial = df_clean.isnull().sum().sum()
    dups_initial = df_clean.duplicated().sum()
    
    dups_removed = 0
    if drop_dups:
        before_dup = len(df_clean)
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        dups_removed = before_dup - len(df_clean)
        
    if missing_strategy == 'drop':
        df_clean = df_clean.dropna().reset_index(drop=True)
    elif missing_strategy in ['mean', 'median', 'mode']:
        num_cols = df_clean.select_dtypes(include=[np.number]).columns
        cat_cols = df_clean.select_dtypes(exclude=[np.number]).columns
        
        for col in num_cols:
            if df_clean[col].isnull().any():
                if missing_strategy == 'mean':
                    val = df_clean[col].mean()
                elif missing_strategy == 'median':
                    val = df_clean[col].median()
                else:
                    mode_s = df_clean[col].mode()
                    val = mode_s[0] if len(mode_s) > 0 else 0
                df_clean[col] = df_clean[col].fillna(val)
                
        for col in cat_cols:
            if df_clean[col].isnull().any():
                mode_s = df_clean[col].mode()
                val = mode_s[0] if len(mode_s) > 0 else 'Unknown'
                df_clean[col] = df_clean[col].fillna(val)
    elif missing_strategy == 'custom_zero':
        df_clean = df_clean.fillna(0)
        
    if 'spending' in df_clean.columns and 'annual_income' in df_clean.columns:
        if 'monthly_spending' in df_clean.columns:
            df_clean['monthly_spending'] = (df_clean['spending'] / 12).round(2)
        if 'budget_deviation' in df_clean.columns:
            df_clean['budget_deviation'] = (df_clean['spending'] - df_clean['annual_income']).round(2)
        if 'loyal_customer' in df_clean.columns:
            df_clean['loyal_customer'] = np.where(df_clean['spending'] <= (df_clean['annual_income'] * 1.02), 1, 0)
            
    outliers_treated_count = 0
    if outlier_treatment != 'none' and outlier_cols:
        for col in outlier_cols:
            if col in df_clean.columns and np.issubdtype(df_clean[col].dtype, np.number):
                series = df_clean[col]
                q25 = series.quantile(0.25)
                q75 = series.quantile(0.75)
                iqr = q75 - q25
                lb = q25 - (iqr_k * iqr)
                ub = q75 + (iqr_k * iqr)
                
                outlier_mask = (df_clean[col] < lb) | (df_clean[col] > ub)
                outliers_treated_count += outlier_mask.sum()
                
                if outlier_treatment == 'cap':
                    df_clean[col] = df_clean[col].clip(lower=lb, upper=ub)
                elif outlier_treatment == 'remove':
                    df_clean = df_clean[~outlier_mask].reset_index(drop=True)
                    
    missing_final = df_clean.isnull().sum().sum()
    rows_final = len(df_clean)
    
    summary = {
        'rows_before': rows_initial,
        'rows_after': rows_final,
        'rows_delta': rows_final - rows_initial,
        'missing_before': int(missing_initial),
        'missing_after': int(missing_final),
        'missing_reduction': int(missing_initial - missing_final),
        'duplicates_removed': int(dups_removed),
        'outliers_treated': int(outliers_treated_count),
        'outlier_strategy': outlier_treatment
    }
    
    return df_clean, summary

def engineer_features(df, encoding_strategy='One-Hot', scaling_strategy='Standard', scaling_cols=None):
    if df is None:
        return None, {}, {}, {}
        
    df_feat = df.copy()
    feature_tags = {}
    
    for c in df_feat.columns:
        feature_tags[c] = "⬜ Original"
        
    if 'spending' in df_feat.columns and 'annual_income' in df_feat.columns:
        df_feat['feat_budget_variance_ratio'] = (df_feat['spending'] / (df_feat['annual_income'] + 1e-5)).round(4)
        feature_tags['feat_budget_variance_ratio'] = "🟩 Engineered"
        
    if 'spending' in df_feat.columns and 'headcount' in df_feat.columns:
        df_feat['feat_spending_per_headcount'] = (df_feat['spending'] / (df_feat['headcount'] + 1)).round(2)
        feature_tags['feat_spending_per_headcount'] = "🟩 Engineered"
        
    if 'approval_days' in df_feat.columns and 'online_score' in df_feat.columns:
        df_feat['feat_procurement_friction'] = (df_feat['approval_days'] / (df_feat['online_score'] + 1.0)).round(4)
        feature_tags['feat_procurement_friction'] = "🟩 Engineered"
        
    if 'risk_rating' in df_feat.columns and 'spending' in df_feat.columns:
        risk_weight_map = {'Low': 1.0, 'Medium': 1.25, 'High': 1.6, 'Critical': 2.0}
        numeric_risk = df_feat['risk_rating'].map(risk_weight_map).fillna(1.0)
        df_feat['feat_risk_weighted_spend'] = (df_feat['spending'] * numeric_risk).round(2)
        feature_tags['feat_risk_weighted_spend'] = "🟩 Engineered"

    if 'fiscal_quarter' in df_feat.columns:
        q_map = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
        q_num = df_feat['fiscal_quarter'].map(q_map).fillna(1)
        df_feat['feat_quarter_sin'] = np.sin(2 * np.pi * q_num / 4.0).round(4)
        df_feat['feat_quarter_cos'] = np.cos(2 * np.pi * q_num / 4.0).round(4)
        feature_tags['feat_quarter_sin'] = "🟩 Engineered"
        feature_tags['feat_quarter_cos'] = "🟩 Engineered"
        
    cat_candidates = ['department', 'project_category', 'city', 'gender', 'cost_center_tier', 'risk_rating', 'fiscal_quarter']
    cat_cols_to_encode = [c for c in cat_candidates if c in df_feat.columns]
    
    encoding_maps = {}
    if encoding_strategy == 'One-Hot':
        for col in cat_cols_to_encode:
            dummies = pd.get_dummies(df_feat[col], prefix=col, drop_first=False, dtype=int)
            for d_col in dummies.columns:
                df_feat[d_col] = dummies[d_col]
                feature_tags[d_col] = "🟦 Encoded"
            encoding_maps[col] = list(dummies.columns)
            df_feat = df_feat.drop(columns=[col])
    else:
        for col in cat_cols_to_encode:
            categories = sorted(df_feat[col].dropna().unique().tolist())
            cat_map = {cat: i for i, cat in enumerate(categories)}
            ord_col = f"{col}_ordinal"
            df_feat[ord_col] = df_feat[col].map(cat_map).fillna(0).astype(int)
            feature_tags[ord_col] = "🟦 Encoded"
            encoding_maps[col] = cat_map
            df_feat = df_feat.drop(columns=[col])
            
    if 'project_id' in df_feat.columns:
        df_feat = df_feat.drop(columns=['project_id'])
        
    if scaling_cols is None:
        scaling_cols = [c for c in ['age', 'headcount', 'annual_income', 'spending', 'monthly_spending', 'online_score', 'approval_days', 'feat_budget_variance_ratio', 'feat_spending_per_headcount', 'feat_procurement_friction', 'feat_risk_weighted_spend'] if c in df_feat.columns]
        
    scalers_dict = {}
    if scaling_strategy != 'None' and len(scaling_cols) > 0:
        if scaling_strategy == 'Standard':
            scaler = StandardScaler()
        elif scaling_strategy == 'MinMax':
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()
            
        scaled_vals = scaler.fit_transform(df_feat[scaling_cols])
        for i, col in enumerate(scaling_cols):
            scaled_name = f"{col}_scaled"
            df_feat[scaled_name] = np.round(scaled_vals[:, i], 4)
            feature_tags[scaled_name] = "🟧 Scaled"
        scalers_dict[scaling_strategy] = scaler
        
    return df_feat, feature_tags, encoding_maps, scalers_dict
