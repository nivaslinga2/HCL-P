import io
import json
import time
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    roc_curve, roc_auc_score, precision_recall_curve, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score
)

def prepare_data_for_modeling(df_features, target_col):
    if df_features is None or target_col not in df_features.columns:
        return None, None
        
    leakage_cols = ['project_id', 'budget_deviation']
    drop_cols = [c for c in leakage_cols if c in df_features.columns]
    
    if target_col in ('monthly_spending', 'spending'):
        drop_cols.extend([c for c in ['spending', 'monthly_spending', 'loyal_customer', 'feat_budget_variance_ratio'] if c in df_features.columns])
    elif target_col == 'loyal_customer':
        drop_cols.extend([c for c in ['loyal_customer', 'feat_budget_variance_ratio'] if c in df_features.columns])
        
    X = df_features.drop(columns=list(set(drop_cols)), errors='ignore')
    X = X.select_dtypes(include=[np.number])
    y = df_features[target_col]
    
    valid_idx = X.notnull().all(axis=1) & y.notnull()
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]
    
    return X, y

def train_baseline_model(X, y, model_type='RandomForest', is_classification=True, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if is_classification else None
    )
    
    start_time = time.time()
    if is_classification:
        if model_type == 'LogisticRegression':
            model = LogisticRegression(max_iter=1000, random_state=random_state)
        else:
            model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_state)
            
        model.fit(X_train, y_train)
        fit_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob = y_pred
            
        metrics = {
            'Accuracy': float(accuracy_score(y_test, y_pred)),
            'Precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'Recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'F1': float(f1_score(y_test, y_pred, zero_division=0))
        }
    else:
        if model_type == 'LinearRegression':
            model = LinearRegression()
        else:
            model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=random_state)
            
        model.fit(X_train, y_train)
        fit_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        y_prob = None
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'R²': float(r2)
        }
        
    feat_importances = None
    coefficients = None
    if hasattr(model, 'feature_importances_'):
        feat_importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        coefficients = model.coef_.flatten()
        
    return {
        'model': model,
        'model_name': model_type,
        'is_classification': is_classification,
        'metrics': metrics,
        'fit_time': fit_time,
        'feature_names': list(X.columns),
        'feature_importances': feat_importances,
        'coefficients': coefficients,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

def run_hyperparameter_tuning(X, y, model_type='RandomForest', is_classification=True, param_grid=None, cv=3, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state,
        stratify=y if is_classification else None
    )
    
    if is_classification:
        if model_type == 'LogisticRegression':
            base_estimator = LogisticRegression(max_iter=1000, random_state=random_state)
            default_grid = {
                'C': [0.01, 0.1, 1.0, 10.0],
                'solver': ['lbfgs', 'liblinear']
            }
        else:
            base_estimator = RandomForestClassifier(random_state=random_state)
            default_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [4, 8, 12, None],
                'min_samples_split': [2, 5]
            }
        scoring = 'accuracy'
    else:
        if model_type == 'LinearRegression':
            base_estimator = Ridge(random_state=random_state)
            default_grid = {
                'alpha': [0.1, 1.0, 10.0, 50.0],
                'fit_intercept': [True, False]
            }
        else:
            base_estimator = RandomForestRegressor(random_state=random_state)
            default_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [4, 8, 12, None],
                'min_samples_split': [2, 5]
            }
        scoring = 'r2'
        
    grid = param_grid if param_grid else default_grid
    
    start_time = time.time()
    grid_search = GridSearchCV(
        estimator=base_estimator,
        param_grid=grid,
        scoring=scoring,
        cv=cv,
        n_jobs=1,
        return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    total_fit_time = time.time() - start_time
    
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    if is_classification:
        try:
            y_prob = best_model.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob = y_pred
        metrics = {
            'Accuracy': float(accuracy_score(y_test, y_pred)),
            'Precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'Recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'F1': float(f1_score(y_test, y_pred, zero_division=0))
        }
    else:
        y_prob = None
        metrics = {
            'MAE': float(mean_absolute_error(y_test, y_pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'R²': float(r2_score(y_test, y_pred))
        }
        
    results_df = pd.DataFrame(grid_search.cv_results_)
    cols_to_keep = ['params', 'mean_test_score', 'std_test_score', 'mean_fit_time', 'rank_test_score']
    for p in grid.keys():
        col_name = f"param_{p}"
        if col_name in results_df.columns:
            cols_to_keep.append(col_name)
    results_df = results_df[cols_to_keep].sort_values(by='rank_test_score').reset_index(drop=True)
    
    feat_importances = None
    coefficients = None
    if hasattr(best_model, 'feature_importances_'):
        feat_importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        coefficients = best_model.coef_.flatten()
        
    return {
        'best_model': best_model,
        'best_params': grid_search.best_params_,
        'best_cv_score': float(grid_search.best_score_),
        'metrics': metrics,
        'results_df': results_df,
        'total_fit_time': total_fit_time,
        'feature_names': list(X.columns),
        'feature_importances': feat_importances,
        'coefficients': coefficients,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob
    }

def compute_evaluation_diagnostics(y_test, y_pred, y_prob=None, is_classification=True):
    if is_classification:
        cm = confusion_matrix(y_test, y_pred)
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = roc_auc_score(y_test, y_prob)
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)
        else:
            fpr, tpr, roc_auc = [0, 1], [0, 1], 0.5
            precision_vals, recall_vals, pr_auc = [0, 1], [1, 0], 0.5
            
        return {
            'confusion_matrix': cm,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': float(roc_auc),
            'precision_curve': precision_vals,
            'recall_curve': recall_vals,
            'pr_auc': float(pr_auc)
        }
    else:
        residuals = y_test - y_pred
        return {
            'residuals': residuals,
            'actual': y_test,
            'predicted': y_pred
        }

def export_pipeline_bundle(model, feature_names, metrics, config_dict):
    bundle = {
        'model': model,
        'feature_names': feature_names,
        'metrics': metrics,
        'export_timestamp': datetime.now().isoformat(),
        'config': config_dict
    }
    joblib_buffer = io.BytesIO()
    joblib.dump(bundle, joblib_buffer)
    joblib_buffer.seek(0)
    
    json_str = json.dumps(bundle['config'], indent=2)
    return joblib_buffer.getvalue(), json_str
