import sys
import pandas as pd
from utils.data_manager import (
    load_initial_data, compute_global_kpis, apply_global_filters,
    generate_missing_report, detect_outliers, clean_data, engineer_features
)
from utils.ml_pipeline import (
    prepare_data_for_modeling, train_baseline_model, run_hyperparameter_tuning,
    compute_evaluation_diagnostics, export_pipeline_bundle
)

def run_tests():
    df, src = load_initial_data()
    assert df is not None and len(df) > 0, "Dataset is empty"

    kpis = compute_global_kpis(df)
    assert 'total_records' in kpis and 'compliance_rate' in kpis

    miss_rep = generate_missing_report(df)
    assert len(miss_rep) > 0
    out_idx, out_sum = detect_outliers(df, ['spending', 'annual_income'], method='IQR')
    assert len(out_sum) > 0

    df_clean, c_sum = clean_data(df, missing_strategy='median', drop_dups=True, outlier_treatment='cap', outlier_cols=['spending'])
    assert df_clean is not None and df_clean.isnull().sum().sum() == 0

    df_features, tags, enc_maps, scalers = engineer_features(df_clean, encoding_strategy='One-Hot', scaling_strategy='Standard')
    assert df_features is not None
    assert any('🟩' in tag for tag in tags.values())

    X_cls, y_cls = prepare_data_for_modeling(df_features, 'loyal_customer')
    cls_res = train_baseline_model(X_cls, y_cls, model_type='RandomForest', is_classification=True)
    assert 'Accuracy' in cls_res['metrics']

    X_reg, y_reg = prepare_data_for_modeling(df_features, 'monthly_spending')
    reg_res = train_baseline_model(X_reg, y_reg, model_type='RandomForest', is_classification=False)
    assert 'R²' in reg_res['metrics']

    tune_res = run_hyperparameter_tuning(
        X_cls, y_cls, model_type='RandomForest', is_classification=True,
        param_grid={'n_estimators': [20, 50], 'max_depth': [4, 6]}, cv=2
    )
    assert tune_res['best_params'] is not None

    diag = compute_evaluation_diagnostics(cls_res['y_test'], cls_res['y_pred'], cls_res['y_prob'], is_classification=True)
    assert 'roc_auc' in diag

    jb_bytes, json_str = export_pipeline_bundle(
        cls_res['model'], cls_res['feature_names'], cls_res['metrics'],
        {'target': 'loyal_customer', 'model': 'RandomForest'}
    )
    assert len(jb_bytes) > 0 and len(json_str) > 0

    print("All tests passed.")

if __name__ == '__main__':
    run_tests()
