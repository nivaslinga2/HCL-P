import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st

from utils.data_manager import (
    load_initial_data, compute_global_kpis, apply_global_filters,
    generate_missing_report, detect_outliers, clean_data, engineer_features
)
from utils.visualizations import (
    plot_histogram_age, plot_histogram_annual_income, plot_box_spending_by_gender,
    plot_bar_city_counts, plot_missing_percentage, plot_box_per_column,
    plot_violin_distribution, plot_correlation_heatmap, plot_scatter_income_vs_spending,
    plot_histogram_overlay_before_after, plot_box_side_by_side, plot_bar_missing_reduction,
    plot_histogram_raw_vs_scaled, plot_strip_encoded_vs_target, plot_scatter_engineered_vs_target,
    plot_feature_importances, plot_model_coefficients, plot_y_true_vs_y_pred,
    plot_metric_across_params, plot_scatter_metric_vs_param,
    plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve,
    plot_predicted_vs_actual_scatter, plot_residuals_histogram, plot_residuals_vs_fitted,
    plot_baseline_vs_tuned_bar, plot_segments_treemap, plot_kpi_trends_over_runs,
    plot_churn_rate_by_city, plot_density_heatmap_score_vs_spending
)
from utils.ml_pipeline import (
    prepare_data_for_modeling, train_baseline_model, run_hyperparameter_tuning,
    compute_evaluation_diagnostics, export_pipeline_bundle
)

st.set_page_config(
    page_title="Budget Deviation Predictor",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

css_path = os.path.join(os.path.dirname(__file__), 'styles', 'theme.css')
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

if 'df_raw' not in st.session_state:
    st.session_state['df_raw'] = None
if 'data_source_name' not in st.session_state:
    st.session_state['data_source_name'] = "None"
if 'df_clean' not in st.session_state:
    st.session_state['df_clean'] = None
if 'df_features' not in st.session_state:
    st.session_state['df_features'] = None
if 'feature_tags' not in st.session_state:
    st.session_state['feature_tags'] = {}
if 'cleaning_summary' not in st.session_state:
    st.session_state['cleaning_summary'] = None
if 'baseline_results' not in st.session_state:
    st.session_state['baseline_results'] = None
if 'tuned_results' not in st.session_state:
    st.session_state['tuned_results'] = None
if 'current_model' not in st.session_state:
    st.session_state['current_model'] = None
if 'pipeline_run_history' not in st.session_state:
    st.session_state['pipeline_run_history'] = []
if 'nav_target' not in st.session_state:
    st.session_state['nav_target'] = "Dashboard"

if st.session_state['df_raw'] is None:
    df_init, src_name = load_initial_data()
    st.session_state['df_raw'] = df_init
    st.session_state['data_source_name'] = src_name

def render_kpi_cards(kpi_dict):
    cols = st.columns(5)
    keys = ['total_records', 'total_budget', 'total_spending', 'net_deviation', 'compliance_rate']
    for idx, key in enumerate(keys):
        kpi = kpi_dict.get(key, {})
        with cols[idx]:
            accent = kpi.get('color', '#3b82f6')
            st.markdown(f"""
            <div class="kpi-card" style="--kpi-accent: {accent};">
                <div class="kpi-title">{kpi.get('label', 'Metric')}</div>
                <div class="kpi-value">{kpi.get('value', '-')}</div>
                <div class="kpi-subtext" style="color: {accent};">
                    <span>●</span> {kpi.get('delta', '')}
                </div>
            </div>
            """, unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
        <h2 style="margin: 0; color: #60a5fa; font-size: 1.4rem; font-weight: 700;">Budget Predictor</h2>
        <p style="margin: 0.2rem 0 0 0; font-size: 0.8rem; color: #94a3b8;">Department Financial Variance</p>
    </div>
    """, unsafe_allow_html=True)
    
    pages = [
        "Dashboard",
        "Dataset & EDA",
        "Missing & Outliers",
        "Data Cleaning",
        "Feature Engineering",
        "Baseline Model",
        "Hyperparameter Tuning",
        "Model Evaluation",
        "Pipeline Runner"
    ]
    
    default_index = pages.index(st.session_state['nav_target']) if st.session_state['nav_target'] in pages else 0
    selected_page = st.selectbox("Navigation", pages, index=default_index, key="main_navigation_select")
    st.session_state['nav_target'] = selected_page
    
    st.markdown("---")
    
    st.markdown("#### Data Source")
    uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
    if uploaded_file is not None:
        df_up, src = load_initial_data(uploaded_file)
        if df_up is not None:
            st.session_state['df_raw'] = df_up
            st.session_state['data_source_name'] = uploaded_file.name
            st.success(f"Loaded {len(df_up)} rows")
            
    st.caption(f"Current: `{st.session_state['data_source_name']}`")
    
    filters_enabled = st.session_state['df_raw'] is not None
    with st.expander("Filters", expanded=True):
        if not filters_enabled:
            st.info("Upload dataset to activate filters.")
            selected_depts = []
            selected_cities = []
            selected_quarters = []
            budget_range = (0, 1000000)
        else:
            df_cur = st.session_state['df_raw']
            
            if 'department' in df_cur.columns:
                dept_opts = ['All'] + sorted(df_cur['department'].dropna().unique().tolist())
                selected_depts = st.multiselect("Department", dept_opts, default=['All'])
            else:
                selected_depts = ['All']
                
            if 'city' in df_cur.columns:
                city_opts = ['All'] + sorted(df_cur['city'].dropna().unique().tolist())
                selected_cities = st.multiselect("City", city_opts, default=['All'])
            else:
                selected_cities = ['All']
                
            if 'fiscal_quarter' in df_cur.columns:
                q_opts = ['All'] + sorted(df_cur['fiscal_quarter'].dropna().unique().tolist())
                selected_quarters = st.multiselect("Quarter", q_opts, default=['All'])
            else:
                selected_quarters = ['All']
                
            if 'annual_income' in df_cur.columns:
                b_min = float(df_cur['annual_income'].min())
                b_max = float(df_cur['annual_income'].max())
                budget_range = st.slider("Budget ($)", min_value=b_min, max_value=b_max, value=(b_min, b_max), format="$%d")
            else:
                budget_range = None
                
    st.markdown("---")
    
    target_mode = st.radio("Prediction Task", ["Classification (loyal_customer)", "Regression (monthly_spending)"], index=0)
    is_classification_target = "Classification" in target_mode
    active_target_col = 'loyal_customer' if is_classification_target else 'monthly_spending'
    
    st.markdown("---")
    
    if st.button("Reset State", use_container_width=True):
        st.session_state['df_clean'] = None
        st.session_state['df_features'] = None
        st.session_state['baseline_results'] = None
        st.session_state['tuned_results'] = None
        st.session_state['current_model'] = None
        st.session_state['pipeline_run_history'] = []
        st.success("App state reset.")
        st.rerun()

df_raw = st.session_state['df_raw']
df_filtered = apply_global_filters(df_raw, selected_depts, selected_cities, selected_quarters, budget_range)
kpis = compute_global_kpis(df_filtered)

if selected_page == "Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>Budget Deviation Predictor</h1>
        <p>Analyze department spending, identify variance risks, and forecast budget compliance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    render_kpi_cards(kpis)
    
    st.markdown("""
    <div class="section-banner">
        <h3>Quick Navigation</h3>
        <p>Select a stage to explore or run the workflow</p>
    </div>
    """, unsafe_allow_html=True)
    
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    with nav_col1:
        if st.button("Dataset & EDA", use_container_width=True):
            st.session_state['nav_target'] = "Dataset & EDA"
            st.rerun()
        if st.button("Missing & Outliers", use_container_width=True):
            st.session_state['nav_target'] = "Missing & Outliers"
            st.rerun()
        if st.button("Data Cleaning", use_container_width=True):
            st.session_state['nav_target'] = "Data Cleaning"
            st.rerun()
            
    with nav_col2:
        if st.button("Feature Engineering", use_container_width=True):
            st.session_state['nav_target'] = "Feature Engineering"
            st.rerun()
        if st.button("Baseline Model", use_container_width=True):
            st.session_state['nav_target'] = "Baseline Model"
            st.rerun()
        if st.button("Hyperparameter Tuning", use_container_width=True):
            st.session_state['nav_target'] = "Hyperparameter Tuning"
            st.rerun()
            
    with nav_col3:
        if st.button("Model Evaluation", use_container_width=True):
            st.session_state['nav_target'] = "Model Evaluation"
            st.rerun()
        if st.button("Pipeline Runner", use_container_width=True):
            st.session_state['nav_target'] = "Pipeline Runner"
            st.rerun()
        if st.button("Download Filtered CSV", use_container_width=True):
            if df_filtered is not None:
                st.download_button(
                    label="Save CSV",
                    data=df_filtered.to_csv(index=False).encode('utf-8'),
                    file_name="department_budget_filtered.csv",
                    mime="text/csv"
                )
                
    st.markdown("<br>", unsafe_allow_html=True)
    
    g_col1, g_col2 = st.columns(2)
    with g_col1:
        st.plotly_chart(plot_churn_rate_by_city(df_filtered), use_container_width=True)
        st.plotly_chart(plot_kpi_trends_over_runs(st.session_state['pipeline_run_history']), use_container_width=True)
    with g_col2:
        st.plotly_chart(plot_box_spending_by_gender(df_filtered), use_container_width=True)
        st.plotly_chart(plot_density_heatmap_score_vs_spending(df_filtered), use_container_width=True)

elif selected_page == "Dataset & EDA":
    st.markdown("""
    <div class="main-header">
        <h1>Dataset Exploration & EDA</h1>
        <p>Inspect dataset schema, summary statistics, and feature distributions.</p>
    </div>
    """, unsafe_allow_html=True)
    
    render_kpi_cards(kpis)
    
    tab_preview, tab_summary, tab_info, tab_charts = st.tabs([
        "Preview", "Summary Statistics", "Data Types & Shape", "Charts"
    ])
    
    with tab_preview:
        n_rows = st.slider("Preview rows", min_value=5, max_value=50, value=10)
        st.dataframe(df_filtered.head(n_rows), use_container_width=True)
        with st.expander("Tail Rows"):
            st.dataframe(df_filtered.tail(n_rows), use_container_width=True)
            
    with tab_summary:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("#### Numerical Attributes")
            st.dataframe(df_filtered.describe().round(2).T, use_container_width=True)
        with col_s2:
            st.markdown("#### Categorical Attributes")
            st.dataframe(df_filtered.describe(include=['object']).T, use_container_width=True)
            
    with tab_info:
        info_c1, info_c2 = st.columns(2)
        with info_c1:
            st.metric("Total Rows", f"{df_filtered.shape[0]:,}")
            st.metric("Total Columns", df_filtered.shape[1])
            st.metric("Memory Usage", f"{df_filtered.memory_usage(deep=True).sum() / 1024:.1f} KB")
        with info_c2:
            dtype_df = pd.DataFrame({
                'Column': df_filtered.columns,
                'Type': df_filtered.dtypes.astype(str),
                'Non-Null': df_filtered.notnull().sum().values,
                'Unique': [df_filtered[c].nunique() for c in df_filtered.columns]
            })
            st.dataframe(dtype_df, use_container_width=True)
            
    with tab_charts:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_histogram_age(df_filtered), use_container_width=True)
            st.plotly_chart(plot_box_spending_by_gender(df_filtered), use_container_width=True)
        with c2:
            st.plotly_chart(plot_histogram_annual_income(df_filtered), use_container_width=True)
            st.plotly_chart(plot_bar_city_counts(df_filtered, top_n=10), use_container_width=True)

elif selected_page == "Missing & Outliers":
    st.markdown("""
    <div class="main-header">
        <h1>Missing Values & Outlier Diagnostics</h1>
        <p>Inspect missing values, outlier thresholds, and feature correlations.</p>
    </div>
    """, unsafe_allow_html=True)
    
    render_kpi_cards(kpis)
    missing_report = generate_missing_report(df_filtered)
    
    with st.sidebar:
        st.markdown("### Outlier Detection")
        outlier_method = st.radio("Method", ["IQR", "Z-score"], index=0)
        if outlier_method == "IQR":
            iqr_k = st.selectbox("IQR Multiplier (k)", [1.5, 3.0], index=0)
            z_thresh = 3.0
        else:
            iqr_k = 1.5
            z_thresh = st.slider("Z-Score Threshold", min_value=2.0, max_value=4.0, value=3.0, step=0.1)
            
        num_candidates = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
        sel_outlier_cols = st.multiselect("Columns", num_candidates, default=num_candidates[:4])
        
    outlier_idx, outlier_summary = detect_outliers(
        df_filtered, sel_outlier_cols, method=outlier_method, k=iqr_k, z_thresh=z_thresh
    )
    
    tab_miss, tab_out, tab_diag = st.tabs(["Missing Values", "Outliers", "Diagnostics"])
    
    with tab_miss:
        st.dataframe(missing_report, use_container_width=True)
        csv_missing = missing_report.to_csv(index=False).encode('utf-8')
        st.download_button("Download Missing Values Report", data=csv_missing, file_name="missing_report.csv", mime="text/csv")
        st.plotly_chart(plot_missing_percentage(missing_report), use_container_width=True)
        
    with tab_out:
        if not outlier_summary.empty:
            st.dataframe(outlier_summary, use_container_width=True)
        else:
            st.info("No numerical columns selected.")
            
        if len(sel_outlier_cols) > 0:
            st.plotly_chart(plot_box_per_column(df_filtered, sel_outlier_cols), use_container_width=True)
            
    with tab_diag:
        d_col1, d_col2 = st.columns(2)
        with d_col1:
            sel_violin_col = st.selectbox("Violin Plot Column", num_candidates, index=min(2, len(num_candidates)-1))
            st.plotly_chart(plot_violin_distribution(df_filtered, sel_violin_col), use_container_width=True)
            st.plotly_chart(plot_scatter_income_vs_spending(df_filtered), use_container_width=True)
        with d_col2:
            st.plotly_chart(plot_correlation_heatmap(df_filtered), use_container_width=True)

elif selected_page == "Data Cleaning":
    st.markdown("""
    <div class="main-header">
        <h1>Data Cleaning & Preprocessing</h1>
        <p>Handle missing values, duplicates, and outlier capping or removal.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("Cleaning Settings", expanded=True):
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            missing_strat = st.selectbox("Missing Values", ["median", "mean", "mode", "drop", "custom_zero"], index=0)
            drop_dups = st.checkbox("Drop Duplicates", value=True)
        with col_c2:
            outlier_strat = st.selectbox("Outliers", ["none", "cap", "remove"], index=1)
            iqr_k_clean = st.select_slider("IQR Multiplier (k)", options=[1.5, 2.0, 2.5, 3.0], value=1.5)
        with col_c3:
            num_cols_avail = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
            treat_cols = st.multiselect(
                "Columns for Outlier Handling",
                num_cols_avail,
                default=[c for c in ['spending', 'annual_income', 'approval_days'] if c in num_cols_avail]
            )
            btn_clean = st.button("Apply Cleaning", use_container_width=True, type="primary")

    if btn_clean or st.session_state['df_clean'] is None:
        cleaned_df, summary = clean_data(
            df_filtered,
            missing_strategy=missing_strat,
            drop_dups=drop_dups,
            outlier_treatment=outlier_strat,
            outlier_cols=treat_cols,
            iqr_k=iqr_k_clean
        )
        st.session_state['df_clean'] = cleaned_df
        st.session_state['cleaning_summary'] = summary
        st.success("Cleaned dataset saved.")

    df_clean = st.session_state['df_clean']
    summary = st.session_state['cleaning_summary']

    if summary:
        m_c1, m_c2, m_c3, m_c4 = st.columns(4)
        m_c1.metric("Rows Cleaned", f"{summary['rows_after']:,}", delta=f"{summary['rows_delta']} rows")
        m_c2.metric("Missing Handled", f"{summary['missing_after']}", delta=f"-{summary['missing_reduction']}", delta_color="inverse")
        m_c3.metric("Duplicates Dropped", f"{summary['duplicates_removed']}")
        m_c4.metric("Outliers Handled", f"{summary['outliers_treated']}", delta=summary['outlier_strategy'])

    tab_before, tab_after, tab_diff, tab_vcompare = st.tabs([
        "Before", "After", "Summary", "Comparison"
    ])

    with tab_before:
        st.dataframe(df_filtered.head(15), use_container_width=True)
    with tab_after:
        st.dataframe(df_clean.head(15), use_container_width=True)
        csv_clean_bytes = df_clean.to_csv(index=False).encode('utf-8')
        st.download_button("Download Cleaned CSV", data=csv_clean_bytes, file_name="df_clean.csv", mime="text/csv")
    with tab_diff:
        diff_df = pd.DataFrame([
            {'Metric': 'Rows', 'Before': summary['rows_before'], 'After': summary['rows_after'], 'Change': summary['rows_delta']},
            {'Metric': 'Missing Values', 'Before': summary['missing_before'], 'After': summary['missing_after'], 'Change': -summary['missing_reduction']},
            {'Metric': 'Duplicates', 'Before': summary['duplicates_removed'], 'After': 0, 'Change': -summary['duplicates_removed']},
            {'Metric': 'Outliers Handled', 'Before': summary['outliers_treated'], 'After': 0 if summary['outlier_strategy'] != 'none' else summary['outliers_treated'], 'Change': summary['outlier_strategy']}
        ])
        st.dataframe(diff_df, use_container_width=True)
    with tab_vcompare:
        cmp_col = st.selectbox("Variable", [c for c in ['spending', 'annual_income', 'approval_days', 'online_score'] if c in df_clean.columns])
        vc1, vc2 = st.columns(2)
        with vc1:
            st.plotly_chart(plot_histogram_overlay_before_after(df_filtered, df_clean, cmp_col), use_container_width=True)
            st.plotly_chart(plot_bar_missing_reduction(summary['missing_before'], summary['missing_after']), use_container_width=True)
        with vc2:
            st.plotly_chart(plot_box_side_by_side(df_filtered, df_clean, cmp_col), use_container_width=True)

elif selected_page == "Feature Engineering":
    st.markdown("""
    <div class="main-header">
        <h1>Feature Engineering</h1>
        <p>Derive financial variance ratios, encode categories, and scale numerical inputs.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state['df_clean'] is None:
        st.warning("Please clean data first or run auto-clean.")
        if st.button("Auto-Clean Dataset"):
            st.session_state['df_clean'], st.session_state['cleaning_summary'] = clean_data(df_filtered)
            st.rerun()
        st.stop()
        
    df_clean = st.session_state['df_clean']
    
    with st.expander("Settings", expanded=True):
        fe_col1, fe_col2, fe_col3 = st.columns(3)
        with fe_col1:
            enc_choice = st.selectbox("Categorical Encoding", ["One-Hot", "Ordinal"], index=0)
        with fe_col2:
            scale_choice = st.selectbox("Scaling", ["Standard", "MinMax", "Robust", "None"], index=0)
        with fe_col3:
            btn_engineer = st.button("Transform Features", use_container_width=True, type="primary")
            
    if btn_engineer or st.session_state['df_features'] is None:
        df_feat, tags, enc_maps, scalers = engineer_features(
            df_clean, encoding_strategy=enc_choice, scaling_strategy=scale_choice
        )
        st.session_state['df_features'] = df_feat
        st.session_state['feature_tags'] = tags
        st.success("Feature set ready.")

    df_features = st.session_state['df_features']
    tags = st.session_state['feature_tags']

    tag_counts = {'🟩 Engineered': 0, '🟦 Encoded': 0, '🟧 Scaled': 0, '⬜ Original': 0}
    for t in tags.values():
        tag_counts[t] = tag_counts.get(t, 0) + 1
        
    t_c1, t_c2, t_c3, t_c4 = st.columns(4)
    t_c1.markdown(f'<div class="badge-engineered">🟩 Engineered: {tag_counts["🟩 Engineered"]}</div>', unsafe_allow_html=True)
    t_c2.markdown(f'<div class="badge-encoded">🟦 Encoded: {tag_counts["🟦 Encoded"]}</div>', unsafe_allow_html=True)
    t_c3.markdown(f'<div class="badge-scaled">🟧 Scaled: {tag_counts["🟧 Scaled"]}</div>', unsafe_allow_html=True)
    t_c4.markdown(f'<span style="color:#94a3b8; font-weight:600;">⬜ Original: {tag_counts["⬜ Original"]}</span>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("Preview Feature Matrix"):
        st.dataframe(df_features.head(10), use_container_width=True)

    f_chart_col1, f_chart_col2 = st.columns(2)
    with f_chart_col1:
        raw_candidates = ['annual_income', 'spending', 'online_score']
        scaled_candidates = [f"{c}_scaled" for c in raw_candidates if f"{c}_scaled" in df_features.columns]
        if len(scaled_candidates) > 0:
            st.plotly_chart(plot_histogram_raw_vs_scaled(df_features, raw_candidates[0], scaled_candidates[0]), use_container_width=True)
        else:
            st.info("No scaled columns found.")
            
        eng_cols = [c for c, t in tags.items() if t == "🟩 Engineered"]
        if eng_cols:
            st.plotly_chart(plot_scatter_engineered_vs_target(df_features, eng_cols[0], active_target_col), use_container_width=True)

    with f_chart_col2:
        enc_cols = [c for c, t in tags.items() if t == "🟦 Encoded"]
        if enc_cols:
            st.plotly_chart(plot_strip_encoded_vs_target(df_features, enc_cols[0], active_target_col), use_container_width=True)
        if len(eng_cols) > 1:
            st.plotly_chart(plot_scatter_engineered_vs_target(df_features, eng_cols[1], active_target_col), use_container_width=True)

elif selected_page == "Baseline Model":
    st.markdown(f"""
    <div class="main-header">
        <h1>Baseline Model</h1>
        <p>Target: <b>{target_mode}</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state['df_features'] is None:
        st.warning("Please complete Feature Engineering first.")
        st.stop()
        
    df_features = st.session_state['df_features']
    X, y = prepare_data_for_modeling(df_features, active_target_col)
    
    with st.expander("Settings", expanded=True):
        m_c1, m_c2, m_c3 = st.columns(3)
        with m_c1:
            if is_classification_target:
                model_algo = st.selectbox("Algorithm", ["RandomForest", "LogisticRegression"], index=0)
            else:
                model_algo = st.selectbox("Algorithm", ["RandomForest", "LinearRegression"], index=0)
        with m_c2:
            test_split = st.slider("Test Size", min_value=0.15, max_value=0.35, value=0.20, step=0.05)
        with m_c3:
            rand_seed = st.number_input("Seed", value=42, step=1)
            btn_train_baseline = st.button("Train Model", use_container_width=True, type="primary")

    if btn_train_baseline or st.session_state['baseline_results'] is None:
        with st.spinner("Training baseline model..."):
            res = train_baseline_model(
                X, y,
                model_type=model_algo,
                is_classification=is_classification_target,
                test_size=test_split,
                random_state=rand_seed
            )
            st.session_state['baseline_results'] = res
            st.session_state['current_model'] = res['model']
            st.success(f"Trained {model_algo} in {res['fit_time']:.3f}s")

    base_res = st.session_state['baseline_results']

    m_cols = st.columns(len(base_res['metrics']))
    for idx, (m_name, m_val) in enumerate(base_res['metrics'].items()):
        with m_cols[idx]:
            st.metric(m_name, f"{m_val:.4f}")

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        if base_res['feature_importances'] is not None:
            st.plotly_chart(plot_feature_importances(base_res['feature_names'], base_res['feature_importances']), use_container_width=True)
        elif base_res['coefficients'] is not None:
            st.plotly_chart(plot_model_coefficients(base_res['feature_names'], base_res['coefficients']), use_container_width=True)
    with p_col2:
        st.plotly_chart(plot_y_true_vs_y_pred(base_res['y_test'], base_res['y_pred'], is_classification=is_classification_target), use_container_width=True)

elif selected_page == "Hyperparameter Tuning":
    st.markdown(f"""
    <div class="main-header">
        <h1>Hyperparameter Tuning</h1>
        <p>Target: <b>{target_mode}</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state['df_features'] is None:
        st.warning("Please complete Feature Engineering and Baseline Model first.")
        st.stop()
        
    df_features = st.session_state['df_features']
    X, y = prepare_data_for_modeling(df_features, active_target_col)
    
    with st.expander("Search Grid", expanded=True):
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            if is_classification_target:
                tune_algo = st.selectbox("Algorithm", ["RandomForest", "LogisticRegression"], index=0)
            else:
                tune_algo = st.selectbox("Algorithm", ["RandomForest", "LinearRegression"], index=0)
            cv_folds = st.slider("CV Folds", min_value=2, max_value=5, value=3)
        with t_col2:
            if "RandomForest" in tune_algo:
                n_est = st.multiselect("n_estimators", [50, 100, 150, 200], default=[50, 100, 150])
                m_depth = st.multiselect("max_depth", [4, 8, 12, None], default=[4, 8, None])
                p_grid = {'n_estimators': n_est, 'max_depth': m_depth}
            else:
                c_vals = st.multiselect("Regularization", [0.01, 0.1, 1.0, 10.0], default=[0.1, 1.0, 10.0])
                p_grid = {'C' if is_classification_target else 'alpha': c_vals}
                
            btn_tune = st.button("Run Grid Search", use_container_width=True, type="primary")

    if btn_tune:
        with st.spinner("Searching parameters..."):
            progress_bar = st.progress(0)
            for p in range(100):
                time.sleep(0.004)
                progress_bar.progress(p + 1)
            tuned = run_hyperparameter_tuning(
                X, y,
                model_type=tune_algo,
                is_classification=is_classification_target,
                param_grid=p_grid,
                cv=cv_folds
            )
            st.session_state['tuned_results'] = tuned
            st.success(f"Tuning finished in {tuned['total_fit_time']:.2f}s (Best CV: {tuned['best_cv_score']:.4f})")

    tuned_res = st.session_state['tuned_results']
    if tuned_res:
        st.markdown(f"""
        <div class="kpi-card" style="--kpi-accent: #10b981; margin: 1rem 0;">
            <div class="kpi-title">Best Parameters</div>
            <div class="kpi-value" style="font-size: 1.15rem;">{str(tuned_res['best_params'])}</div>
            <div class="kpi-subtext" style="color: #34d399;">
                Best CV Score: <b>{tuned_res['best_cv_score']:.4f}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Set as Active Model", use_container_width=True):
            st.session_state['current_model'] = tuned_res['best_model']
            st.success("Tuned model activated.")

        st.dataframe(tuned_res['results_df'], use_container_width=True)

        p_c1, p_c2 = st.columns(2)
        with p_c1:
            st.plotly_chart(plot_metric_across_params(tuned_res['results_df']), use_container_width=True)
        with p_c2:
            param_first = [c for c in tuned_res['results_df'].columns if c.startswith('param_')]
            if param_first:
                st.plotly_chart(plot_scatter_metric_vs_param(tuned_res['results_df'], param_first[0]), use_container_width=True)

elif selected_page == "Model Evaluation":
    st.markdown(f"""
    <div class="main-header">
        <h1>Model Evaluation</h1>
        <p>Target: <b>{target_mode}</b></p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state['baseline_results'] is None:
        st.warning("Please run Baseline Model first.")
        st.stop()
        
    base_res = st.session_state['baseline_results']
    tuned_res = st.session_state['tuned_results']
    
    st.markdown("### Comparison")
    cmp_col1, cmp_col2 = st.columns(2)
    with cmp_col1:
        st.markdown(f"#### Baseline ({base_res['model_name']})")
        for m_k, m_v in base_res['metrics'].items():
            st.metric(f"Baseline {m_k}", f"{m_v:.4f}")
            
    with cmp_col2:
        if tuned_res:
            st.markdown("#### Tuned Model")
            for m_k, m_v in tuned_res['metrics'].items():
                delta_val = m_v - base_res['metrics'].get(m_k, 0)
                st.metric(f"Tuned {m_k}", f"{m_v:.4f}", delta=f"{delta_val:+.4f}")
        else:
            st.info("Run Hyperparameter Tuning to view comparison.")

    if tuned_res:
        st.plotly_chart(plot_baseline_vs_tuned_bar(base_res['metrics'], tuned_res['metrics']), use_container_width=True)

    active_eval = tuned_res if tuned_res else base_res
    diag_data = compute_evaluation_diagnostics(
        active_eval['y_test'], active_eval['y_pred'], active_eval.get('y_prob'),
        is_classification=is_classification_target
    )

    if is_classification_target:
        tab_cm, tab_roc, tab_pr = st.tabs(["Confusion Matrix", "ROC Curve", "Precision-Recall"])
        with tab_cm:
            st.plotly_chart(plot_confusion_matrix(diag_data['confusion_matrix']), use_container_width=True)
        with tab_roc:
            st.plotly_chart(plot_roc_curve(diag_data['fpr'], diag_data['tpr'], diag_data['roc_auc']), use_container_width=True)
        with tab_pr:
            st.plotly_chart(plot_precision_recall_curve(diag_data['recall_curve'], diag_data['precision_curve'], diag_data['pr_auc']), use_container_width=True)
    else:
        tab_fit, tab_res_dist, tab_res_fit = st.tabs(["Predicted vs Actual", "Residuals", "Residuals vs Fitted"])
        with tab_fit:
            st.plotly_chart(plot_predicted_vs_actual_scatter(diag_data['actual'], diag_data['predicted']), use_container_width=True)
        with tab_res_dist:
            st.plotly_chart(plot_residuals_histogram(diag_data['residuals']), use_container_width=True)
        with tab_res_fit:
            st.plotly_chart(plot_residuals_vs_fitted(diag_data['predicted'], diag_data['residuals']), use_container_width=True)

elif selected_page == "Pipeline Runner":
    st.markdown("""
    <div class="main-header">
        <h1>Pipeline Runner</h1>
        <p>Run cleaning, encoding, features, training, and export artifacts.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("Pipeline Steps", expanded=True):
        p_c1, p_c2, p_c3, p_c4, p_c5 = st.columns(5)
        step_clean = p_c1.checkbox("1. Cleaning", value=True)
        step_encode = p_c2.checkbox("2. Encoding", value=True)
        step_features = p_c3.checkbox("3. Features", value=True)
        step_train = p_c4.checkbox("4. Train", value=True)
        step_eval = p_c5.checkbox("5. Evaluate", value=True)
        
        btn_run_full = st.button("Run Pipeline", use_container_width=True, type="primary")

    if btn_run_full:
        with st.status("Running pipeline...", expanded=True) as status_box:
            if step_clean:
                st.write("Cleaning data...")
                df_c, clean_sum = clean_data(df_filtered)
                st.session_state['df_clean'] = df_c
                st.session_state['cleaning_summary'] = clean_sum
            else:
                df_c = st.session_state['df_clean'] if st.session_state['df_clean'] is not None else df_filtered

            if step_encode or step_features:
                st.write("Transforming features...")
                df_f, f_tags, _, _ = engineer_features(df_c)
                st.session_state['df_features'] = df_f
                st.session_state['feature_tags'] = f_tags
            else:
                df_f = st.session_state['df_features']

            if step_train:
                st.write("Fitting model...")
                X_pipe, y_pipe = prepare_data_for_modeling(df_f, active_target_col)
                pipe_model_res = train_baseline_model(
                    X_pipe, y_pipe,
                    model_type='RandomForest',
                    is_classification=is_classification_target
                )
                st.session_state['baseline_results'] = pipe_model_res
                st.session_state['current_model'] = pipe_model_res['model']
            else:
                pipe_model_res = st.session_state['baseline_results']

            if step_eval and pipe_model_res:
                st.write("Evaluating...")
                score = list(pipe_model_res['metrics'].values())[0]
                st.session_state['pipeline_run_history'].append({
                    'Run': f"Run {len(st.session_state['pipeline_run_history'])+1}",
                    'Score': round(score, 4),
                    'Metric': list(pipe_model_res['metrics'].keys())[0]
                })

            status_box.update(label="Pipeline finished", state="complete", expanded=False)

    st.markdown("### Execution Status")
    stat_c1, stat_c2, stat_c3, stat_c4, stat_c5 = st.columns(5)
    
    def render_step_pill(title, done):
        cls = "status-pill-success" if done else "status-pill-pending"
        icon = "✓" if done else "•"
        return f'<div class="{cls}"><span>{icon}</span> <span>{title}</span></div>'
        
    stat_c1.markdown(render_step_pill("1. Cleaning", st.session_state['df_clean'] is not None), unsafe_allow_html=True)
    stat_c2.markdown(render_step_pill("2. Encoding", bool(st.session_state['feature_tags'])), unsafe_allow_html=True)
    stat_c3.markdown(render_step_pill("3. Features", st.session_state['df_features'] is not None), unsafe_allow_html=True)
    stat_c4.markdown(render_step_pill("4. Model Fit", st.session_state['current_model'] is not None), unsafe_allow_html=True)
    stat_c5.markdown(render_step_pill("5. Evaluated", len(st.session_state['pipeline_run_history']) > 0), unsafe_allow_html=True)

    if st.session_state['baseline_results'] is not None and st.session_state['current_model'] is not None:
        m_info = st.session_state['baseline_results']
        st.markdown("### Summary")
        sc1, sc2, sc3 = st.columns(3)
        primary_metric_name = list(m_info['metrics'].keys())[0]
        primary_metric_val = list(m_info['metrics'].values())[0]
        sc1.metric(f"Final {primary_metric_name}", f"{primary_metric_val:.4f}")
        sc2.metric("Fit Time", f"{m_info['fit_time']:.3f}s")
        sc3.metric("Features", f"{len(m_info['feature_names'])}")

        st.markdown("### Export Artifacts")
        config_dict = {
            'target_column': active_target_col,
            'is_classification': is_classification_target,
            'model_type': m_info['model_name'],
            'metrics': m_info['metrics'],
            'feature_names': m_info['feature_names'],
            'dataset_source': st.session_state['data_source_name']
        }
        joblib_bytes, config_json = export_pipeline_bundle(
            st.session_state['current_model'],
            m_info['feature_names'],
            m_info['metrics'],
            config_dict
        )

        d_col1, d_col2, d_col3 = st.columns(3)
        with d_col1:
            st.download_button(
                label="Download Model (.joblib)",
                data=joblib_bytes,
                file_name="budget_pipeline.joblib",
                mime="application/octet-stream"
            )
        with d_col2:
            st.download_button(
                label="Download Config (.json)",
                data=config_json.encode('utf-8'),
                file_name="pipeline_config.json",
                mime="application/json"
            )
        with d_col3:
            if st.session_state['df_clean'] is not None:
                st.download_button(
                    label="Download Cleaned CSV",
                    data=st.session_state['df_clean'].to_csv(index=False).encode('utf-8'),
                    file_name="cleaned_financial_data.csv",
                    mime="text/csv"
                )

        pl_c1, pl_c2 = st.columns(2)
        with pl_c1:
            if m_info['feature_importances'] is not None:
                st.plotly_chart(plot_feature_importances(m_info['feature_names'], m_info['feature_importances']), use_container_width=True)
        with pl_c2:
            st.plotly_chart(plot_segments_treemap(df_filtered), use_container_width=True)

st.markdown("""
<div class="app-footer">
    <b>Budget Deviation Predictor</b> • Departmental Financial Variance & Compliance Analytics
</div>
""", unsafe_allow_html=True)
