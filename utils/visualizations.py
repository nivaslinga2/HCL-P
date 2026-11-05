import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

COLOR_THEME = {
    'primary': '#3b82f6',
    'secondary': '#6366f1',
    'accent': '#8b5cf6',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'neutral': '#94a3b8',
    'bg': '#0f172a',
    'card_bg': '#1e293b'
}

PALETTE_DISCRETE = px.colors.qualitative.Safe
PALETTE_CONTINUOUS = 'Viridis'

def apply_chart_style(fig, title="", height=420):
    fig.update_layout(
        title={
            'text': f"<b>{title}</b>" if title else "",
            'y': 0.96,
            'x': 0.02,
            'xanchor': 'left',
            'yanchor': 'top',
            'font': {'size': 14, 'color': '#f8fafc', 'family': 'Inter, sans-serif'}
        },
        template='plotly_dark',
        paper_bgcolor='rgba(15, 23, 42, 0.65)',
        plot_bgcolor='rgba(30, 41, 59, 0.4)',
        height=height,
        margin=dict(l=40, r=30, t=55, b=45),
        font=dict(family='Inter, sans-serif', color='#cbd5e1', size=12),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=11),
            bgcolor='rgba(15, 23, 42, 0.4)'
        ),
        hoverlabel=dict(
            bgcolor='#1e293b',
            font_size=12,
            font_family='Inter, sans-serif'
        )
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.07)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.07)')
    return fig

def plot_histogram_age(df):
    col = 'age' if 'age' in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    fig = px.histogram(
        df, x=col, nbins=25, marginal='box',
        color_discrete_sequence=['#38bdf8']
    )
    fig.update_layout(bargap=0.1)
    return apply_chart_style(fig, "Age Distribution")

def plot_histogram_annual_income(df):
    col = 'annual_income' if 'annual_income' in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    fig = px.histogram(
        df, x=col, nbins=30, marginal='violin',
        color_discrete_sequence=['#818cf8']
    )
    fig.update_layout(bargap=0.1)
    return apply_chart_style(fig, "Budget Allocation ($)")

def plot_box_spending_by_gender(df):
    spend_col = 'spending' if 'spending' in df.columns else ('monthly_spending' if 'monthly_spending' in df.columns else df.select_dtypes(include=[np.number]).columns[0])
    gender_col = 'gender' if 'gender' in df.columns else (df.select_dtypes(include=['object']).columns[0] if len(df.select_dtypes(include=['object']).columns) > 0 else None)
    
    fig = px.box(
        df, x=gender_col, y=spend_col, color=gender_col,
        color_discrete_sequence=PALETTE_DISCRETE,
        points='outliers'
    )
    return apply_chart_style(fig, "Spending by Gender")

def plot_bar_city_counts(df, top_n=10):
    city_col = 'city' if 'city' in df.columns else ('department' if 'department' in df.columns else None)
    if city_col and city_col in df.columns:
        counts = df[city_col].value_counts().head(top_n).reset_index()
        counts.columns = [city_col, 'Count']
        fig = px.bar(
            counts, x=city_col, y='Count',
            color='Count',
            color_continuous_scale='Tealgrn'
        )
    else:
        fig = px.bar(title="City data unavailable")
    return apply_chart_style(fig, f"Top {top_n} Locations")

def plot_missing_percentage(missing_report_df):
    fig = px.bar(
        missing_report_df, x='Column', y='Missing %',
        color='Missing %',
        color_continuous_scale='Reds',
        text='Missing %'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    return apply_chart_style(fig, "Missing Values (%)")

def plot_box_per_column(df, num_cols):
    sub_df = df[num_cols].dropna()
    melted = sub_df.melt(var_name='Variable', value_name='Value')
    fig = px.box(
        melted, x='Variable', y='Value', color='Variable',
        color_discrete_sequence=PALETTE_DISCRETE,
        points='outliers'
    )
    return apply_chart_style(fig, "Outlier Spread (IQR)")

def plot_violin_distribution(df, selected_col):
    fig = px.violin(
        df, y=selected_col, box=True, points='all',
        color_discrete_sequence=['#a78bfa']
    )
    return apply_chart_style(fig, f"Density Distribution: {selected_col}")

def plot_correlation_heatmap(df):
    num_df = df.select_dtypes(include=[np.number]).dropna()
    corr = num_df.corr().round(2)
    fig = px.imshow(
        corr, text_auto=True, aspect='auto',
        color_continuous_scale='Viridis'
    )
    return apply_chart_style(fig, "Correlation Matrix", height=480)

def plot_scatter_income_vs_spending(df):
    inc_col = 'annual_income' if 'annual_income' in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    sp_col = 'spending' if 'spending' in df.columns else ('monthly_spending' if 'monthly_spending' in df.columns else inc_col)
    target = 'loyal_customer' if 'loyal_customer' in df.columns else None
    
    fig = px.scatter(
        df, x=inc_col, y=sp_col, color=target,
        color_discrete_sequence=['#ef4444', '#10b981'],
        hover_data=['department'] if 'department' in df.columns else None
    )
    min_v = min(df[inc_col].min(), df[sp_col].min())
    max_v = max(df[inc_col].max(), df[sp_col].max())
    fig.add_trace(go.Scatter(
        x=[min_v, max_v], y=[min_v, max_v],
        mode='lines', line=dict(color='rgba(255,255,255,0.4)', dash='dash'),
        name='1:1 Parity Line'
    ))
    return apply_chart_style(fig, "Budget vs. Actual Spending")

def plot_histogram_overlay_before_after(df_before, df_after, column):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df_before[column].dropna(),
        name='Before',
        opacity=0.6,
        marker_color='#ef4444'
    ))
    fig.add_trace(go.Histogram(
        x=df_after[column].dropna(),
        name='After',
        opacity=0.6,
        marker_color='#10b981'
    ))
    fig.update_layout(barmode='overlay')
    return apply_chart_style(fig, f"Before vs After Cleaning: {column}")

def plot_box_side_by_side(df_before, df_after, column):
    s_before = pd.DataFrame({'Value': df_before[column], 'Stage': 'Before'})
    s_after = pd.DataFrame({'Value': df_after[column], 'Stage': 'After'})
    combined = pd.concat([s_before, s_after], ignore_index=True)
    
    fig = px.box(
        combined, x='Stage', y='Value', color='Stage',
        color_discrete_map={'Before': '#f59e0b', 'After': '#3b82f6'},
        points='outliers'
    )
    return apply_chart_style(fig, f"Outlier Comparison: {column}")

def plot_bar_missing_reduction(missing_before, missing_after):
    data = pd.DataFrame({
        'Status': ['Before', 'After', 'Resolved'],
        'Count': [missing_before, missing_after, max(0, missing_before - missing_after)],
        'Color': ['#ef4444', '#f59e0b', '#10b981']
    })
    fig = px.bar(
        data, x='Status', y='Count', color='Status',
        color_discrete_map={'Before': '#ef4444', 'After': '#f59e0b', 'Resolved': '#10b981'},
        text='Count'
    )
    fig.update_traces(textposition='outside')
    return apply_chart_style(fig, "Missing Values Handled")

def plot_histogram_raw_vs_scaled(df, raw_col, scaled_col):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df[raw_col], name=f"Raw ({raw_col})",
        marker_color='#94a3b8', opacity=0.7, xaxis='x1', yaxis='y1'
    ))
    fig.add_trace(go.Histogram(
        x=df[scaled_col], name=f"Scaled ({scaled_col})",
        marker_color='#f59e0b', opacity=0.7, xaxis='x2', yaxis='y2'
    ))
    fig.update_layout(
        grid=dict(rows=1, columns=2, pattern='independent')
    )
    return apply_chart_style(fig, f"Raw vs Scaled: {raw_col}")

def plot_strip_encoded_vs_target(df, enc_col, target_col):
    fig = px.strip(
        df.head(400), x=enc_col, y=target_col, color=target_col,
        color_discrete_sequence=PALETTE_DISCRETE
    )
    return apply_chart_style(fig, f"Encoded Feature: {enc_col}")

def plot_scatter_engineered_vs_target(df, eng_col, target_col):
    fig = px.scatter(
        df, x=eng_col, y=target_col, color=target_col if df[target_col].nunique() <= 5 else eng_col,
        color_continuous_scale='Viridis',
        trendline='ols'
    )
    return apply_chart_style(fig, f"Engineered Feature: {eng_col}")

def plot_feature_importances(feature_names, importances, top_n=12):
    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    imp_df = imp_df.sort_values(by='Importance', ascending=True).tail(top_n)
    
    fig = px.bar(
        imp_df, x='Importance', y='Feature', orientation='h',
        color='Importance', color_continuous_scale='Tealgrn'
    )
    return apply_chart_style(fig, f"Top {top_n} Feature Importances")

def plot_model_coefficients(feature_names, coefs, top_n=12):
    c_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
    c_df['AbsCoef'] = c_df['Coefficient'].abs()
    c_df = c_df.sort_values(by='AbsCoef', ascending=True).tail(top_n)
    
    fig = px.bar(
        c_df, x='Coefficient', y='Feature', orientation='h',
        color='Coefficient', color_continuous_scale='RdBu_r'
    )
    return apply_chart_style(fig, "Feature Coefficients")

def plot_y_true_vs_y_pred(y_true, y_pred, is_classification=False):
    comp_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    })
    
    if is_classification:
        counts = comp_df.groupby(['Actual', 'Predicted']).size().reset_index(name='Count')
        fig = px.bar(
            counts, x='Actual', y='Count', color='Predicted',
            barmode='group', color_discrete_sequence=['#ef4444', '#10b981']
        )
    else:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=comp_df['Actual'], name='Actual', opacity=0.6, marker_color='#3b82f6'))
        fig.add_trace(go.Histogram(x=comp_df['Predicted'], name='Predicted', opacity=0.6, marker_color='#f59e0b'))
        fig.update_layout(barmode='overlay')
        
    return apply_chart_style(fig, "Actual vs. Predicted")

def plot_metric_across_params(results_df, score_col='mean_test_score'):
    res = results_df.copy().reset_index()
    res['Run'] = [f"Run {i+1}" for i in range(len(res))]
    fig = px.bar(
        res, x='Run', y=score_col, color=score_col,
        color_continuous_scale='Viridis'
    )
    return apply_chart_style(fig, "Validation Score across Search Runs")

def plot_scatter_metric_vs_param(results_df, param_col, score_col='mean_test_score'):
    fig = px.scatter(
        results_df, x=param_col, y=score_col, size=score_col,
        color=score_col, color_continuous_scale='Tealrose'
    )
    return apply_chart_style(fig, f"Tuning: {param_col}")

def plot_parallel_coordinates(results_df, dimensions, score_col='mean_test_score'):
    fig = px.parallel_coordinates(
        results_df, dimensions=dimensions + [score_col],
        color=score_col, color_continuous_scale='Viridis'
    )
    return apply_chart_style(fig, "Parameter Parallel Coordinates")

def plot_confusion_matrix(cm, labels=['Overrun (0)', 'Compliant (1)']):
    fig = px.imshow(
        cm, x=labels, y=labels, text_auto=True,
        color_continuous_scale='Blues'
    )
    fig.update_xaxes(title="Predicted")
    fig.update_yaxes(title="Actual")
    return apply_chart_style(fig, "Confusion Matrix")

def plot_roc_curve(fpr, tpr, roc_auc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines',
        name=f"ROC (AUC = {roc_auc:.3f})",
        line=dict(color='#3b82f6', width=3)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        name='Random (0.500)',
        line=dict(color='rgba(255,255,255,0.4)', dash='dash')
    ))
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate")
    return apply_chart_style(fig, f"ROC Curve (AUC: {roc_auc:.3f})")

def plot_precision_recall_curve(recall, precision, pr_auc):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recall, y=precision, mode='lines',
        name=f"PR (AUC = {pr_auc:.3f})",
        line=dict(color='#10b981', width=3)
    ))
    fig.update_xaxes(title="Recall")
    fig.update_yaxes(title="Precision")
    return apply_chart_style(fig, f"Precision-Recall Curve (AUC: {pr_auc:.3f})")

def plot_predicted_vs_actual_scatter(y_actual, y_predicted):
    df_eval = pd.DataFrame({'Actual': y_actual, 'Predicted': y_predicted})
    fig = px.scatter(
        df_eval, x='Actual', y='Predicted',
        opacity=0.65, color_discrete_sequence=['#38bdf8']
    )
    min_v = min(df_eval['Actual'].min(), df_eval['Predicted'].min())
    max_v = max(df_eval['Actual'].max(), df_eval['Predicted'].max())
    fig.add_trace(go.Scatter(
        x=[min_v, max_v], y=[min_v, max_v],
        mode='lines', line=dict(color='#ef4444', dash='dash', width=2),
        name='1:1 Line'
    ))
    return apply_chart_style(fig, "Predicted vs Actual")

def plot_residuals_histogram(residuals):
    fig = px.histogram(
        x=residuals, nbins=30, marginal='rug',
        color_discrete_sequence=['#a855f7']
    )
    fig.update_layout(bargap=0.1)
    return apply_chart_style(fig, "Residuals Distribution")

def plot_residuals_vs_fitted(y_pred, residuals):
    df_res = pd.DataFrame({'Fitted': y_pred, 'Residuals': residuals})
    fig = px.scatter(
        df_res, x='Fitted', y='Residuals',
        opacity=0.6, color_discrete_sequence=['#ec4899']
    )
    fig.add_hline(y=0, line_dash='dash', line_color='white')
    return apply_chart_style(fig, "Residuals vs Fitted")

def plot_baseline_vs_tuned_bar(metrics_baseline, metrics_tuned):
    keys = list(metrics_baseline.keys())
    records = []
    for k in keys:
        if k in metrics_tuned:
            records.append({'Metric': k, 'Score': metrics_baseline[k], 'Model': 'Baseline'})
            records.append({'Metric': k, 'Score': metrics_tuned[k], 'Model': 'Tuned'})
            
    comp_df = pd.DataFrame(records)
    fig = px.bar(
        comp_df, x='Metric', y='Score', color='Model', barmode='group',
        color_discrete_map={'Baseline': '#64748b', 'Tuned': '#10b981'},
        text='Score'
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    return apply_chart_style(fig, "Baseline vs Tuned Comparison")

def plot_segments_treemap(df):
    req_cols = [c for c in ['department', 'project_category', 'fiscal_quarter'] if c in df.columns]
    val_col = 'spending' if 'spending' in df.columns else ('annual_income' if 'annual_income' in df.columns else None)
    
    if len(req_cols) >= 2 and val_col:
        fig = px.treemap(
            df, path=req_cols, values=val_col,
            color=val_col, color_continuous_scale='Viridis'
        )
    else:
        fig = px.bar(title="Insufficient columns for treemap")
    return apply_chart_style(fig, "Spending Treemap", height=460)

def plot_kpi_trends_over_runs(runs_history):
    if not runs_history:
        runs_history = [
            {'Run': 'Run 1', 'Score': 0.72, 'Metric': 'Score'},
            {'Run': 'Run 2', 'Score': 0.78, 'Metric': 'Score'},
            {'Run': 'Run 3', 'Score': 0.83, 'Metric': 'Score'},
            {'Run': 'Run 4', 'Score': 0.87, 'Metric': 'Score'}
        ]
    hist_df = pd.DataFrame(runs_history)
    fig = px.line(
        hist_df, x='Run', y='Score', markers=True,
        color_discrete_sequence=['#38bdf8']
    )
    return apply_chart_style(fig, "Score Progression across Runs")

def plot_churn_rate_by_city(df):
    city_col = 'city' if 'city' in df.columns else 'department'
    target_col = 'loyal_customer' if 'loyal_customer' in df.columns else None
    
    if city_col in df.columns and target_col in df.columns:
        agg = df.groupby(city_col)[target_col].agg(
            Overrun_Rate=lambda x: (1 - x.mean()) * 100,
            Projects='count'
        ).reset_index()
        fig = px.bar(
            agg.sort_values(by='Overrun_Rate', ascending=False),
            x=city_col, y='Overrun_Rate',
            color='Overrun_Rate', color_continuous_scale='Sunsetdark',
            text='Overrun_Rate'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    else:
        fig = px.bar(title="Data unavailable")
    return apply_chart_style(fig, "Overrun Rate (%) by City")

def plot_density_heatmap_score_vs_spending(df):
    x_col = 'online_score' if 'online_score' in df.columns else df.select_dtypes(include=[np.number]).columns[0]
    y_col = 'spending' if 'spending' in df.columns else ('monthly_spending' if 'monthly_spending' in df.columns else x_col)
    
    fig = px.density_heatmap(
        df, x=x_col, y=y_col,
        nbinsx=25, nbinsy=25,
        color_continuous_scale='Viridis'
    )
    return apply_chart_style(fig, "Procurement Score vs Spending")
