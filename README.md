# Departmental Budget Deviation Predictor

An interactive analytics and machine learning application to analyze departmental spending, detect budget overruns, and forecast fiscal compliance.

---

## Overview

This project provides an end-to-end data science workflow to analyze budget allocations versus actual expenditures across departments:
- **Exploratory Data Analysis (EDA):** Summary stats, distributions, and demographic breakdowns.
- **Data Quality & Diagnostics:** Missing value audits and IQR/Z-score outlier detection.
- **Preprocessing:** Imputation, deduplication, and winsorization / outlier removal.
- **Feature Engineering:** Variance ratios, headcount spending, friction indices, cyclical quarters, one-hot/ordinal encoding, and scaling.
- **Model Training:** Baseline models for classification (budget discipline) or regression (expenditure prediction).
- **Hyperparameter Optimization:** Multi-fold cross-validation grid search.
- **Model Evaluation:** Confusion matrix, ROC-AUC, Precision-Recall curves, residuals, and side-by-side metric comparison.
- **Pipeline Runner & Serialization:** One-click pipeline execution with `.joblib` model bundle and `.json` config export.

---

## Prediction Targets

1. **Classification (`loyal_customer`):**
   - `1`: Compliant (actual spending within allocated budget + 2% margin).
   - `0`: Overrun (budget deviation exceeded).
2. **Regression (`monthly_spending`):**
   - Predicts monthly expenditure based on department size, budget, approval timelines, and operational metrics.

---

## Project Structure

```text
HCLTech-P/
├── app.py                     # Streamlit web application
├── verify_pipeline.py         # Pipeline test & verification script
├── requirements.txt           # Project dependencies
├── README.md                  # Project documentation
├── data/
│   ├── departmental_budget_deviations.csv  # Dataset
│   └── generate_dataset.py                # Synthetic data generator
├── styles/
│   └── theme.css              # Custom styling
└── utils/
    ├── data_manager.py        # Loading, cleaning, and feature engineering
    ├── ml_pipeline.py         # Model training, tuning, and evaluation
    └── visualizations.py      # Plotly interactive chart definitions
```

---

## Installation

1. Clone or download the repository:
   ```bash
   cd HCLTech-P
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Application

Start the Streamlit dashboard:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## Running Tests

Run the pipeline verification script:

```bash
python verify_pipeline.py
```

---

## Key Features

- **Global Filtering:** Filter dynamically by department, office city, fiscal quarter, and budget range.
- **Side-by-Side Model Benchmarking:** Compare baseline vs. hyperparameter-tuned model metrics directly.
- **Exportable Artifacts:** Download cleaned datasets (`.csv`), serialized models (`.joblib`), and configuration metadata (`.json`).

