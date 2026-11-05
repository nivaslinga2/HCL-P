import numpy as np
import pandas as pd

def generate_budget_dataset(n_samples=1250, random_state=42):
    np.random.seed(random_state)
    
    departments = [
        'IT & Cloud', 'Engineering', 'Marketing & Growth', 'Sales Operations', 
        'Finance & Accounting', 'Human Resources', 'Legal & Compliance', 'R&D Innovation'
    ]
    
    project_categories = [
        'Cloud Infrastructure', 'Software Licenses', 'Consulting Services',
        'Hardware Upgrades', 'Marketing Campaigns', 'Staff Augmentation', 'Facility Operations'
    ]
    
    cities = [
        'New York', 'London', 'Singapore', 'San Francisco', 'Tokyo',
        'Bangalore', 'Berlin', 'Sydney', 'Toronto', 'Chicago'
    ]
    
    quarters = ['Q1', 'Q2', 'Q3', 'Q4']
    risk_ratings = ['Low', 'Medium', 'High', 'Critical']
    cost_tiers = ['Tier 1', 'Tier 2', 'Tier 3']
    genders = ['Female', 'Male', 'Non-Binary']
    
    dept_choices = np.random.choice(departments, size=n_samples, p=[0.18, 0.20, 0.16, 0.14, 0.10, 0.08, 0.06, 0.08])
    cat_choices = np.random.choice(project_categories, size=n_samples)
    city_choices = np.random.choice(cities, size=n_samples, p=[0.18, 0.16, 0.12, 0.14, 0.10, 0.10, 0.06, 0.05, 0.05, 0.04])
    quarter_choices = np.random.choice(quarters, size=n_samples)
    gender_choices = np.random.choice(genders, size=n_samples, p=[0.48, 0.47, 0.05])
    cost_tier_choices = np.random.choice(cost_tiers, size=n_samples, p=[0.50, 0.35, 0.15])
    
    age = np.random.normal(loc=41, scale=8.5, size=n_samples).round().astype(int)
    age = np.clip(age, 24, 65)
    
    headcount = np.random.randint(5, 110, size=n_samples)
    
    dept_budget_mult = {
        'IT & Cloud': 1.35, 'Engineering': 1.45, 'Marketing & Growth': 1.15,
        'Sales Operations': 1.10, 'Finance & Accounting': 0.90, 'Human Resources': 0.75,
        'Legal & Compliance': 0.85, 'R&D Innovation': 1.40
    }
    base_budget = np.random.gamma(shape=12, scale=7000, size=n_samples) + (headcount * 850)
    annual_income = np.array([base_budget[i] * dept_budget_mult[dept_choices[i]] for i in range(n_samples)]).round(2)
    
    online_score = np.random.normal(loc=68, scale=14, size=n_samples)
    online_score = np.clip(online_score, 15.0, 99.0).round(1)
    
    approval_days = np.random.exponential(scale=11, size=n_samples).round().astype(int) + 3
    approval_days = np.clip(approval_days, 2, 48)
    
    risk_probs = []
    for i in range(n_samples):
        score = online_score[i]
        days = approval_days[i]
        if score > 75 and days < 10:
            risk_probs.append('Low')
        elif score > 55 and days < 20:
            risk_probs.append('Medium')
        elif days > 25:
            risk_probs.append('Critical')
        else:
            risk_probs.append('High')
    risk_rating = np.array(risk_probs)
    
    risk_inflation = {'Low': 0.94, 'Medium': 1.01, 'High': 1.12, 'Critical': 1.25}
    quarter_factor = {'Q1': 0.96, 'Q2': 1.00, 'Q3': 1.03, 'Q4': 1.14}
    
    noise = np.random.normal(0, 0.07, size=n_samples)
    spending = []
    for i in range(n_samples):
        base_mult = risk_inflation[risk_rating[i]] * quarter_factor[quarter_choices[i]] * (1.0 - (online_score[i] - 50) * 0.002) + noise[i]
        actual = annual_income[i] * base_mult
        spending.append(round(actual, 2))
    spending = np.array(spending)
    
    monthly_spending = np.round(spending / 12, 2)
    budget_deviation = np.round(spending - annual_income, 2)
    loyal_customer = np.where(spending <= (annual_income * 1.02), 1, 0)
    
    project_ids = [f"PRJ-FIN-{1000 + i}" for i in range(n_samples)]
    
    df = pd.DataFrame({
        'project_id': project_ids,
        'department': dept_choices,
        'project_category': cat_choices,
        'city': city_choices,
        'fiscal_quarter': quarter_choices,
        'cost_center_tier': cost_tier_choices,
        'age': age,
        'gender': gender_choices,
        'headcount': headcount,
        'approval_days': approval_days.astype(float),
        'online_score': online_score,
        'risk_rating': risk_rating,
        'annual_income': annual_income,
        'spending': spending,
        'monthly_spending': monthly_spending,
        'budget_deviation': budget_deviation,
        'loyal_customer': loyal_customer
    })
    
    nan_mask_approval = np.random.rand(n_samples) < 0.035
    df.loc[nan_mask_approval, 'approval_days'] = np.nan
    
    nan_mask_score = np.random.rand(n_samples) < 0.028
    df.loc[nan_mask_score, 'online_score'] = np.nan
    
    nan_mask_risk = np.random.rand(n_samples) < 0.024
    df.loc[nan_mask_risk, 'risk_rating'] = np.nan
    
    nan_mask_spending = np.random.rand(n_samples) < 0.015
    df.loc[nan_mask_spending, 'spending'] = np.nan
    df.loc[nan_mask_spending, 'monthly_spending'] = np.nan
    
    outlier_idx = np.random.choice(n_samples, size=8, replace=False)
    for idx in outlier_idx[:4]:
        df.loc[idx, 'spending'] = df.loc[idx, 'spending'] * 2.8
        df.loc[idx, 'monthly_spending'] = round(df.loc[idx, 'spending'] / 12, 2)
        df.loc[idx, 'budget_deviation'] = df.loc[idx, 'spending'] - df.loc[idx, 'annual_income']
        df.loc[idx, 'loyal_customer'] = 0
        
    for idx in outlier_idx[4:]:
        df.loc[idx, 'annual_income'] = df.loc[idx, 'annual_income'] * 2.5
    
    dup_rows = df.iloc[np.random.choice(n_samples, size=6, replace=False)].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    return df

if __name__ == '__main__':
    df = generate_budget_dataset(n_samples=1200, random_state=101)
    output_path = 'data/departmental_budget_deviations.csv'
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} rows and saved to {output_path}")
