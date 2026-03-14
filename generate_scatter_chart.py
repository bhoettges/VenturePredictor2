"""
Reproduces the Actual vs. Predicted scatter plot (2x2 grid) from the saved model.
Re-runs the data pipeline with the same random_state to get the identical test split.
"""
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from financial_forecasting_model import load_and_clean_data, engineer_features, create_multistep_targets

plt.style.use('seaborn-v0_8-whitegrid')

# --- Load saved model and metadata ---
with open('lightgbm_financial_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model_pipeline = model_data['model_pipeline']
feature_cols = model_data['feature_cols']
target_cols = model_data['target_cols']
clip_bounds = model_data['clip_bounds']
train_medians_dict = model_data['train_medians']
results = model_data['performance_results']
TARGET_CAP = model_data.get('target_cap', 500)

# --- Rebuild the same test set ---
df_clean = load_and_clean_data("202402_Copy_Fixed.csv")
df_featured = engineer_features(df_clean)
df_model_ready = create_multistep_targets(df_featured, target_col='ARR YoY Growth (in %)', horizon=4)

df_model_ready.dropna(subset=target_cols, inplace=True)
for col in target_cols:
    df_model_ready[col] = df_model_ready[col].clip(-TARGET_CAP, TARGET_CAP)

non_feature_cols = ['Financial Quarter', 'id_company', 'time_idx', 'Year', 'Quarter Num',
                    'ARR YoY Growth (in %)', 'id'] + target_cols
orig_feature_cols = [col for col in df_model_ready.columns if col not in non_feature_cols]

clean_map = {c: re.sub(r'[^a-zA-Z0-9_]', '_', c) for c in orig_feature_cols}
df_model_ready.rename(columns=clean_map, inplace=True)

# Same split as training (random_state=42)
company_ids = df_model_ready['id_company'].unique()
train_cids, test_cids = train_test_split(company_ids, test_size=0.2, random_state=42)
test_indices = df_model_ready[df_model_ready['id_company'].isin(test_cids)].index

X_test = df_model_ready.loc[test_indices, feature_cols].copy()
y_test = df_model_ready.loc[test_indices, target_cols]

for col in feature_cols:
    lo, hi = clip_bounds[col]
    X_test[col] = X_test[col].clip(lo, hi)

train_medians = pd.Series(train_medians_dict)
X_test = X_test.fillna(train_medians)

# --- Predict ---
y_pred = model_pipeline.predict(X_test)
y_pred_df = pd.DataFrame(y_pred, columns=target_cols, index=y_test.index)

# --- Generate Actual vs Predicted scatter plots ---
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Performance: Actual vs. Predicted Growth Rate', fontsize=18)

for i, ax in enumerate(axes.flatten()):
    col = target_cols[i]
    r2 = results[col]['R2']

    sns.regplot(x=y_test[col], y=y_pred_df[col], ax=ax,
                scatter_kws={'alpha': 0.4, 's': 25, 'edgecolor': 'w', 'linewidths': 0.5},
                line_kws={'color': '#E41A1C', 'lw': 2.5})

    lo = min(y_test[col].min(), y_pred_df[col].min())
    hi = max(y_test[col].max(), y_pred_df[col].max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=2, label='Perfect Prediction')

    ax.set_title(f'{col} Forecast (R² = {r2:.3f})', fontsize=12)
    ax.set_xlabel('Actual Value', fontsize=10)
    ax.set_ylabel('Predicted Value', fontsize=10)
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('chart_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved chart_actual_vs_predicted.png")
