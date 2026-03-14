import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')

with open('lightgbm_financial_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

results = model_data['performance_results']
importances_df = model_data['feature_importance']

target_cols = list(results.keys())
mae_vals = [results[col]['MAE'] for col in target_cols]
r2_vals = [results[col]['R2'] for col in target_cols]

# Chart 1: MAE by Forecast Horizon
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(target_cols, mae_vals, color='#C0392B', width=0.6)
ax.set_xlabel('horizon', fontsize=12)
ax.set_ylabel('MAE', fontsize=12)
ax.set_title('Model Performance by Forecast Horizon (MAE)', fontsize=14)
ax.set_ylim(bottom=0)
plt.tight_layout()
plt.savefig('chart_mae_by_horizon.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved chart_mae_by_horizon.png")

# Chart 2: R² by Forecast Horizon
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(target_cols, r2_vals, color='#27AE60', width=0.6)
ax.set_xlabel('horizon', fontsize=12)
ax.set_ylabel('R2', fontsize=12)
ax.set_title('Model Performance by Forecast Horizon (R²)', fontsize=14)
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig('chart_r2_by_horizon.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved chart_r2_by_horizon.png")

# Chart 3: Top 25 Feature Importances
top25 = importances_df.head(25)
fig, ax = plt.subplots(figsize=(12, 10))
sns.barplot(x=top25['mean_importance'], y=top25.index, palette='viridis_r', ax=ax)
ax.set_title('Top 25 Most Predictive Features (Averaged Across All Forecast Horizons)', fontsize=16)
ax.set_xlabel('Mean Importance Score', fontsize=12)
ax.set_ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig('chart_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
print("Saved chart_feature_importance.png")

print("\nPerformance Summary:")
for col in target_cols:
    print(f"  {col}: MAE={results[col]['MAE']:.4f}, R²={results[col]['R2']:.4f}")
