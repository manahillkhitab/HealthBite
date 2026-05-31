"""
HealthBite - Notebook 2: Data Cleaning
Run: python notebooks/02_clean_data.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('static/images', exist_ok=True)
os.makedirs('data/cleaned', exist_ok=True)

# ─────────────────────────────────────────────
# Cell 1: Load Raw Data
# ─────────────────────────────────────────────
df = pd.read_csv('data/raw/USDA.csv')
print(f"Original dataset: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ─────────────────────────────────────────────
# Cell 2: Map Columns
# USDA.csv columns: ID, Description, Calories, Protein,
#   TotalFat, Carbohydrate, Sodium, SaturatedFat,
#   Cholesterol, Sugar, Calcium, Iron, Potassium,
#   VitaminC, VitaminE, VitaminD
# ─────────────────────────────────────────────
column_mapping = {
    'Description'  : 'name',
    'Calories'     : 'calories',
    'Protein'      : 'protein',
    'Carbohydrate' : 'carbs',
    'TotalFat'     : 'fat',
    'Sugar'        : 'sugar',
    'Sodium'       : 'sodium',
    'SaturatedFat' : 'saturated_fat',
}

# Verify columns exist
missing = [c for c in column_mapping if c not in df.columns]
if missing:
    print(f"[WARNING] Columns not found: {missing}")
    print(f"Available columns: {df.columns.tolist()}")
else:
    print("[OK] All expected columns found!")

df_clean = df[list(column_mapping.keys())].copy()
df_clean.columns = list(column_mapping.values())

# Add fiber column (set to 0 if missing; USDA.csv doesn't have fiber)
df_clean['fiber'] = 0.0

print(f"\nSelected columns: {df_clean.columns.tolist()}")
print(f"\nFirst 3 rows:\n{df_clean.head(3)}")

# ─────────────────────────────────────────────
# Cell 3: Clean Data
# ─────────────────────────────────────────────
print(f"\nBefore cleaning: {df_clean.shape}")

# Convert numerics (handle any stray strings)
numeric_cols = ['calories', 'protein', 'carbs', 'fat', 'sugar', 'sodium',
                'saturated_fat', 'fiber']
for col in numeric_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Drop rows with any NaN in key columns
df_clean = df_clean.dropna(subset=['calories', 'protein', 'carbs', 'fat',
                                    'sugar', 'sodium'])
print(f"After dropna       : {df_clean.shape}")

# Remove invalid entries
df_clean = df_clean[df_clean['calories'] > 0]
print(f"After calories > 0 : {df_clean.shape}")

# Clip extreme outliers (99th percentile cap)
for col in numeric_cols:
    cap = df_clean[col].quantile(0.99)
    df_clean[col] = df_clean[col].clip(upper=cap)

# Remove duplicate food names
df_clean = df_clean.drop_duplicates(subset=['name'])
print(f"After dedup        : {df_clean.shape}")

# Reset index
df_clean = df_clean.reset_index(drop=True)

# ─────────────────────────────────────────────
# Cell 4: Statistics
# ─────────────────────────────────────────────
print("\n" + "=" * 50)
print("NUTRITIONAL STATISTICS")
print("=" * 50)
print(df_clean[numeric_cols].describe().round(2))

# ─────────────────────────────────────────────
# Cell 5: Distribution Plots
# ─────────────────────────────────────────────
plot_cols = ['calories', 'protein', 'carbs', 'fat', 'sugar', 'sodium',
             'saturated_fat']
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.suptitle('HealthBite - Nutritional Distributions', fontsize=16, fontweight='bold')

for idx, col in enumerate(plot_cols):
    ax = axes[idx // 4, idx % 4]
    df_clean[col].hist(bins=50, ax=ax, color='#4CAF50', edgecolor='white', alpha=0.85)
    ax.set_title(col.replace('_', ' ').title(), fontsize=11)
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Hide the last empty subplot
axes[1, 3].set_visible(False)
plt.tight_layout()
plt.savefig('static/images/distributions.png', dpi=100, bbox_inches='tight')
plt.close()
print("[OK] Saved: static/images/distributions.png")

# ─────────────────────────────────────────────
# Cell 6: Correlation Heatmap
# ─────────────────────────────────────────────
plt.figure(figsize=(10, 8))
corr = df_clean[plot_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, cmap='RdYlGn', center=0,
            square=True, linewidths=0.5, fmt='.2f',
            mask=mask, vmin=-1, vmax=1)
plt.title('HealthBite - Nutrient Correlations', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('static/images/correlation.png', dpi=100, bbox_inches='tight')
plt.close()
print("[OK] Saved: static/images/correlation.png")

# ─────────────────────────────────────────────
# Cell 7: Save Cleaned Data
# ─────────────────────────────────────────────
df_clean.to_csv('data/cleaned/foods_clean.csv', index=False)
print(f"\n[OK] Saved: data/cleaned/foods_clean.csv")
print(f"   Total foods  : {len(df_clean)}")
print(f"\nFirst 10 foods:")
print(df_clean[['name', 'calories', 'protein', 'sugar', 'sodium']].head(10).to_string())
