"""
HealthBite - Notebook 1: Data Exploration
Run: python notebooks/01_explore_data.py
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ─────────────────────────────────────────────
# Cell 1: Load Data
# ─────────────────────────────────────────────
pd.set_option('display.max_columns', None)

df = pd.read_csv('data/raw/USDA.csv')

print("=" * 50)
print("HEALTHBITE - DATA EXPLORATION")
print("=" * 50)
print(f"\nDataset shape : {df.shape}")
print(f"Total rows    : {len(df)}")
print(f"Total columns : {len(df.columns)}")

# ─────────────────────────────────────────────
# Cell 2: Column Names
# ─────────────────────────────────────────────
print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

# ─────────────────────────────────────────────
# Cell 3: Data Types & Missing Values
# ─────────────────────────────────────────────
print("\nData Types:")
print(df.dtypes)

print("\nMissing Values per Column:")
print(df.isnull().sum())

# ─────────────────────────────────────────────
# Cell 4: Sample Row
# ─────────────────────────────────────────────
print("\nSample nutrition values (row 0):")
print(df.iloc[0])

# ─────────────────────────────────────────────
# Cell 5: Basic Stats
# ─────────────────────────────────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print("\nNumerical columns summary:")
print(df[numeric_cols].describe())

print("\n[OK] Exploration complete!")
