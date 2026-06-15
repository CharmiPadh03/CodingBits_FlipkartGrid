import pandas as pd
import numpy as np
import os

df = pd.read_csv('train.csv')

print("--- Data Info ---")
print(df.info())

print("\n--- Missing Values ---")
print(df.isnull().sum())

print("\n--- Numerical Description ---")
print(df.describe())

print("\n--- Categorical Unique Values ---")
for col in df.select_dtypes(include=['object']).columns:
    print(f"\n{col} unique values: {df[col].nunique()}")
    print(df[col].value_counts().head(10))

print("\n--- Demand Distribution (Head) ---")
print(df['demand'].describe())

