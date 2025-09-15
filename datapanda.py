"""
Assignment Week 7: Data Analysis and Visualization with the Iris Dataset
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Task 1: Load and Explore the Dataset
try:
    # Load Iris dataset from sklearn
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    print("First five rows of the dataset:")
    print(df.head())
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Check data types and missing values
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

# Clean dataset (fill or drop missing values)
if df.isnull().values.any():
    df = df.fillna(df.mean(numeric_only=True))
    df = df.dropna()
    print("\nMissing values handled.")
else:
    print("\nNo missing values detected.")

# Task 2: Basic Data Analysis
print("\nBasic statistics:")
print(df.describe())

# Group by species and compute mean of numerical columns
grouped = df.groupby('species').mean(numeric_only=True)
print("\nMean values by species:")
print(grouped)

# Identify patterns or interesting findings
print("\nInteresting findings:")
for col in iris.feature_names:
    max_species = grouped[col].idxmax()
    print(f"Species with highest average {col}: {max_species}")

# Task 3: Data Visualization
sns.set(style="whitegrid")

# 1. Line chart: mean sepal length per sample index (not a time series, but for demonstration)
plt.figure(figsize=(8, 4))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
plt.title('Sepal Length Across Samples')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.tight_layout()
plt.show()

# 2. Bar chart: average petal length per species
plt.figure(figsize=(6, 4))
sns.barplot(x=grouped.index, y=grouped['petal length (cm)'], palette='viridis')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.tight_layout()
plt.show()

# 3. Histogram: distribution of sepal width
plt.figure(figsize=(6, 4))
sns.histplot(df['sepal width (cm)'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# 4. Scatter plot: sepal length vs petal length, colored by species
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='deep')
plt.title('Sepal Length vs Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.tight_layout()
plt.show()

print("\nAll tasks completed. Each plot is labeled and provides insights into the Iris dataset.")

# Run the script using the command: C:/Python312/python.exe datapanda.py
