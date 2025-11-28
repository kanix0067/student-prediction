import pandas as pd
import numpy as np

df = pd.read_csv("C:\\Users\\KALAIYARASAN\\Desktop\\student\\data\\StudentsPerformance.csv")

print("Dataset Loaded Successfully!\n")

print("First 5 rows of dataset:")
print(df.head())

print("\nDataset Info:")
print(df.info())


print("\nMissing Values:")
print(df.isnull().sum())

print("\nDataset Description:")
print(df.describe())

print("\nMissing Values before cleaning:")
print(df.isnull().sum())

numeric_cols = ['writing score', 'reading score', 'math score']
for col in numeric_cols:
    df[col].fillna(df[col].mean(), inplace=True)

if 'gender' in df.columns:
    df['gender'] = df['gender'].map({'male': 0, 'female': 1})

if 'lunch' in df.columns:
    df['lunch'] = df['lunch'].map({'standard': 1, 'free/reduced': 0})

if 'test preparation course' in df.columns:
    df['test preparation course'] = df['test preparation course'].map({'completed': 1, 'none': 0})

print("\nFirst 5 rows after cleaning:")
print(df.head())
print("\nData types after cleaning:")
print(df.dtypes)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
sns.histplot(df['math score'], bins=10, kde=True)
plt.title("Math Score Distribution")

plt.subplot(1, 3, 2)
sns.histplot(df['reading score'], bins=10, kde=True)
plt.title("Reading Score Distribution")

plt.subplot(1, 3, 3)
sns.histplot(df['writing score'], bins=10, kde=True)
plt.title("Writing Score Distribution")

plt.tight_layout()
plt.show()

numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(6, 5))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Between Numeric Columns")
plt.show()

plt.figure(figsize=(6, 5))
sns.scatterplot(x='reading score', y='writing score', hue='gender', data=df)
plt.title("Reading vs Writing Scores by Gender")
plt.show()

plt.figure(figsize=(6, 5))
sns.barplot(x='lunch', y='math score', data=df)
plt.title("Average Math Score by Lunch Type")
plt.show()
plt.show(block=False)
plt.pause(2)
plt.close()

