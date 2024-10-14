import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import PolynomialFeatures  # Check this part
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

csv_file = 'data/House Price Prediction Dataset.csv'
df = pd.read_csv(csv_file)

# Check for missing values
missing_data = df.isnull().sum()
print(missing_data[missing_data > 0])

# Drop rows with missing values
df_cleaned = df.dropna()

# Define categorical variables
categorical_cols = ['Location', 'Condition', 'Garage']

# Apply One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Identify numerical columns
numerical_cols = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt', 'Price']

# Apply standardization
scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])

# Check the results
print(df_encoded.head())

# Select numerical columns
numerical_columns = df.select_dtypes(include=[np.number]).columns

# Descriptive statistics (Mean, Median, Mode, Variance, Standard Deviation)
for col in numerical_columns:
    print(f"Descriptive Statistics for '{col}' Column:")
    print(f"Mean: {df[col].mean()}")
    print(f"Median: {df[col].median()}")
    print(f"Mode: {df[col].mode()[0]}")
    print(f"Variance: {df[col].var()}")
    print(f"Standard Deviation: {df[col].std()}\n")

# 1. Z-Test: For the 'Area' column
# Compare the sample mean with the population mean
z_statistic, p_value = stats.ttest_1samp(df['Area'], 2800)
print(f"Z-Test - Statistic: {z_statistic}, P-value: {p_value}")

# 2. Independent T-Test: Compare 'Area' and 'Bedrooms' columns
t_statistic, p_value = stats.ttest_ind(df['Area'], df['Bedrooms'])
print(f"Independent T-Test - Statistic: {t_statistic}, P-value: {p_value}")

# 3. ANOVA (F-Test): Compare 'Area', 'Bedrooms', and 'Bathrooms' columns
f_statistic, p_value = stats.f_oneway(df['Area'], df['Bedrooms'], df['Bathrooms'])
print(f"ANOVA (F-Test) - Statistic: {f_statistic}, P-value: {p_value}")
