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

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures  # Check this part
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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

# IQR (Interquartile Range) Method to Detect Outliers
Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_iqr = df[(df['Price'] < lower_bound) | (df['Price'] > upper_bound)]

# Detect outliers using Z-score method
z_scores = (df['Price'] - df['Price'].mean()) / df['Price'].std()
outliers_z = df[np.abs(z_scores) > 3]

# Print outliers
print("Outliers detected using IQR method:")
print(outliers_iqr)

print("\nOutliers detected using Z-score method:")
print(outliers_z)

# IQR Method for Outlier Detection in 'Area', 'Bedrooms', and 'Bathrooms' Columns
for col in ['Area', 'Bedrooms', 'Bathrooms']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_iqr = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    # Print outliers
    print(f"Outliers in {col} column detected using IQR method:")
    print(outliers_iqr)

# Detect outliers using Z-score Method in 'Area', 'Bedrooms', and 'Bathrooms' Columns
for col in ['Area', 'Bedrooms', 'Bathrooms']:
    z_scores = (df[col] - df[col].mean()) / df[col].std()
    outliers_z = df[np.abs(z_scores) > 3]
    
    # Print outliers
    print(f"Outliers in {col} column detected using Z-score method:")
    print(outliers_z)

# Select only numerical columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Correlation Matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_cols].corr()

# Visualize with Heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.show()

# Polynomial Features Creation
poly = PolynomialFeatures(degree=2, include_bias=False)
numerical_cols = ['Area', 'Bedrooms', 'Bathrooms']
poly_features = poly.fit_transform(df[numerical_cols])

# Add new features to the dataframe
df_poly = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(numerical_cols))
df = pd.concat([df, df_poly], axis=1)

print(df.head())

Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers_iqr = df[(df['Price'] < lower_bound) | (df['Price'] > upper_bound)]

# Z-skor yöntemi ile aykırı değerleri tespit etme
z_scores = (df['Price'] - df['Price'].mean()) / df['Price'].std()
outliers_z = df[np.abs(z_scores) > 3]

# Aykırı değerleri yazdırma
print("IQR Yöntemi ile Aykırı Değerler:")
print(outliers_iqr)

print("\nZ-Skor Yöntemi ile Aykırı Değerler:")
print(outliers_z)

# IQR (Interquartile Range) Yöntemi ile Aykırı Değerleri Tespit Etme

# 'Area', 'Bedrooms' ve 'Bathrooms' sütunları üzerinde işlem yapıyoruz
for col in ['Area', 'Bedrooms', 'Bathrooms']:
    
    # İlk çeyreklik değer (Q1): Verilerin %25'inin altında kalan değer
    Q1 = df[col].quantile(0.25)
    
    # Üçüncü çeyreklik değer (Q3): Verilerin %75'inin altında kalan değer
    Q3 = df[col].quantile(0.75)
    
    # IQR (Interquartile Range): Q3 - Q1 farkı, çeyreklikler arası yayılımı temsil eder
    IQR = Q3 - Q1

    # Alt sınır: Q1 - 1.5 * IQR. Bu sınırın altındaki değerler aykırı kabul edilir
    lower_bound = Q1 - 1.5 * IQR
    
    # Üst sınır: Q3 + 1.5 * IQR. Bu sınırın üstündeki değerler aykırı kabul edilir
    upper_bound = Q3 + 1.5 * IQR

    # Aykırı değerler: Belirlenen alt ve üst sınırların dışındaki değerler
    outliers_iqr = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    # Aykırı değerleri yazdır
    print(f"IQR Yöntemi ile {col} sütunundaki Aykırı Değerler:")
    print(outliers_iqr)

# Z-Skor Yöntemi ile Aykırı Değerleri Tespit Etme

# 'Area', 'Bedrooms' ve 'Bathrooms' sütunları üzerinde işlem yapıyoruz
for col in ['Area', 'Bedrooms', 'Bathrooms']:
    
    # Z-skoru hesaplama: Verinin ortalamadan kaç standart sapma uzaklıkta olduğunu gösterir
    z_scores = (df[col] - df[col].mean()) / df[col].std()
    
    # Z-skoru 3'ten büyük olan (veya -3'ten küçük olan) değerler aykırı olarak kabul edilir
    outliers_z = df[np.abs(z_scores) > 3]
    
    # Aykırı değerleri yazdır
    print(f"Z-Skor Yöntemi ile {col} sütunundaki Aykırı Değerler:")
    print(outliers_z)

# Sadece sayısal sütunları seçelim
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Korelasyon Matrisi
plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_cols].corr()

# Isı Haritası ile Görselleştirme
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.show()


# Polinom özellikler oluşturma
poly = PolynomialFeatures(degree=2, include_bias=False)
numerical_cols = ['Area', 'Bedrooms', 'Bathrooms']
poly_features = poly.fit_transform(df[numerical_cols])

# Yeni özellikleri dataframe'e ekleme
df_poly = pd.DataFrame(poly_features, columns=poly.get_feature_names_out(numerical_cols))
df = pd.concat([df, df_poly], axis=1)

print(df.head())

# Z Test - Is the average price 500,000?
# Null Hypothesis: Average 500,000
# Alternative Hypothesis: Average not 500,000

# Get the 'Price' column
house_prices = df['Price']  # Price data

# Calculate the sample mean and standard deviation
mean_price = np.mean(house_prices)
std_price = np.std(house_prices)

# Set the population mean to 500,000
pop_mean = 500000

# Calculate the z-score and p-value
z_score, p_value = stats.ttest_1samp(house_prices, pop_mean)

# Print the results
print(f"Z-Score: {z_score}, P-value: {p_value}")

# Interpretation of results:
# Evaluating the hypothesis based on the p-value:
# If the p-value is less than 0.05, the null hypothesis is rejected, and it is concluded that the average price is not 500,000.
# If the p-value is greater than 0.05, the null hypothesis is accepted, and it is assumed that the average price is 500,000.

if p_value < 0.05:
    print("Result is significant: The average price is different from 500,000.")
    # Comment:
    # Since the z-score is 6.10 and the p-value is 1.31e-09, we conclude that the average price is not 500,000.
    # Such a small p-value strongly indicates that the average price in the data is significantly different from 500,000.
else:
    print("Result is not significant: The average price is 500,000.")
    # Comment:
    # If the p-value had been greater than 0.05, the average price would have been accepted as 500,000.



# ANOVA Test - Let's compare prices in different locations.
# Null Hypothesis: The average prices of all locations are equal.
# Alternative Hypothesis: At least one location's average price is different from the others.

# Let's group the prices by different locations
# For example, we are comparing different cities in the 'Location' column
locations = df['Location'].unique()

# Separate prices by location and store them in a list
grouped_prices = [df[df['Location'] == loc]['Price'] for loc in locations]

# Perform ANOVA test
f_stat, p_value = stats.f_oneway(*grouped_prices)

# Print the results
print(f"F-Stat: {f_stat}, P-value: {p_value}")

# Interpretation of results:
# Evaluating the hypothesis based on the p-value:
# If the p-value is less than 0.05, the null hypothesis is rejected, and at least one location's average price is different from the others.
if p_value < 0.05:
    print("Result is significant: At least one location's average price is different from the others.")
    # Comment:
    # There is a direct relationship between the F-statistic and the p-value.
    # If the p-value is less than 0.05, it is concluded that at least one location's average price is different from the others.
else:
    print("Result is not significant: The average prices of all locations are equal.")
    # Comment:
    # If the p-value is greater than 0.05, we accept that the average prices of all locations are equal.
    

# ANOVA Testi - Farklı lokasyonlardaki fiyatları karşılaştıralım.
# Null Hipotez: Tüm lokasyonların ortalama fiyatları eşittir.
# Alternatif Hipotez: En az bir lokasyonun ortalama fiyatı diğerlerinden farklıdır.

# Farklı lokasyonlardaki fiyatları gruplara ayıralım
# Örnek olarak, 'Location' sütunundaki farklı şehirleri karşılaştırıyoruz
locations = df['Location'].unique()

# Fiyatları lokasyonlara göre ayırıp listede tutalım
grouped_prices = [df[df['Location'] == loc]['Price'] for loc in locations]

# ANOVA testi yapalım
f_stat, p_value = stats.f_oneway(*grouped_prices)

# Sonuçları yazdıralım
print(f"F-Stat: {f_stat}, P-değeri: {p_value}")

# Sonuç yorumlaması:
# P-değerine göre hipotezi değerlendiriyoruz:
# Eğer p-değeri 0.05'ten küçükse, null hipotez reddedilir ve en az bir lokasyonun ortalama fiyatı diğerlerinden farklıdır.
if p_value < 0.05:
    print("Sonuç anlamlı: En az bir lokasyonun ortalama fiyatı diğerlerinden farklıdır.")
    # Yorum:
    # F skoru ile p-değeri arasında doğrudan bir ilişki vardır.
    # Eğer p-değeri 0.05'ten küçükse, en az bir lokasyonun fiyat ortalamasının diğerlerinden farklı olduğu sonucuna varılır.
else:
    print("Sonuç anlamlı değil: Tüm lokasyonların ortalama fiyatları birbirine eşittir.")
    # Yorum:
    # Eğer p-değeri 0.05'ten büyükse, tüm lokasyonların ortalama fiyatlarının birbirine eşit olduğunu kabul ederiz.


# Pearson korelasyonu hesapla (Price ile Area arasında)
pearson_corr, pearson_p_value = stats.pearsonr(df['Price'], df['Area'])

# Sonuçları yazdır
print(f"Pearson Korelasyon Katsayısı: {pearson_corr}, P-değeri: {pearson_p_value}")

# Yorumlama
if pearson_p_value < 0.05:
    print("Sonuç anlamlı: Price ve Area arasında istatistiksel olarak anlamlı bir ilişki vardır.")
else:
    print("Sonuç anlamlı değil: Price ve Area arasında istatistiksel olarak anlamlı bir ilişki yoktur.")

# Spearman Korelasyonu (Price ve Area)
spearman_corr, spearman_p_value = stats.spearmanr(df['Price'], df['Area'])

# Sonuçları yazdır
print(f"Spearman Korelasyon Katsayısı: {spearman_corr}, P-değeri: {spearman_p_value}")

# Yorumlama
if spearman_p_value < 0.05:
    print("Sonuç anlamlı: Price ve Area arasında istatistiksel olarak anlamlı bir monotonik ilişki vardır.")
else:
    print("Sonuç anlamlı değil: Price ve Area arasında istatistiksel olarak anlamlı bir monotonik ilişki yoktur.")


# Kendall's Tau Korelasyonu (Price ve Area)
kendall_corr, kendall_p_value = stats.kendalltau(df['Price'], df['Area'])

# Sonuçları yazdır
print(f"Kendall's Tau Korelasyon Katsayısı: {kendall_corr}, P-değeri: {kendall_p_value}")

# Yorumlama
if kendall_p_value < 0.05:
    print("Sonuç anlamlı: Price ve Area arasında istatistiksel olarak anlamlı bir monotonik ilişki vardır.")
else:
    print("Sonuç anlamlı değil: Price ve Area arasında istatistiksel olarak anlamlı bir monotonik ilişki yoktur.")