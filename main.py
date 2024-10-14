import pandas as pd
csv_file = 'data/House Price Prediction Dataset.csv'
df = pd.read_csv(csv_file)
df.info()
df.head()
