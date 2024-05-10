import pandas as pd

file_name = 'data/creditcard.csv'

data = pd.read_csv(file_name)

print(data.corr())

print(data.head())


