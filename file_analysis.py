import pandas as pd


def file_analysis(data):
    print(data.info())
    print(data.head())
    frauds = len(data[data["Class"] == 1])
    print(frauds)
    print((frauds / len(data)) * 100)
    print(len(data))


d1 = pd.read_csv('data/creditcard.csv')

d1.drop('Time', axis=1, inplace=True)

d1.reset_index(drop=True, inplace=True)

file_analysis(d1)

print(d1.iterrows())
