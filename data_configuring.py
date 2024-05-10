import pandas as pd
import random


def random_change(value):
    return value * random.uniform(0.9, 1.1)


data = pd.read_csv('./data/new_data.csv')

# data_duplicates = data.copy()
data_oversampling = data.copy()
#
# total = len(data)
# frauds = len(data[data["Class"] == 1])
# c = 0
# while frauds / total < 0.05:
#     if data.iloc[c]['Class'] == 1:
#         new_row = data.iloc[c]
#         data_duplicates.loc[len(data_duplicates)] = new_row
#         frauds += 1
#         total += 1
#         print(frauds/total)
#     if c == len(data) - 1:
#         c = 0
#     c += 1
# print("End to data duplication ===========")

total = len(data)
frauds = len(data[data["Class"] == 1])
c = 0
print(frauds/total)
while frauds / total < 0.05:
    if data.iloc[c]['Class'] == 1:
        new_row = data.iloc[c].copy()
        for key in new_row.index:
            if key != 'Class':
                new_row[key] = random_change(new_row[key])

        data_oversampling.loc[len(data_oversampling)] = new_row
        frauds += 1
        total += 1
        print(frauds/total)
    if c == len(data) - 1:
        c = 0
    c += 1
print("End to data oversampling ===========")

data_oversampling.to_csv('./data/oversampled.csv')
# data_duplicates.to_csv('./data/duplicated.csv')
