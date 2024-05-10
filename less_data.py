import pandas as pd

data = pd.read_csv('./data/creditcard.csv')

data_frauds = data[data['Class'] == 1]
data_normals = data[data['Class'] == 0]
new_data = data.iloc[0:0].copy()


fraud_limit = 40
normal_limit = 27960

while fraud_limit > 0:
    example_row = data_frauds.sample().iloc[0]
    new_data.loc[len(new_data)] = example_row
    data_frauds = data_frauds.drop(example_row.name)
    fraud_limit -= 1
print("End to fraud limit ==============")

while normal_limit > 0:
    example_row = data_normals.sample().iloc[0]
    new_data.loc[len(new_data)] = example_row
    data_normals = data_normals.drop(example_row.name)
    normal_limit -= 1
print("End to normal limit ==============")

print(len(new_data))
new_data_frauds = len(new_data[new_data['Class'] == 1])
new_data_normals = len(new_data[new_data['Class'] == 0])
print(new_data_frauds / len(new_data))

new_data.to_csv("./data/new_data.csv")
