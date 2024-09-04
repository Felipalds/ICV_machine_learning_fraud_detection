

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

TOTAL_POINTS_OVERSSAMPLING = 19509
# TOTAL_POINTS_NEW_DATA = 5803
TOTAL_POINTS = TOTAL_POINTS_OVERSSAMPLING
# data = pd.read_csv("./data/new_data.csv")

data = pd.read_csv("./data/duplicated.csv")

X = data.drop('Class', axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Data dropped")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

best_mlp_acc = -1
best_i_mlp = 0
best_j_mlp = 0
best_k_mlp = 0
best_l_mlp = 0
best_y_pred = 0
# arvore de decisao e nayve bayes

for i in (5,6,10,12)
    for j in ('constant','invscaling', 'adaptive'):
        for k in (50,100,150,300,500,1000):
            for l in ('identity', 'logistic', 'tanh', 'relu'):
                MLP = MLPClassifier(hidden_layer_sizes=(i,i,1), learning_rate=j, max_iter=k, activation=l )
                MLP.fit(X_train_scaled,y_train)

                y_pred_mlp = MLP.predict(X_test_scaled)
                accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

                if (accuracy_mlp > best_mlp_acc):
                    best_mlp_acc = accuracy_mlp
                    best_i_mlp = i
                    best_j_mlp = j
                    best_k_mlp = k
                    best_l_mlp = l
                    best_y_pred = y_pred_mlp

print("Acc do  MLP:",best_mlp_acc)
print("C =", best_i_mlp)
print("Kernel =", best_j_mlp)
print("Max iter =", best_k_mlp)
print("Activation =", best_l_mlp)
cm_mlp = confusion_matrix(y_test, best_y_pred)
TP_MLP = cm_mlp[1][1]
FN_MLP = cm_mlp[1][0]
FP_MLP = cm_mlp[0][1]
TN_MLP = cm_mlp[0][0]
mlp_points = TP_MLP * 30 + FN_MLP * -50 + TN_MLP * 1 + FP_MLP * -5
print(f"MLP points: {mlp_points}")
mlp_percentage = mlp_points / TOTAL_POINTS
print("Acuracia ponderada MLP:", mlp_percentage)
mlp_recall = TP_MLP / (FN_MLP + TP_MLP)
print(f"MLP RECALL: ", mlp_recall)
