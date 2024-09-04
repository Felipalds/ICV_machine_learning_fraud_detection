from numpy.lib.function_base import average
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.space import Categorical
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# TOTAL_POINTS = 19509
data = pd.read_csv("./data/oversampled.csv")
result_array = []
# print(data)

print ("========== SEPARATING THE DATA ==========")
X = data.drop('Class', axis=1)
y = data['Class']

scaler = StandardScaler()
X = scaler.fit_transform(X)

x_train, x_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.5)

x_validation, x_test, y_validation, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5
)
print ("========== DATA SEPARATED AND SCALED ==========")


print("========== START KNN ==========")
knn_results = []
knn_param_space = {
    'n_neighbors' : (1, 10),
    'metric' : ['euclidean', 'manhattan', 'cosine', 'minkowski'],
}
knn = KNeighborsClassifier()
knn_bayes_search = BayesSearchCV(knn, knn_param_space, n_iter=50, cv=5, n_jobs=-1)
knn_bayes_search.fit(x_train, y_train)
knn_best_model = knn_bayes_search.best_estimator_

knn_opinion = knn_bayes_search.best_estimator_.predict(x_test)
knn_recall = recall_score(y_test, knn_opinion, average='macro')
knn_score = accuracy_score(y_test, knn_opinion)

print(f"Best KNN model: {knn_best_model} \n Best acc: {knn_score} \n Recall: {knn_recall}")

print("========== START SVM ==========")
svm_param_space = {
    'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
    'C' : (10, 30)
}
svm = SVC()
svm_bayes_search = BayesSearchCV(svm, svm_param_space, n_iter=50, cv=5, n_jobs=-1)
svm_bayes_search.fit(x_train, y_train)
svm_best_model = svm_bayes_search.best_estimator_

svm_opinion = svm_best_model.predict(x_test)
svm_recall = recall_score(y_test, svm_opinion, average='macro')
svm_score = accuracy_score(y_test, svm_opinion)
print(f"Best SVM model: {svm_best_model} \n Best acc: {svm_score} \n Recall: {svm_recall}")

print("========== START DECISION TREE ==========")
dt_param_space = {
    'criterion' : ['gini', 'entropy'],
    'max_depth' : (1, 10),
    'min_samples_split' : (2, 10),
    'min_samples_leaf' : (1, 10)
}
dt = DecisionTreeClassifier()
dt_bayes_search = BayesSearchCV(dt, dt_param_space, n_iter=50, cv=5, n_jobs=-1)
dt_bayes_search.fit(x_train, y_train)
dt_best_model = dt_bayes_search.best_estimator_

dt_opinion = dt_best_model.predict(x_test)
dt_recall = recall_score(y_test, dt_opinion, average='macro')
dt_score = accuracy_score(y_test, dt_opinion)
print(f"Best Decision Tree model: {dt_best_model} \n Best acc: {dt_score} \n Recall {dt_recall}")

print("========== START MLP ==========")
mlp_param_space = {
    # 'hidden_layer_sizes': Categorical([(5, 5, 1), (6, 6, 1), (10, 10, 1), (12, 12, 1)]),  # Different layer sizes
    'hidden_layer_sizes': Categorical([5, 6, 10, 12]),  # Different layer sizes
    'learning_rate': Categorical(['constant', 'invscaling', 'adaptive']),  # Learning rate schedules
    'max_iter': Categorical([50, 100, 150, 300, 500, 1000]),  # Iterations
    'activation': Categorical(['identity', 'logistic', 'tanh', 'relu'])  # Activation functions
}

mlp = MLPClassifier()
mlp_bayes_search = BayesSearchCV(mlp, mlp_param_space, n_iter=50, cv=5, n_jobs=-1)
mlp_bayes_search.fit(x_train, y_train)
mlp_best_model = mlp_bayes_search.best_estimator_

mlp_opinion = mlp_best_model.predict(x_test)
mlp_recall = recall_score(y_test, mlp_opinion, average='macro')
mlp_score = accuracy_score(y_test, mlp_opinion)
print(f"Best MLP model: {mlp_best_model} \n Acc: {mlp_score} \n Recall {mlp_recall}")

print("========== START NAIVE BAYES ==========")
nb = GaussianNB()
nb.fit(x_train, y_train)

nb_opinion = nb.predict(x_test)
nb_acc = accuracy_score(y_test, nb_opinion)
nb_recall = recall_score(y_test, nb_opinion, average='macro')

print(f"Naive Bayes model: {nb} \n Acc: {nb_acc} \n Recall {nb_recall}")



# print("Confusion Matrix")
# cm_knn = confusion_matrix(y_test, y_pred_knn)
# cm_svm = confusion_matrix(y_test, y_pred_svm)
# print("KNN\n", cm_knn)
# print("SVM\n", cm_svm)

# TP_KNN = cm_knn[1][1]
# FN_KNN = cm_knn[1][0]
# FP_KNN = cm_knn[0][1]
# TN_KNN = cm_knn[0][0]
# knn_points = TP_KNN * 30 + FN_KNN * -50 + TN_KNN * 1 + FP_KNN * -5
# print(f"KNN points: {knn_points}")
# knn_percentage = knn_points / TOTAL_POINTS
# print("Acuracia ponderada KNN:", knn_percentage)
# knn_recall = TP_KNN / (FN_KNN + TP_KNN)
# print(f"KNN RECALL: ", knn_recall)

# svm_points = TP_SVM * 30 + FN_SVM * -50 + TN_SVM * 1 + FP_SVM * -5
# print(f"SVM points: {svm_points}")
# svm_percentage = svm_points / TOTAL_POINTS
# print("Acuracia ponderada SVM: ", svm_percentage)
# svm_recall = TP_SVM / (FN_SVM + TP_SVM)
