from numpy.lib.function_base import average
import pandas as pd
from pandas import DataFrame
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
from skopt.space import Categorical, Integer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from mlp_wrapper import MLPWrapper

data_original = pd.read_csv("./data/new_data.csv")
data_duplicated = pd.read_csv("./data/oversampled.csv")
data_duplicated = pd.read_csv("./data/duplicated.csv")

datas = [data_original, data_duplicated, data_duplicated]
out = ["original.csv", "oversampled.csv", "duplicated.csv"]


for c in range(0, 3):

    data = datas[c]

    X = data.drop('Class', axis=1)
    y = data['Class']

    bankrupts = len([x for x in y if x == 1])
    non_bankrupts = len([x for x in y if x == 0])

    print(f"Bankrupts (1s): {bankrupts}")
    print(f"Non-bankrupts (0s): {non_bankrupts}")

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y)

    bankrupts = len([x for x in y_test if x == 1])
    non_bankrupts = len([x for x in y_test if x == 0])

    print(f"Bankrupts (1s): {bankrupts}")
    print(f"Non-bankrupts (0s): {non_bankrupts}")
    print(bankrupts, non_bankrupts)

    print(len(x_test))

    print ("========== DATA SEPARATED AND SCALED ==========")

    knn_accuracies = []
    knn_recalls = []
    knn_best_models = []
    knn_best_recall = 0
    knn_best_recall_model = None

    svm_accuracies = []
    svm_recalls = []
    svm_best_models = []
    svm_best_recall = 0
    svm_best_recall_model = None

    mlp_accuracies = []
    mlp_recalls = []
    mlp_best_models = []
    mlp_best_recall = 0
    mlp_best_recall_model = None

    nb_accuracies = []
    nb_recalls = []
    nb_best_models = []
    nb_best_recall = 0
    nb_best_recall_model = None

    dt_accuracies = []
    dt_recalls = []
    dt_best_models = []
    dt_best_recall = 0
    dt_best_recall_model = None


    for i in range(0, 10):
        print("knn")

        knn_param_space = {
            'n_neighbors' : (1, 30),
            'metric' : ['euclidean', 'manhattan', 'cosine', 'minkowski'],
            'weights' : ['uniform', 'distance']
        }
        knn = KNeighborsClassifier()
        knn_bayes_search = BayesSearchCV(knn, knn_param_space, n_iter=50, cv=5, n_jobs=5)
        knn_bayes_search.fit(x_train, y_train)
        knn_best_model = knn_bayes_search.best_estimator_

        knn_opinion = knn_bayes_search.best_estimator_.predict(x_test)
        knn_recall = recall_score(y_test, knn_opinion)

        if knn_recall > knn_best_recall:
            knn_best_recall = knn_recall
            knn_best_recall_model = knn_best_model

        knn_score = accuracy_score(y_test, knn_opinion)

        # print("Matrix", confusion_matrix(y_test, knn_opinion))
        # print(f"Best KNN model: {knn_best_model} \n Best acc: {knn_score} \n Recall: {knn_recall}")
        knn_recalls.append(knn_recall)
        knn_accuracies.append(knn_score)
        knn_best_models.append(knn_best_model)


        print("svm")
        svm_param_space = {
            'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
            'C' : (10, 30)
        }
        svm = SVC()
        svm_bayes_search = BayesSearchCV(svm, svm_param_space, n_iter=50, cv=5, n_jobs=5)
        svm_bayes_search.fit(x_train, y_train)
        svm_best_model = svm_bayes_search.best_estimator_

        svm_opinion = svm_best_model.predict(x_test)
        svm_recall = recall_score(y_test, svm_opinion)
        svm_score = accuracy_score(y_test, svm_opinion)

        if svm_recall > svm_best_recall:
            svm_best_recall = svm_recall
            svm_best_recall_model = svm_best_model

        svm_accuracies.append(svm_score)
        svm_recalls.append(svm_recall)
        svm_best_models.append(svm_best_model)

        print("dt")
        dt_param_space = {
            'criterion' : ['gini', 'entropy'],
            'max_depth' : (1, 10),
            'min_samples_split' : (2, 10),
            'min_samples_leaf' : (1, 10)
        }
        dt = DecisionTreeClassifier()
        dt_bayes_search = BayesSearchCV(dt, dt_param_space, n_iter=50, cv=5, n_jobs=5)
        dt_bayes_search.fit(x_train, y_train)
        dt_best_model = dt_bayes_search.best_estimator_

        dt_opinion = dt_best_model.predict(x_test)
        dt_recall = recall_score(y_test, dt_opinion)
        dt_score = accuracy_score(y_test, dt_opinion)

        if dt_recall > dt_best_recall:
            dt_best_recall = dt_recall
            dt_best_recall_model = dt_best_model

        dt_accuracies.append(dt_score)
        dt_recalls.append(dt_recall)
        dt_best_models.append(dt_best_model)

        print("mlp")
        mlp_param_space = {
            'layer1': Integer(10, 100),
            'layer2': Integer(10, 100),
            'layer3': Integer(10, 100),
            'learning_rate': Categorical(['constant', 'invscaling', 'adaptive']),  # Learning rate schedules
            'activation': Categorical(['identity', 'logistic', 'tanh', 'relu'])  # Activation functions
        }

        mlp = MLPWrapper()
        mlp_bayes_search = BayesSearchCV(mlp, mlp_param_space, n_iter=50, cv=5, n_jobs=5)
        mlp_bayes_search.fit(x_train, y_train)
        mlp_best_model = mlp_bayes_search.best_estimator_

        mlp_opinion = mlp_best_model.predict(x_test)
        mlp_recall = recall_score(y_test, mlp_opinion)
        mlp_score = accuracy_score(y_test, mlp_opinion)

        if mlp_recall > mlp_best_recall:
            mlp_best_recall = mlp_recall
            mlp_best_recall_model = mlp_best_model

        mlp_accuracies.append(mlp_score)
        mlp_recalls.append(mlp_recall)
        mlp_best_models.append(mlp_best_model)

        print("nb")
        nb = GaussianNB()
        nb.fit(x_train, y_train)

        nb_opinion = nb.predict(x_test)
        nb_acc = accuracy_score(y_test, nb_opinion)
        nb_recall = recall_score(y_test, nb_opinion)

        if nb_recall > nb_best_recall:
            nb_best_recall = nb_recall
            nb_best_recall_model = nb

        nb_accuracies.append(nb_acc)
        nb_recalls.append(nb_recall)
        nb_best_models.append(nb)

    df = DataFrame({
        'knn_acc': knn_accuracies,
        'knn_recall': knn_recalls,
        'svm_acc': svm_accuracies,
        'svm_recall': svm_recalls,
        'mlp_acc': mlp_accuracies,
        'mlp_recall': mlp_recalls,
        'nb_acc': nb_accuracies,
        'nb_recall': nb_recalls,
        'dt_acc': dt_accuracies,
        'dt_recall': dt_recalls
        })

    df.to_csv(out[c], index=False)
