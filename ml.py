import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

TOTAL_POINTS = 19509
data = pd.read_csv("./data/new_data.csv")
print(data)

X = data.drop('Class', axis=1)
y = data['Class']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

zeros = 0
ones = 0
for i in y_test:
    if i == 0:
        zeros += 1

    if i == 1:
        ones += 1
print(zeros, ones)

print("Data dropped")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Data scalled")

svm_model = SVC(kernel='linear')
svm_model.fit(X_train_scaled, y_train)

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_scaled, y_train)

print("Data fitted")

y_pred_svm = svm_model.predict(X_test_scaled)
y_pred_knn = knn_model.predict(X_test_scaled)


print("Data predicted")
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_kvm = accuracy_score(y_test, y_pred_knn)
print(f'Accuracy SVM: {accuracy_svm * 100:.2f}%')
print(f'Accuracy KVM: {accuracy_kvm * 100:.2f}%')

print("Confusion Matrix")
cm_knn = confusion_matrix(y_test, y_pred_knn)
cm_svm = confusion_matrix(y_test, y_pred_svm)
print("KNN\n", cm_knn)
print("SVM\n", cm_svm)

TP_KNN = cm_knn[1][1]
FN_KNN = cm_knn[1][0]
FP_KNN = cm_knn[0][1]
TN_KNN = cm_knn[0][0]
knn_points = TP_KNN * 30 + FN_KNN * -50 + TN_KNN * 1 + FP_KNN * -5
print(f"KNN points: {knn_points}")
knn_percentage = knn_points / TOTAL_POINTS
print("Acuracia ponderada KNN:", knn_percentage)
knn_recall = TP_KNN / (FN_KNN + TP_KNN)
print(f"KNN RECALL: ", knn_recall)

TP_SVM = cm_svm[1][1]
FN_SVM = cm_svm[1][0]
FP_SVM = cm_svm[0][1]
TN_SVM = cm_svm[0][0]
svm_points = TP_SVM * 30 + FN_SVM * -50 + TN_SVM * 1 + FP_SVM * -5
print(f"SVM points: {svm_points}")
svm_percentage = svm_points / TOTAL_POINTS
print("Acuracia ponderada SVM: ", svm_percentage)
svm_recall = TP_SVM / (FN_SVM + TP_SVM)

print(f"SVM RECALL: ", svm_recall)

