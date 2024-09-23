from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier


class MLPWrapper(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        layer1=10,
        layer2=10,
        layer3=10,
        learning_rate='constant',
        # max_iter = 50,
        activation='identity'
    ):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.learning_rate = learning_rate
        # self.max_iter = max_iter,
        self.activation= activation

    def fit(self, X, y):
        model = MLPClassifier(
            hidden_layer_sizes=[self.layer1, self.layer2, self.layer3],
            learning_rate=self.learning_rate,
            max_iter = 300,
            activation=self.activation
        )
        model.fit(X, y)
        self.model = model
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
