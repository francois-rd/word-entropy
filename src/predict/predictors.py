from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC, LinearSVC
from scipy.stats import geom


class Predictor:
    def __init__(self, random_state):
        self._best = None
        self._rng = random_state

    @property
    def name(self): raise NotImplementedError

    def find_best(self, X, y):
        """
        Finding the best model could involve anything from trying out a single
        version model with sensible hyper-parameter defaults all the way to
        extensive hyper-parameter sweeps with Bayesian optimization.

        :param X: training data, (n_samples, n_features)
        :param y: training labels, (n_samples)
        """
        raise NotImplementedError

    def predict(self, X):
        """
        Use the best model to predict labels for a test set.

        :param X: test data, (n_samples, n_features)
        :return: predicted labels, (n_samples)
        """
        return self._best.predict(X)


class OVRLogisticRegression(Predictor):
    @property
    def name(self): return r"Log$_{\text{OVR}}$"

    def find_best(self, X, y):
        self._best = LogisticRegression(
            class_weight='balanced',
            multi_class='ovr',
            random_state=self._rng
        ).fit(X, y)


class MultinomialLogisticRegression(Predictor):
    @property
    def name(self): return r"Log$_{\text{Multi}}$"

    def find_best(self, X, y):
        self._best = LogisticRegression(
            class_weight='balanced',
            multi_class='multinomial',
            random_state=self._rng
        ).fit(X, y)


class SVM(Predictor):
    @property
    def name(self): return r"SVM$_{\text{RBF}}$"

    def find_best(self, X, y):
        #  https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
        self._best = SVC(
            class_weight='balanced',
            random_state=self._rng
        ).fit(X, y)


class LinearSVM(Predictor):
    @property
    def name(self): return r"SVM$_{\text{Lin}}$"

    def find_best(self, X, y):
        #  https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
        self._best = LinearSVC(
            dual=False,
            class_weight='balanced',
            random_state=self._rng
        ).fit(X, y)


class RandomForest(Predictor):
    @property
    def name(self): return "RF"

    def find_best(self, X, y):
        param_dists = {
            "max_depth": [None] + list(range(1, 21)),
            "min_samples_split": geom(0.25, loc=1),
            "min_samples_leaf": geom(0.5, loc=1)
        }
        self._best = RandomizedSearchCV(RandomForestClassifier(
            class_weight='balanced',
            random_state=self._rng
        ), param_dists).fit(X, y)


class MajorityClass(Predictor):
    @property
    def name(self): return "Majority"

    def find_best(self, X, y):
        self._best = DummyClassifier(random_state=self._rng).fit(X, y)
