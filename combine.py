from sklearn import naive_bayes, ensemble, linear_model, dummy, svm
from mlxtend import classifier, regressor
import numpy as np
from sklearn.model_selection import cross_val_score
from support import log_loss_resid

class MetaEstimator():
    def __init__(self, method = 'stacking', estimators = None, method_type = None,
                 cutoff_categorical = 10, baseline = False):
        self.method = method
        self.estimators = estimators
        self.method_type = method_type
        self.cutoff_categorical = cutoff_categorical
        self.fitted = None
        self.predictions = None
        self.resid = None
        self.classes = None
        self.baseline = baseline
        self.losses = []

    def get_estimators(self, y):
        self.estimators = []
        if self.method_type == 'regr':
            self.estimators.append(linear_model.ElasticNetCV(random_state=1))
            self.estimators.append(ensemble.GradientBoostingRegressor(random_state=1))
            self.estimators.append(ensemble.RandomForestRegressor(random_state=1))
            if y.shape[0] < 1000:
                self.estimators.append(svm.SVR())
        else:
            if y.shape[0] < 50:
                if len(np.unique(y)) == 2:
                    self.estimators.append(naive_bayes.BernoulliNB())
                    self.estimators.append(linear_model.SGDClassifier(loss='log', random_state=1))  # Logistic Regression
                else:
                    self.estimators.append(naive_bayes.MultinomialNB())
            self.estimators.append(naive_bayes.GaussianNB())
            self.estimators.append(ensemble.RandomForestClassifier(random_state=1))
            if y.shape[0] < 1000:
                self.estimators.append(svm.SVC(probability=True))

    def fit(self, x, y):
        if self.method_type is None:
            is_above = len(np.unique(y, axis=0)) > self.cutoff_categorical
            self.method_type = ('classif','regr')[is_above]
        if self.estimators is None:
            if self.method is not None:
                self.get_estimators(y)
            else:
                if self.method_type == 'regr':
                    self.estimators = linear_model.LassoCV()
                elif self.method_type == 'classif':
                    self.estimators = ensemble.RandomForestClassifier(random_state=1)
        else:
            if self.method_type == 'regr':
                self.estimators = self.estimators[0]
            elif self.method_type == 'classif':
                self.estimators = self.estimators[1]

        if self.method_type == 'classif':
            self.classes = dummy.DummyClassifier().fit(x, y).classes_

        if self.method == 'stacking':
            if self.method_type == 'regr':
                self.fitted = regressor.StackingRegressor(regressors=self.estimators,
                                            meta_regressor=linear_model.LinearRegression()).fit(x, y)
            elif self.method_type == 'classif':
                self.fitted = classifier.StackingClassifier(classifiers=self.estimators,
                            meta_classifier=linear_model.LogisticRegression(random_state = 1)).fit(x, y)
        elif self.method == 'multiplexing':
            for i in self.estimators:
                self.losses.append(np.mean(cross_val_score(i, x, y)))

            self.fitted  = self.estimators[np.argmin(self.losses)].fit(x, y)
        else:
            self.fitted = self.estimators.fit(x, y)

        return self.fitted

    def fit_baseline(self, x, y):
        if self.method_type is None:
            is_above = len(np.unique(y, axis=0)) > self.cutoff_categorical
            self.method_type = ('classif','regr')[is_above]

        if self.method_type == 'regr':
            self.fitted = dummy.DummyRegressor().fit(x, y)
        else:
            self.fitted = dummy.DummyClassifier().fit(x, y)
            self.classes = dummy.DummyClassifier().fit(x, y).classes_

    def predict(self, x):
        if self.method_type == 'regr':
            self.predictions = self.fitted.predict(x)
        else:
            self.predictions = self.fitted.predict_proba(x)

        return self.predictions

    def get_residuals(self, x_train, x_test, y_train, y_test):
        if self.baseline == False:
            self.fit(x_train,y_train)
            if self.method_type == 'regr':
                self.resid = np.power(self.predict(x_test) - y_test, 2)
            else:
                self.resid = log_loss_resid(self.fitted, self.predict(x_test), y_test, self.classes, self.baseline)

        else:
            self.fit_baseline(x_train, y_train)
            if self.method_type == 'regr':
                self.resid = np.power(self.predict(x_test) - y_test, 2)
            else:
                self.resid = log_loss_resid(self.fitted, self.predict(x_test), y_test, self.classes, self.baseline)

        return self.resid
