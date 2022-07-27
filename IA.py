from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.svm import NuSVR
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from Descriptor import Descriptors
import numpy as np
import warnings


class Traning:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        warnings.filterwarnings("ignore")

    def radomflorest(self):
        modelo = RandomForestRegressor(n_estimators=200)
        scores = cross_val_score(modelo, self.x, self.y,
                                 cv=KFold(n_splits=10, random_state=None,
                                          shuffle=False),
                                 scoring='neg_root_mean_squared_error')

        print("RF: {:.2f} | {:.2f}".format(np.mean(scores), np.std(scores)))
        # RSME, Desvio padrão
        modelo.fit(self.x, self.y)
        return modelo

    def svr(self):
        modelo = make_pipeline(StandardScaler(), SVR(kernel='linear',
                                                     C=1, epsilon=5.0))
        scores = cross_val_score(modelo, self.x, self.y,
                                 cv=KFold(n_splits=10, random_state=None,
                                          shuffle=False),
                                 scoring='neg_root_mean_squared_error')

        print("SVR: {:.2f} | {:.2f}".format(np.mean(scores), np.std(scores)))
        # RSME, Desvio padrão
        modelo.fit(self.x, self.y)
        return modelo

    def linear_svr(self):
        modelo = make_pipeline(StandardScaler(), LinearSVR(C=1, epsilon=5.0,
                                                           random_state=0,
                                                           tol=1e-5))
        scores = cross_val_score(modelo, self.x, self.y,
                                 cv=KFold(n_splits=10, random_state=None,
                                          shuffle=False),
                                 scoring='neg_root_mean_squared_error')

        print("LinearSVR: {:.2f} | {:.2f}".format(np.mean(scores),
                                                  np.std(scores)))
        # RSME, Desvio padrão
        modelo.fit(self.x, self.y)
        return modelo

    def nu_svr(self):
        modelo = make_pipeline(StandardScaler(), NuSVR(kernel='linear',
                                                       C=1.0, nu=1.0))
        scores = cross_val_score(modelo, self.x, self.y,
                                 cv=KFold(n_splits=10, random_state=None,
                                          shuffle=False),
                                 scoring='neg_root_mean_squared_error')

        print("NuSVR: {:.2f} | {:.2f}".format(np.mean(scores), np.std(scores)))
        # RSME, Desvio padrão
        modelo.fit(self.x, self.y)

        return modelo

    def linear_regression(self):
        modelo = LinearRegression()

        scores = cross_val_score(modelo, self.x, self.y,
                                 cv=KFold(n_splits=10, random_state=None,
                                          shuffle=False),
                                 scoring='neg_root_mean_squared_error')

        print("LR: {:.2f} | {:.2f}".format(np.mean(scores), np.std(scores)))
        # RSME, Desvio padrão
        modelo.fit(self.x, self.y)

        return modelo
