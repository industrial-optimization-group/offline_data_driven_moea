import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from desdeo_problem.surrogatemodels.SurrogateModels import BaseRegressor, ModelError

class SurrogateKriging(BaseRegressor):
    def __init__(self):
        self.X: np.ndarray = None
        self.y: np.ndarray = None
        self.m = None

    def fit(self, X, y):
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values.reshape(-1, 1)

        # Make a 2-D array if needed
        y = np.atleast_1d(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
        self.m = GaussianProcessRegressor(alpha=0, kernel=kernel, n_restarts_optimizer=9)
        self.m.fit(X, y)

    def predict(self, X):
        #y_mean, y_stdev = np.asarray(self.m.predict(X)[0]).reshape(1,-1)
        y_mean, y_stdev = np.asarray(self.m.predict(X))
        y_mean = (y_mean.reshape(1,-1))
        y_stdev = (y_stdev.reshape(1,-1))
        return (y_mean, y_stdev)

