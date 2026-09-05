import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Optional, Union

from .tree import Custom_DecisionTreeRegressor

class Custom_GradientBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_impurity_decrease: float = 0.0,
                 n_estimators: int = 5,
                 learning_rate: float = 0.1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        
        self.F_0 = 0.0
        self._models = []

    @staticmethod
    def sigmoid_transform(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X, y):
        n_samples = len(y)
        # F_0
        pos = np.sum(y == 1)
        neg = n_samples - pos
        self.F_0 = np.log((pos + 1e-7) / (neg + 1e-7))
        
        # raw scores
        F = np.full(n_samples, self.F_0)
        
        for _ in range(self.n_estimators):
            # Odds. Negative gradient
            proba = self.sigmoid_transform(F)
            residuals = y - proba
            
            # Fit tree on odds
            tree = Custom_DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_impurity_decrease=self.min_impurity_decrease
            )
            tree.fit(X, residuals)
            self._models.append(tree)
            
            # Upgrade F
            F += self.learning_rate * tree.predict(X)

    def predict(self, X):
        F = np.full(X.shape[0], self.F_0)
        for tree in self._models:
            F += self.learning_rate * tree.predict(X)
        proba = self.sigmoid_transform(F)
        return (proba >= 0.5).astype(int)

    def predict_proba(self, X):
        F = np.full(X.shape[0], self.F_0)
        for tree in self._models:
            F += self.learning_rate * tree.predict(X)
        proba = self.sigmoid_transform(F)
        return np.column_stack([1 - proba, proba])