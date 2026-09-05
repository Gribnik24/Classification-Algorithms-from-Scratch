import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from collections import Counter


class Custom_KNeighborsClassifier(BaseEstimator, ClassifierMixin):
    """
    K-Nearest Neighbors classifier with custom implementation.
    
    Params:
    n_neighbors : int, default=5
        Number of neighbors to use for classification.
    weights : {'uniform', 'distance'}, default='uniform'
        Weight function used in prediction.
    p : float, default=2
        Power parameter for the Minkowski metric.
    """
    def __init__(self, n_neighbors: int = 5, weights: str = 'uniform', p: float = 2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def fit(self, X, y):
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
            
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y).ravel()
        self.classes_ = np.unique(self.y_train)
        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        n_test = X.shape[0]
        probas = np.zeros((n_test, len(self.classes_)))
        
        # Find the distance from point X[i] to all points from X_train
        for point in range(n_test):
            diff = X[point] - self.X_train
            
            # Minkowski distance formula
            dist = np.sum(np.abs(diff) ** self.p, axis=-1)
            dist = np.power(dist, 1.0 / self.p)
            
            # K nearest neighbors indexes
            k_indices = np.argsort(dist, axis=0)[:self.n_neighbors]
            k_labels = self.y_train[k_indices]
            
            # Calculate weights
            if self.weights == 'uniform':
                weights = np.ones(self.n_neighbors)
            elif self.weights == 'distance':
                weights = 1.0 / (dist[k_indices] + 1e-10)
            else:
                raise ValueError(f"Unknown weights type: {self.weights}")
            
            # Aggregate classes
            for j, label in enumerate(self.classes_):
                mask = k_labels == label
                probas[point, j] = np.sum(weights[mask])
                
            # Normalize probabilities
            if np.sum(probas[point]) > 0:
                probas[point] /= np.sum(probas[point])
                
        return probas

    def predict(self, X):
        X = np.asarray(X)
        n_test = X.shape[0]
        predictions = np.zeros(n_test, dtype=int)
        
        # Find the distance from point X[i] to all points from X_train
        for i in range(n_test):
            diff = X[i] - self.X_train
            
            # Minkowski distance formula
            dist = np.sum(np.abs(diff) ** self.p, axis=-1)
            dist = np.power(dist, 1.0 / self.p)
            
            # K nearest neighbors indexes
            k_indices = np.argsort(dist, axis=0)[:self.n_neighbors]
            k_labels = self.y_train[k_indices]
            
            # Choosing class
            votes = Counter(k_labels)
            predictions[i] = votes.most_common(1)[0][0]
            
        return predictions