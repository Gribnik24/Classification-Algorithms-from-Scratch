import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class Custom_NaiveBayes(BaseEstimator, ClassifierMixin):
    """
    Gaussian Naive Bayes classifier with custom implementation.
    Assumes features follow a normal distribution for each class.
    """
    def __init__(self, var_smoothing: float = 1e-9):
        self.var_smoothing = var_smoothing
        
        self.X_train = None
        self.y_train = None
        self.classes_ = None
        self.var_ = None # Mean values for each class and feature
        self.sigma_ = None # Variance values for each class and feature
        self.class_prior_ = None # Prior probability P(y=c)

    def fit(self, X, y):
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
            
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y).ravel()
        n_samples, n_features = self.X_train.shape
        self.classes_ = np.unique(self.y_train)
        n_classes = len(self.classes_)

        # Initialize arrays to store parameters
        self.var_ = np.zeros((n_classes, n_features))
        self.sigma_ = np.zeros((n_classes, n_features))
        self.class_prior_ = np.zeros(n_classes)

        # Calculate statistics for each class
        for i, c in enumerate(self.classes_):
            X_c = self.X_train[self.y_train == c]
            
            self.var_[i] = X_c.mean(axis=0)
            self.sigma_[i] = X_c.var(axis=0)
            
            # Add smoothing to prevent division by zero
            self.sigma_[i] += self.var_smoothing
            
            self.class_prior_[i] = X_c.shape[0] / n_samples

        return self

    def _calculate_log_probabilities(self, X):
        """
        Calculate the log-posterior probability P(y|x) for each class.
        Using log-probabilities to avoid numerical underflow.
        """
        # X shape: (n_test, n_features)
        # var_ shape: (n_classes, n_features)
        # sigma_ shape: (n_classes, n_features)
        
        # Broadcasting: (n_test, 1, n_features) - (1, n_classes, n_features)
        # Result shape: (n_test, n_classes, n_features)
        diff = X[:, np.newaxis, :] - self.var_[np.newaxis, :, :]
        
        # Gaussian PDF formula (log version):
        log_likelihood = -0.5 * np.log(2 * np.pi * self.sigma_) - 0.5 * (diff ** 2 / self.sigma_)
        
        # Sum over features (sum of logs = log of product). Summing along the last axis (features)
        log_likelihood = np.sum(log_likelihood, axis=2)
        
        # Add log-prior: log(P(y))
        log_posterior = log_likelihood + np.log(self.class_prior_)
        
        return log_posterior

    def predict_proba(self, X):
        log_posterior = self._calculate_log_probabilities(X)
        
        # Softmax function to convert log-probabilities to probabilities
        exp_posterior = np.exp(log_posterior - np.max(log_posterior, axis=1, keepdims=True))
        
        return exp_posterior / np.sum(exp_posterior, axis=1, keepdims=True)

    def predict(self, X):
        log_posterior = self._calculate_log_probabilities(X)
        
        # Return the class with the highest log-probability
        return self.classes_[np.argmax(log_posterior, axis=1)]