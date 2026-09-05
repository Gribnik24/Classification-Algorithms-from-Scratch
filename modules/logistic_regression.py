import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class Custom_LogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression algorithm with SGD (stochastic gradient descent) implementation
    Possible 'regularization' param options:
    - none: classic SGDLogisticRegression
    - l1: L1 regularization (with 'alpha' param)
    - l2: L2 regularization (with 'alpha' param)
    - elasticnet: ElasticNet (with 'alpha' and 'l1_ratio' (0 = L2, 1 = L1) params)
    """
    def __init__(self, fit_intercept: bool = True,
                 random_state: int = 42,
                 learning_rate: float = 0.001,
                 epochs: int = 1000,
                 batch_size: int = 32,
                 regularization: str = 'none',  # 'none', 'l1', 'l2', 'elasticnet'
                 alpha: float = 0.01,           # Regularization strength
                 l1_ratio: float = 0.5):        # Only for elasticnet (0 = L2, 1 = L1)
        self.fit_intercept = fit_intercept
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function with overflow protection
        """
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values

        # Reshape y to column vector if needed
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        X_length, n_features = X.shape

        # Add intercept column if needed
        if self.fit_intercept:
            front_addition = np.ones(X_length)
            X = np.column_stack([front_addition, X])
            n_features += 1

        # Initialize weights
        np.random.seed(self.random_state)
        self.weights = np.zeros((n_features, 1))

        # SGD training
        for epoch in range(self.epochs):
            # Shuffle data
            np.random.seed(self.random_state + epoch)
            indices = np.random.permutation(X_length)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch SGD
            for i in range(0, X_length, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                batch_size_actual = len(X_batch)

                # Linear combination
                linear_model = X_batch.dot(self.weights)
                # Sigmoid - transform to probabilities
                predictions = self._sigmoid(linear_model)
                # Error
                error = predictions - y_batch

                # Gradint Log Loss: dw = (1/m) * X^T * (sigmoid(Xw) - y)
                gradients = (1 / batch_size_actual) * X_batch.T.dot(error)

                # Regularization
                if self.regularization == 'none':
                    pass
                elif self.regularization == 'l1':
                    reg_term = self.alpha * np.sign(self.weights)
                    if self.fit_intercept:
                        reg_term[0] = 0
                    gradients += reg_term
                elif self.regularization == 'l2':
                    reg_term = 2 * self.alpha * self.weights
                    if self.fit_intercept:
                        reg_term[0] = 0
                    gradients += reg_term
                elif self.regularization == 'elasticnet':
                    l1_grad = self.alpha * self.l1_ratio * np.sign(self.weights)
                    l2_grad = 2 * self.alpha * (1 - self.l1_ratio) * self.weights
                    if self.fit_intercept:
                        l1_grad[0] = 0
                        l2_grad[0] = 0
                    gradients += l1_grad + l2_grad
                else:
                    raise ValueError('Wrong regularization param.')

                # Gradient clipping
                grad_norm = np.linalg.norm(gradients)
                if grad_norm > 1.0:
                    gradients = gradients / grad_norm

                # Update weights
                self.weights -= self.learning_rate * gradients

        # Extract intercept and coefficients for sklearn compatibility
        if self.fit_intercept:
            self.intercept_ = self.weights[0]
            self.coef_ = self.weights[1:].flatten()
        else:
            self.intercept_ = 0
            self.coef_ = self.weights.flatten()

        return self

    def predict_proba(self, X) -> np.ndarray:
        """
        Return probability estimates for class 1
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.fit_intercept:
            front_addition = np.ones(X.shape[0])
            X = np.column_stack([front_addition, X])

        linear_model = X.dot(self.weights)
        probas_class1 = self._sigmoid(linear_model).flatten()

        # Return both class probabilities [P(y=0), P(y=1)]
        probas_class0 = 1 - probas_class1
        return np.column_stack([probas_class0, probas_class1])

    def predict(self, X) -> np.ndarray:
        """
        Return class labels (0 or 1) based on threshold 0.5
        """
        probas = self.predict_proba(X)[:, 1]
        return (probas >= 0.5).astype(int)