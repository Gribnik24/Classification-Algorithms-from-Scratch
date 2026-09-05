import pandas as pd
import numpy as np
import scipy.stats
from math import sqrt
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Optional, Union

from .tree import Custom_DecisionTreeClassifier

class Custom_RandomForestClassifier(BaseEstimator, ClassifierMixin):
    """
    Random Forest Classifier without bootstrap based on gini criterion
    Params:
    - max_depth (int, default=None): The maximum depth of the tree
    - min_samples_split (int, default=2): The minimum number of samples required to split an internal node
    - min_impurity_decrease (float, default=0.0): A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
    - n_estimators (int, default=5): The number of trees in the forest
    - max_features ({'sqrt', int}, default='sqrt'): The number of features to consider when looking for the best split
    - random_state (int, default=42): random state
    """
    def __init__(self,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_impurity_decrease: float = 0.0,
                 n_estimators: int = 5,
                 max_features: Union[str, int] = 'sqrt',
                 random_state: int = 42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        
        self._forest = []

    def fit(self, X, y):
        # Get features names and choose the number of features to use
        features_names = X.columns
        if self.max_features == 'sqrt':
            features_slice_num = round(sqrt(X.shape[1]))
        else:
            features_slice_num = self.max_features
        
        # Fit {n_estimators} trees
        for estimator_num in range(self.n_estimators):
            # Delete extra features from df
            features_to_use = features_names.copy()
            rng = np.random.RandomState(self.random_state + estimator_num)
            perm = rng.permutation(features_names)
            features_to_use = perm[:features_slice_num]
            X_to_use = X[features_to_use]
            
            # Create and fit tree
            tree_model = Custom_DecisionTreeClassifier(max_depth=self.max_depth,
                                                       min_samples_split=self.min_samples_split,
                                                       min_impurity_decrease=self.min_impurity_decrease
                                                       )
            tree_model.fit(X_to_use, y)
            
            self._forest.append(tree_model)
    
    def predict(self, X):
        # Calculate preds column for each tree
        tree_preds = np.column_stack([tree.predict(X) for tree in self._forest])
        # Shrink using stats.mode to find the most voted class for each sample
        most_voted_preds, _ = scipy.stats.mode(tree_preds, axis=1)
        return most_voted_preds.squeeze()

    def predict_proba(self, X):
        # Calculate preds-proba with the possible column for each tree
        tree_preds = np.stack([tree.predict_proba(X) for tree in self._forest])
        # Shrink using np.mean to find the weighted sum of probabilities of each tree for classes in every sample
        most_voted_preds = np.mean(tree_preds, axis=0)
        return most_voted_preds.squeeze()