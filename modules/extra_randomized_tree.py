import pandas as pd
import numpy as np
import scipy.stats
from sklearn.base import ClassifierMixin, BaseEstimator
from typing import Optional, Union, List, Tuple

class Node:
    def __init__(self, X: pd.DataFrame, y: Union[pd.DataFrame, np.array],
                 left_node: Optional['Node'] = None, right_node: Optional['Node'] = None,
                 threshhold: Optional[float] = None, splitting_feature: Optional[str] = None,
                 labels_possibilities: Optional[List[Tuple[Union[int, str], float]]] = None):
        self.X = X
        self.y = y
        self.left_node = left_node
        self.right_node = right_node
        self.threshhold = threshhold
        self.splitting_feature = splitting_feature
        self.labels_possibilities = labels_possibilities
        
class Custom_ExtraTree(BaseEstimator, ClassifierMixin):
    """
    Implementation of one Extra Tree based with randomized threshhold
    Params:
    - max_depth (int, default=None): The maximum depth of the tree
    - min_samples_split (int, default=2): The minimum number of samples required to split an internal node
    - min_impurity_decrease (float, default=0.0): A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
    - random_state (int, default=42): random state
    """
    def __init__(self,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_impurity_decrease: float = 0.0,
                 random_state: int = 42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        
        self._origin_length = None
        self._origin_labels = None
        self._tree = None
    
    @staticmethod
    def compute_gini(y):
        unique_y = y.unique()
        gini = 0
        
        for value in unique_y:
            value_fraction = (y == value).sum() / len(y)
            gini += (1 - value_fraction) * value_fraction
            
        return gini
    
    @staticmethod
    def compute_weighted_gini(gini1, gini2, length1, length2):
        total_length = length1 + length2
        return (length1 / total_length * gini1) + (length2 / total_length * gini2)
    
    def build_tree(self, X: pd.DataFrame, y: Union[pd.DataFrame, np.array],
                   parent_gini: float, parent_depth: int):
        
        # Min samples split stop criteria -> return None as next Node index
        if len(y) < self.min_samples_split:
            return None
        
        # Depth stop criteria -> return None as next Node index
        if self.max_depth is not None and parent_depth > self.max_depth:
            return None
        
        # Random state for feature threshhold
        rng = np.random.RandomState(self.random_state)
        
        # Going through possible features
        best_gain = -np.inf
        best_column = None
        best_threshhold = None
        best_left_mask, best_right_mask = None, None
        best_left_gini, best_right_gini = None, None
        for column in X.columns:
            # Making splitting with random threshhold
            splitting_feature = X[column]
            min_val = np.min(splitting_feature)
            max_val = np.max(splitting_feature)
            
            # Skip if all values are the same
            if min_val == max_val:
                continue
            
            threshhold = rng.uniform(low=min_val, high=max_val)
            left_mask = splitting_feature <= threshhold
            right_mask = splitting_feature > threshhold
            left_samples = y[left_mask]
            right_samples = y[right_mask]
            
            # Calculating left_gini, right_gini and weighted_gini
            left_gini = self.compute_gini(left_samples)
            right_gini = self.compute_gini(right_samples)
            left_samples_length, right_samples_length = len(left_samples), len(right_samples)
            weighted_gini = self.compute_weighted_gini(left_gini, right_gini,
                                                    left_samples_length, right_samples_length)
        
            # Calculating information gain decresing if there is previous Node else just information gain
            if parent_gini is not None:
                gain = parent_gini - weighted_gini
            else:
                gain = weighted_gini
                
            if gain > best_gain:
                best_gain = gain
                best_column = column
                best_threshhold = threshhold
                best_left_mask = left_mask
                best_right_mask = right_mask
                best_left_gini = left_gini
                best_right_gini = right_gini
                
        # Gain increase stop criteria -> return None as next Node index
        if best_gain < self.min_impurity_decrease:
            return None
        
        # If no valid split was found, return None
        if best_column is None:
            return None
        
        # Find labels possibilities if this Node is the last in branch
        labels_possibilities = [(label, np.sum(y == label) / len(y)) for label in self._origin_labels]
        
        # Create new Nodes constructors recursively
        left_node = self.build_tree(X[best_left_mask].drop(best_column, axis=1),
                                    y[best_left_mask],
                                    best_left_gini,
                                    parent_depth + 1)
        
        right_node = self.build_tree(X[best_right_mask].drop(best_column, axis=1),
                                     y[best_right_mask],
                                     best_right_gini,
                                     parent_depth + 1)
        
        return Node(
            X, y,
            right_node=right_node,
            left_node=left_node,
            threshhold=best_threshhold,
            splitting_feature=best_column,
            labels_possibilities=labels_possibilities
            )
                    
    def fit(self, X, y):
        self._origin_length = len(y)
        self._origin_labels = np.sort(y.unique())
        origin_gini = self.compute_gini(y)
        self._tree = self.build_tree(X, y, parent_gini=origin_gini, parent_depth=0)
           
    def predict(self, X):
        preds = []
        for _, row in X.iterrows():
            current = self._tree
            while current is not None:
                pred = max(current.labels_possibilities, key=lambda x: x[1])[0]
                
                if row[current.splitting_feature] <= current.threshhold:
                    current = current.left_node
                else:
                    current = current.right_node
            preds.append(pred)
        return np.array(preds)            
        
    def predict_proba(self, X):
        preds = []
        for _, row in X.iterrows():
            current = self._tree
            while current is not None:
                pred = np.array([pair[1] for pair in current.labels_possibilities])
                
                if row[current.splitting_feature] <= current.threshhold:
                    current = current.left_node
                else:
                    current = current.right_node
            preds.append(pred)
        return np.array(preds)
    
    
class Custom_ExtraTreesClassifier(BaseEstimator, ClassifierMixin):
    """
    Random Forest Classifier without bootstrap based on gini criterion
    Params:
    - max_depth (int, default=None): The maximum depth of the tree
    - min_samples_split (int, default=2): The minimum number of samples required to split an internal node
    - min_impurity_decrease (float, default=0.0): A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
    - n_estimators (int, default=5): The number of trees in the forest
    - random_state (int, default=42): random state
    """
    def __init__(self,
                 max_depth: Optional[int] = None,
                 min_samples_split: int = 2,
                 min_impurity_decrease: float = 0.0,
                 n_estimators: int = 5,
                 random_state: int = 42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        self._forest = []
        
    def fit(self, X, y):
        features_length, features_names = X.shape[1], X.columns
        
        # Fit {n_estimators} trees
        for estimator_num in range(self.n_estimators):
            # Randomly choose the number of features and features themselves to use
            features_to_use = features_names.copy()
            rng = np.random.RandomState(self.random_state + estimator_num)
            perm = rng.permutation(features_names)
            features_slice_num = rng.randint(low=2, high=features_length)
            features_to_use = perm[:features_slice_num]
            X_to_use = X[features_to_use]
            
            # Create and fit extra tree
            extra_tree_model = Custom_ExtraTree(max_depth=self.max_depth,
                                                min_samples_split=self.min_samples_split,
                                                min_impurity_decrease=self.min_impurity_decrease,
                                                random_state=self.random_state + estimator_num
                                                )
            extra_tree_model.fit(X_to_use, y)
            
            self._forest.append(extra_tree_model)
            
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