import numpy as np
from decision_tree_node import *


class DecisionTree:
    """

    """

    _root: DecisionTreeNode
    _max_depth: int
    _min_samples_split: int
    _min_samples_leaf: int
    _max_features: int

    def __init__(self, max_depth: int = 30, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, max_features: int = None) -> None:
        self._root = None
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._max_features = max_features

    def sorted_feature_indices(self, feature: np.ndarray) -> np.ndarray:
        """Return the array of indices that would sort <feature>."""
        return np.argsort(feature)

    def sorted_feature_boundaries(self, feature: np.ndarray) -> np.ndarray:
        """Given a sorted 1D array <feature>, return the indices of the boundaries between values that are changing. If
        all the values in the <feature> array are the same, return None.

        For example, if <feature> = [0, 0, 0, 1, 1, 1, 2, 2, 3, 4], the return value would be [2.5, 5.5, 7.5, 8.5]. If
        <feature> = [0, 0, 0, 0, 0], the return value would be None.
        """

        # Check if values in <feature> are the same
        if feature[0] == feature[-1]:
            return None

        boundaries = np.where(feature[:-1] != feature[1:])[0]
        return boundaries + 0.5

    def decide_node_value(self, X: np.ndarray, temp_node: DecisionTreeNode) -> DecisionTreeNode:
        """<temp_node> is a leaf node. Decide whether it stays as a leaf node or it branches out."""
        pass

    def create_node(self, depth: int, samples: np.ndarray) -> DecisionTreeNode:
        pass

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:

        # self._root = self.create_node(0, np.arange(y_train.shape[0]))
        pass


class ClassificationTree(DecisionTree):
    """

    """

    def create_node(self, depth: int, samples: np.ndarray) -> ClassificationNode:
        return ClassificationNode(depth, samples)


class RegressionTree(DecisionTree):
    """

    """

    def create_node(self, depth: int, samples: np.ndarray) -> RegressionNode:
        return RegressionNode(depth, samples)
