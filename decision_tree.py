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
    _features_names: np.ndarray

    def __init__(self, max_depth: int = 30, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, max_features: int = None) -> None:
        # check of min_samples_split is greater than min_samples_leaf
        self._root = None
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._max_features = max_features
        self._features_names = None

    @classmethod
    def sorted_feature_indices(cls, X: np.ndarray, feature_names: np.ndarray) -> dict[str, np.ndarray]:
        """Return the array of indices that would sort <feature>."""
        indices_of_sorted_features_dict = {}
        for feature_name, features_column in zip(feature_names, X.T):
            indices_of_sorted_features_dict[feature_name] = np.argsort(features_column)
        return indices_of_sorted_features_dict

    @classmethod
    def sorted_feature_boundaries(cls, feature: np.ndarray) -> np.ndarray:
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

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: np.ndarray = None) -> None:
        """Should not be used on its own but only its children classes"""
        pass


class ClassificationTree(DecisionTree):
    """

    """

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: np.ndarray = None) -> None:
        """Create and traverse the tree."""
        if feature_names is None:
            # Let feature names be x[0], x[1], etc
            self._features_names = np.array([f"x[{i}]" for i in range(X_train.shape[1])])
        else:
            self._features_names = feature_names

        indices_of_sorted_features_dict = self.sorted_feature_indices(X_train, self._features_names)
        self._root = ClassificationNode(0, np.arange(y_train.shape[0]))
        return self._fit_helper(X_train, y_train, self._root, len(np.unique(y_train)), indices_of_sorted_features_dict)

    def _fit_helper(self, X: np.ndarray, y: np.ndarray, node: DecisionTreeNode, num_classes: int,
                           indices_of_sorted_features_dict: dict[str, np.ndarray]) -> None:
        if node is not None:
            node = self._decide_node_value(X, y, node, num_classes, indices_of_sorted_features_dict)
            self._fit_helper(X, y, node.left, num_classes, indices_of_sorted_features_dict)
            self._fit_helper(X, y, node.right, num_classes, indices_of_sorted_features_dict)

    def _decide_node_value(self, X: np.ndarray, y: np.ndarray, temp_node: DecisionTreeNode, num_classes: int,
                           indices_of_sorted_features_dict: dict[str, np.ndarray]) -> ClassificationNode:

        gini_impurity, distribution = ClassificationNode.determine_gini_impurity_and_distribution(y, num_classes)
        output_node = ClassificationNode(temp_node.depth, temp_node.samples)
        output_node.set_properties(gini_impurity, distribution)

        # Check if further splits can be made based on <self._max_depth>, <self._min_samples_split> or
        # <self._min_samples_leaf>
        if temp_node.depth >= self._max_depth or len(temp_node.samples) <= self._min_samples_split:
            # This node is a leaf node
            return output_node

        # Make gini_left and gini_right instead of one gini value
        curr_best_split = {"feature": None, "splitting_criteria": None, "samples_left": None, "samples_right": None,
                           "gini_impurity": None}

        for feature_name, features_column in zip(indices_of_sorted_features_dict.keys(), X.T):
            # Get the features column of only the remaining (available) samples in sorted order
            available_samples = temp_node.samples
            sorted_features_indices = indices_of_sorted_features_dict[feature_name]
            sorted_available_samples_indices = sorted_features_indices[np.isin(sorted_features_indices,
                                                                               available_samples)]
            sorted_available_samples = features_column[sorted_available_samples_indices]
            sorted_available_samples_target = y[sorted_available_samples_indices]

            # Divide up the features column
            feature_boundaries = DecisionTree.sorted_feature_boundaries(sorted_available_samples)
            if feature_boundaries is None:
                continue
            else:
                curr_feature_lowest_gini, curr_feature_best_split = ClassificationTree.find_best_feature_split(
                    feature_boundaries, sorted_available_samples_target, num_classes)
                if (curr_best_split["gini_impurity"] is None or curr_best_split["gini_impurity"]
                        > curr_feature_lowest_gini):
                    # check if the split will violate the <self._min_samples_leaf> condition
                    temp_samples_left = sorted_available_samples_indices[
                        np.arange(len(sorted_available_samples_indices)) < curr_feature_best_split]
                    temp_samples_right = sorted_available_samples_indices[
                        np.arange(len(sorted_available_samples_indices)) > curr_feature_best_split]
                    if len(temp_samples_left) < self._min_samples_leaf or len(
                            temp_samples_right) < self._min_samples_leaf:
                        continue
                    curr_best_split["feature"] = feature_name
                    curr_best_split["splitting_criteria"] = curr_feature_best_split
                    curr_best_split["gini_impurity"] = curr_feature_lowest_gini
                    curr_best_split["samples_left"] = temp_samples_left
                    curr_best_split["samples_right"] = temp_samples_right

        # Figure out if temp_node stays as a leaf node, or it splits
        if curr_best_split["gini_impurity"] is not None:  # there was at least one possible split
            output_node.feature = curr_best_split["feature"]
            output_node.splitting_criteria = curr_best_split["splitting_criteria"]
            output_node.left = ClassificationNode(output_node.depth + 1, curr_best_split["samples_left"])
            output_node.right = ClassificationNode(output_node.depth + 1, curr_best_split["samples_right"])

        return output_node

    def predict(self, X_test: np.ndarray) -> np.ndarray:

        # Check if the features present in <X_test> have been trained on by the model
        if X_test.shape[1] != self._features_names.shape:
            raise ValueError("The given dataset does not have the same number of features as the data set that the "
                             "current model has been trained on.")

        # Predict the classifications
        predictions = np.zeros(X_test.shape[0])
        for index, row in enumerate(X_test):
            curr = self._root
            while curr is not None:
                if curr.splitting_criteria is None:
                    predictions[index] = curr.prediction
                    curr = None
                else:
                    curr_feature = curr.feature
                    feature_index = np.where(self._features_names == curr_feature)[0][0]
                    feature_value = row[feature_index]
                    if feature_value < curr.splitting_criteria:
                        curr = curr.left
                    else:
                        curr = curr.right

        return predictions

    @classmethod
    def find_best_feature_split(cls, feature_boundaries: np.ndarray, sorted_available_samples_target: np.ndarray,
                                num_classes: int) -> tuple[float, float]:

        lowest_gini_curr = None
        best_split_curr = None

        for boundary in feature_boundaries:

            # Get all elements with indices less than the boundary, then greater than the boundary
            targets_for_samples_less = sorted_available_samples_target[
                np.arange(len(sorted_available_samples_target)) < boundary]
            targets_for_samples_greater = sorted_available_samples_target[
                np.arange(len(sorted_available_samples_target)) > boundary]

            # Find Total Gini Impurity which is the weighted average of leaf impurities
            gini_impurity_left = \
                ClassificationNode.determine_gini_impurity_and_distribution(targets_for_samples_less, num_classes)[0]
            weighted_gini_impurity_left = (len(targets_for_samples_less) / len(
                sorted_available_samples_target)) * gini_impurity_left

            gini_impurity_right = \
                ClassificationNode.determine_gini_impurity_and_distribution(targets_for_samples_greater, num_classes)[0]
            weighted_gini_impurity_right = (len(targets_for_samples_greater) / len(
                sorted_available_samples_target)) * gini_impurity_right

            weighted_total_gini_impurity = weighted_gini_impurity_right + weighted_gini_impurity_left

            if lowest_gini_curr is None or lowest_gini_curr > weighted_total_gini_impurity:
                lowest_gini_curr = weighted_total_gini_impurity
                best_split_curr = boundary

        return lowest_gini_curr, best_split_curr


class RegressionTree(DecisionTree):
    """

    """

    def _decide_node_value(self, X: np.ndarray, y: np.ndarray, temp_node: DecisionTreeNode, num_classes: int,
                           indices_of_sorted_features_dict: dict[str, np.ndarray]) -> RegressionNode:
        pass
