import numpy as np
import random
from decision_tree_node import *


class DecisionTree:
    """A custom Decision Tree classifier that acts as a base class for the ClassificationTree and
    RegressionTree classes"""

    _root: DecisionTreeNode
    _max_depth: int
    _min_samples_split: int
    _min_samples_leaf: int
    _max_features: int
    _features_names: np.ndarray

    def __init__(self, max_depth: int = 30, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, max_features: int = None) -> None:
        """Initialize a new DecisionTree instance with _max_depth <max_depth>, _min_samples_split <min_samples_split>,
        _min_samples_leaf <min_samples_leaf>, and max_features <max_features>.

        This class is never expected to be called directly.

        - max_depth represents the max depth of the created tree where branching will stop for the node that reaches
        <max_depth>.
        - min_samples_leaf represents the minimum number of samples that a leaf should have where branching will not
        happen if it leads to nodes that have less than <min_samples_leaf>.
        - min_samples_split represents the minimum number of samples that a node is required to have to be able to be
        further split into leaf nodes; if a node has samples less than <min_samples_leaf>, it will be labelled as a leaf
        node and no further splits will be made.
        - max_features represents the maximum number of features that will be used when deciding to split the node. If
        <max_features> is None, then all the features will be considered at every split.
        """
        self._root = None
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._max_features = max_features
        self._features_names = None

    @property
    def root(self) -> DecisionTreeNode:
        return self._root

    @classmethod
    def sorted_feature_indices(cls, X: np.ndarray, feature_names: np.ndarray) -> dict[str, np.ndarray]:
        """Return the array of indices that would sort <feature>."""
        indices_of_sorted_features_dict = {}
        for feature_name, features_column in zip(feature_names, X.T):
            indices_of_sorted_features_dict[str(feature_name)] = np.argsort(features_column)
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
        """Create the decision tree based on <X_train> and <y_train>."""
        if feature_names is None:
            # Let feature names be x[0], x[1], etc
            self._features_names = np.array([f"x[{i}]" for i in range(X_train.shape[1])])
        else:
            self._features_names = feature_names

        indices_of_sorted_features_dict = self.sorted_feature_indices(X_train, self._features_names)
        self._root.samples = np.arange(y_train.shape[0])
        if (isinstance(self._root, ClassificationNode)):
            return self._fit_helper(X_train, y_train, self._root, len(np.unique(y_train)),
                                    indices_of_sorted_features_dict)
        else:
            return self._fit_helper(X_train, y_train, self._root, None, indices_of_sorted_features_dict)

    def _fit_helper(self, X: np.ndarray, y: np.ndarray, node: DecisionTreeNode, num_classes: int,
                    indices_of_sorted_features_dict: dict[str, np.ndarray]) -> None:
        """Helper method for <self.fit> where it allows the recursive creation of the tree."""
        if node is not None:
            self._decide_node_value(X, y, node, num_classes, indices_of_sorted_features_dict)
            self._fit_helper(X, y, node.left, num_classes, indices_of_sorted_features_dict)
            self._fit_helper(X, y, node.right, num_classes, indices_of_sorted_features_dict)

    def _decide_node_value(self, X: np.ndarray, y: np.ndarray, node: DecisionTreeNode, num_classes: int,
                           indices_of_sorted_features_dict: dict[str, np.ndarray]) -> None:
        """Decide the node value of <node> during training after the <self.fit> function call. It decides based on the
        <X> and <y> training datasets and the values currently stored in <node> which refer to the available samples on
        which the splitting will be decided on and a dictionary <indices_of_sorted_feature_dict> that stores information
        about how each feature's values are sorted."""
        pass

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return a 1D array assigning predictions for the observations in <X_test> of shape nb_observations x
        nb_features by traversing the tree that has been created during training and assigning the predicted value that
        is stored in the leaf node that is reached upon full traversal.
        This function is expected to be called only after training.
        """

        # Check if the features present in <X_test> have been trained on by the model
        if X_test.shape[1] != self._features_names.shape[0]:
            raise ValueError("The given dataset does not have the same number of features as the data set that the "
                             "current model has been trained on.")

        # Predict the classifications
        predictions = np.ones(X_test.shape[0]) * -1
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

    def print_tree(self, node: DecisionTreeNode, prefix="", is_left=True):
        """Print the tree out onto the console as a way to visualize the tree that has been created during training.
        This function is only expected to be called after training."""
        if node is None:
            return

        # Prefix for current node
        connector = "└── " if not is_left else "├── "
        print(prefix + connector + self.format_node(node))

        # Update the prefix for children
        if node.left or node.right:
            # Extend the branch for the left child
            if node.left:
                self.print_tree(node.left, prefix + ("│   " if is_left else "    "), is_left=True)
            # Extend the branch for the right child
            if node.right:
                self.print_tree(node.right, prefix + ("│   " if is_left else "    "), is_left=False)

    def format_node(self, node: DecisionTreeNode) -> str:
        """Return a string that represents the information that will be displayed for the node when printing out the
        tree after calling <self.print_tree>. It will include gini impurity for ClassificationTree and squared error
        for Regression Tree for inner nodes.

        Common information that will be printed out for both trees are:
        - For inner nodes: the feature and splitting criteria, the number of samples, and the prediction of that node.
        - For leaf nodes: the number of samples and the prediction."""
        pass


class ClassificationTree(DecisionTree):
    """A custom Classification Tree classifier."""

    def __init__(self, max_depth: int = 30, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, max_features: int = None) -> None:
        """Initialize a new ClassificationTree instance with _max_depth <max_depth>, _min_samples_split
        <min_samples_split>, _min_samples_leaf <min_samples_leaf>, and max_features <max_features>. The root of the
        tree is created to be an instance of the ClassificationNode class."""
        super().__init__(max_depth, min_samples_split, min_samples_leaf, max_features)
        self._root = ClassificationNode(0)

    def _decide_node_value(self, X: np.ndarray, y: np.ndarray, node: ClassificationNode, num_classes: int,
                           indices_of_sorted_features_dict: dict[str, np.ndarray]) -> None:
        available_samples = node.samples
        gini_impurity, distribution = ClassificationNode.determine_gini_impurity_and_distribution(y[available_samples],
                                                                                                  num_classes)
        node.set_properties(gini_impurity, distribution)

        # Check if further splits can be made based on <self._max_depth>, <self._min_samples_split> or
        # <self._min_samples_leaf>
        if node.depth >= self._max_depth or len(node.samples) < self._min_samples_split:
            # This node is a leaf node
            return

        # Make gini_left and gini_right instead of one gini value
        curr_best_split = {"feature": None, "splitting_criteria": None, "samples_left": None, "samples_right": None,
                           "gini_impurity": None}

        # If the samples in the node are all the same, no need to further split
        if np.all(y[available_samples] == y[available_samples][0]):
            return

        curr_random_features = list(indices_of_sorted_features_dict.keys())
        # Choose the features to consider randomly if _max_features is not None and is less than the provided features
        if self._max_features is not None and self._max_features < X.shape[1]:
            curr_random_features = random.sample(curr_random_features, self._max_features)

        for feature_name, features_column in zip(curr_random_features, X.T):
            # Get the features column of only the remaining (available) samples in sorted order
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
                    temp_samples_left = sorted_available_samples_indices[:int(curr_feature_best_split) + 1]
                    temp_samples_right = sorted_available_samples_indices[int(curr_feature_best_split) + 1:]
                    if len(temp_samples_left) < self._min_samples_leaf or len(
                            temp_samples_right) < self._min_samples_leaf:
                        continue
                    curr_best_split["feature"] = feature_name
                    curr_best_split["splitting_criteria"] = (sorted_available_samples[int(curr_feature_best_split)] + sorted_available_samples[int(curr_feature_best_split) + 1]) / 2
                    curr_best_split["gini_impurity"] = curr_feature_lowest_gini
                    curr_best_split["samples_left"] = temp_samples_left
                    curr_best_split["samples_right"] = temp_samples_right

        # Figure out if temp_node stays as a leaf node, or it splits
        if curr_best_split["gini_impurity"] is not None:  # there was at least one possible split
            node.feature = curr_best_split["feature"]
            node.splitting_criteria = curr_best_split["splitting_criteria"]
            node.left = ClassificationNode(node.depth + 1, curr_best_split["samples_left"])
            node.right = ClassificationNode(node.depth + 1, curr_best_split["samples_right"])

    @classmethod
    def find_best_feature_split(cls, feature_boundaries: np.ndarray, sorted_available_samples_target: np.ndarray,
                                num_classes: int) -> tuple[float, float]:
        """Return the best splitting criteria and its corresponding gini impurity value, which should be the lowest
        possible out of all the possible splits that can be done. This is done based on the <feature_boundaries> that
        stores the indices of the possible boundaries that can be considered and the <sorted_available_samples_target>
        that store the target values for the samples sorted based on the values of the current feature.

        Return None for both if there is no possible split for the samples considered."""

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

    def format_node(self, node: ClassificationNode) -> str:
        if node.feature is None:
            return f"Leaf(prediction={node.prediction}, samples={len(node.samples)})"
        else:
            return (f"Node({node.feature} <= {node.splitting_criteria:.3f}, "
                    f"gini={node.gini_impurity:.3f}, samples={len(node.samples)}, "
                    f"prediction={node.prediction})")


class RegressionTree(DecisionTree):
    """A custom Regression Tree classifier."""

    def __init__(self, max_depth: int = 30, min_samples_split: int = 2,
                 min_samples_leaf: int = 1, max_features: int = None) -> None:
        """Initialize a new ClassificationTree instance with _max_depth <max_depth>, _min_samples_split
        <min_samples_split>, _min_samples_leaf <min_samples_leaf>, and max_features <max_features>. The root of the
        tree is created to be an instance of the RegressionNode class."""
        super().__init__(max_depth, min_samples_split, min_samples_leaf, max_features)
        self._root = RegressionNode(0)

    def _decide_node_value(self, X: np.ndarray, y: np.ndarray, node: RegressionNode, num_classes: int,
                           indices_of_sorted_features_dict: dict[str, np.ndarray]) -> None:

        available_samples = node.samples
        sum_squared_residuals, prediction = RegressionNode.determine_sum_squared_residuals_and_prediction(y[available_samples])
        node.set_properties(sum_squared_residuals, prediction)

        # Check if further splits can be made based on <self._max_depth>, <self._min_samples_split> or
        # <self._min_samples_leaf>
        if node.depth >= self._max_depth or len(node.samples) < self._min_samples_split:
            # This node is a leaf node
            return

        # Make gini_left and gini_right instead of one gini value
        curr_best_split = {"feature": None, "splitting_criteria": None, "samples_left": None, "samples_right": None,
                           "sum_squared_residuals": None}

        # If the samples in the node are all the same, no need to further split
        if np.all(y[available_samples] == y[available_samples][0]):
            return

        curr_random_features = list(indices_of_sorted_features_dict.keys())
        # Choose the features to consider randomly if _max_features is not None and is less than the provided features
        if self._max_features is not None and self._max_features < X.shape[1]:
            curr_random_features = random.sample(curr_random_features, self._max_features)

        for feature_name, features_column in zip(curr_random_features, X.T):
            # Get the features column of only the remaining (available) samples in sorted order
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
                curr_feature_lowest_ssr, curr_feature_best_split = RegressionTree.find_best_feature_split(
                    feature_boundaries, sorted_available_samples_target)
                if (curr_best_split["sum_squared_residuals"] is None or curr_best_split["sum_squared_residuals"]
                        > curr_feature_lowest_ssr):
                    # check if the split will violate the <self._min_samples_leaf> condition
                    temp_samples_left = sorted_available_samples_indices[:int(curr_feature_best_split) + 1]
                    temp_samples_right = sorted_available_samples_indices[int(curr_feature_best_split) + 1:]
                    if len(temp_samples_left) < self._min_samples_leaf or len(
                            temp_samples_right) < self._min_samples_leaf:
                        continue
                    curr_best_split["feature"] = feature_name
                    curr_best_split["splitting_criteria"] = (sorted_available_samples[int(curr_feature_best_split)] + sorted_available_samples[int(curr_feature_best_split) + 1]) / 2
                    curr_best_split["sum_squared_residuals"] = curr_feature_lowest_ssr
                    curr_best_split["samples_left"] = temp_samples_left
                    curr_best_split["samples_right"] = temp_samples_right

        # Figure out if temp_node stays as a leaf node, or it splits
        if curr_best_split["sum_squared_residuals"] is not None:  # there was at least one possible split
            node.feature = curr_best_split["feature"]
            node.splitting_criteria = curr_best_split["splitting_criteria"]
            node.left = RegressionNode(node.depth + 1, curr_best_split["samples_left"])
            node.right = RegressionNode(node.depth + 1, curr_best_split["samples_right"])

    @classmethod
    def find_best_feature_split(cls, feature_boundaries: np.ndarray,
                                sorted_available_samples_target: np.ndarray) -> tuple[float, float]:
        """Return the best splitting criteria and its corresponding sum squared residual value, which should be the
        lowest possible out of all the possible splits that can be done. This is done based on the <feature_boundaries>
        that stores the indices of the possible boundaries that can be considered and the
        <sorted_available_samples_target> that store the target values for the samples sorted based on the values of the
        current feature.

        Return None for both if there is no possible split for the samples considered."""

        lowest_ssr_curr = None
        best_split_curr = None

        for boundary in feature_boundaries:

            # Get all elements with indices less than the boundary, then greater than the boundary
            targets_for_samples_less = sorted_available_samples_target[
                np.arange(len(sorted_available_samples_target)) < boundary]
            targets_for_samples_greater = sorted_available_samples_target[
                np.arange(len(sorted_available_samples_target)) > boundary]

            # Find Total Sum of Squared Residuals
            ssr_left = \
                RegressionNode.determine_sum_squared_residuals_and_prediction(targets_for_samples_less)[0]

            ssr_right = \
                RegressionNode.determine_sum_squared_residuals_and_prediction(targets_for_samples_greater)[0]

            total_ssr = ssr_left + ssr_right

            if lowest_ssr_curr is None or lowest_ssr_curr > total_ssr:
                lowest_ssr_curr = total_ssr
                best_split_curr = boundary

        return lowest_ssr_curr, best_split_curr

    @classmethod
    def format_node(cls, node: RegressionNode) -> str:
        if node.feature is None:
            return f"Leaf(prediction={node.prediction}, samples={len(node.samples)})"
        else:
            return (f"Node({node.feature} <= {node.splitting_criteria:.3f}, "
                    f"squared_error={node.sum_squared_residuals / len(node.samples):.3f}, samples={len(node.samples)}, "
                    f"prediction={node.prediction:.3f})")
