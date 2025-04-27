import numpy as np
import random
from scipy import stats
from decision_tree import *


class RandomForest:
    """A custom Random Forest classifier."""

    _decision_trees: list[DecisionTree]
    _max_features: int
    _num_trees = int
    _max_depth: int
    _min_samples_split: int
    _min_samples_leaf: int
    _features_names: np.ndarray

    def __init__(self, max_features: int, num_trees: int = 100, max_depth: int = 30, min_samples_split: int = 2,
                 min_samples_leaf: int = 1) -> None:
        """Initialize a new RandomForest instance with _max_features <max_features>, _num_trees <num_trees>,
        _max_depth <max_depth>, _min_samples_split <min_samples_split>, and _min_samples_leaf <min_samples_leaf>.

        This class is never expected to be called directly.

        The _max_depth, _min_samples_split, _min_samples_leaf, and _max_features are features for the DecisionTrees that
        will be a part of this RandomForest, with _num_trees setting the number of trees that will make up the
        RandomForest.

        _decision_trees is a list of the DecisionTrees.
        """
        self._decision_trees = []
        self._max_features = max_features
        self._num_trees = num_trees
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._features_names = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: np.ndarray = None) -> None:
        """Create the Random Forest based on <X_train> and <y_train>."""
        for tree in self._decision_trees:
            tree_specific_samples = np.array(random.choices(np.arange(y_train.shape[0]), k=y_train.shape[0]))
            tree.fit(X_train, y_train, feature_names, tree_specific_samples)

    def get_predictions(self, X_test: np.ndarray) -> np.ndarray:
        """Return a numpy array that stacks the predictions done on <X_test> for every Decision Tree in the Random
        Forest."""
        predictions = []
        for tree in self._decision_trees:
            predictions.append(tree.predict(X_test))
        stacked_predictions = np.stack(predictions)
        return stacked_predictions

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Return the aggregate prediction made by the Decision Trees."""
        pass

    def out_of_bag_predictions(self, X_train: np.ndarray) -> list[list[float]]:
        """Return the Out of Bag predictions for every sample in <X_train> for every tree that that sample wasn't a
        part of its training dataset."""
        all_samples_predictions = []
        for sample in range(X_train.shape[0]):
            sample_predictions = []
            for tree in self._decision_trees:
                if sample not in tree.root.samples:
                    reshaped_sample_array = X_train[sample].reshape(1, -1)
                    sample_predictions.append(float(tree.predict(reshaped_sample_array)))
            all_samples_predictions.append(sample_predictions)
        return all_samples_predictions

    def out_of_bag_error(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """Return the Out of Bag error determined by comparing the aggregate prediction of samples in <X_train> made by
        the Decision Trees to their actual values given in <Y_train>."""
        pass


class RegressionRandomForest(RandomForest):
    """A custom Random Forest classifier that is made up of Regression Decision Trees."""

    def __init__(self, max_features: int, num_trees: int = 100, max_depth: int = 30, min_samples_split: int = 2,
                 min_samples_leaf: int = 1) -> None:
        """Initialize a new RegressionRandomForest instance with _max_features <max_features>, _num_trees <num_trees>,
        _max_depth <max_depth>, _min_samples_split <min_samples_split>, and _min_samples_leaf <min_samples_leaf>. The
        _decision_trees list will be initialized as a list of RegressionTree instances."""
        super().__init__(max_features, num_trees, max_depth, min_samples_split, min_samples_leaf)
        self._decision_trees = [RegressionTree(max_depth, min_samples_split, min_samples_leaf,
                                               max_features) for _ in range(num_trees)]

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        stacked_predictions = self.get_predictions(X_test)
        predictions_by_average = np.mean(stacked_predictions, axis=0)
        return predictions_by_average


class ClassificationRandomForest(RandomForest):
    """A custom Random Forest classifier that is made up of Classification Decision Trees."""

    def __init__(self, max_features: int, num_trees: int = 100, max_depth: int = 30, min_samples_split: int = 2,
                 min_samples_leaf: int = 1) -> None:
        """Initialize a new ClassificationRandomForest instance with _max_features <max_features>, _num_trees
        <num_trees>, _max_depth <max_depth>, _min_samples_split <min_samples_split>, and _min_samples_leaf
        <min_samples_leaf>. The _decision_trees list will be initialized as a list of ClassificationTree instances."""
        super().__init__(max_features, num_trees, max_depth, min_samples_split, min_samples_leaf)
        self._decision_trees = [ClassificationTree(max_depth, min_samples_split, min_samples_leaf,
                                                   max_features) for _ in range(num_trees)]

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        stacked_predictions = self.get_predictions(X_test)
        predictions_by_majority = stats.mode(stacked_predictions, axis=0)[0]
        return predictions_by_majority

    def out_of_bag_error(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        out_of_bag_predictions = self.out_of_bag_predictions(X_train)
        out_of_bag_aggregate_predictions = []
        for sample_predictions in out_of_bag_predictions:
            if not sample_predictions:
                out_of_bag_aggregate_predictions.append(None)
            else:
                out_of_bag_aggregate_predictions.append(stats.mode(sample_predictions)[0])
        # Calculate the accuracy where values are not None (only happens when a training sample was included in every
        # tree
        out_of_bag_aggregate_predictions = np.array(out_of_bag_aggregate_predictions)
        considered_samples = out_of_bag_aggregate_predictions != None
        comparison = (y_train[considered_samples] == out_of_bag_aggregate_predictions[considered_samples])
        return 1 - np.mean(comparison)
