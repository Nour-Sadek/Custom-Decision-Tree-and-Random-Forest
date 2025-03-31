import numpy as np


class DecisionTreeNode:
    """

    """

    _feature: str
    _splitting_criteria: float
    _samples: list[int]
    _num_samples: int
    _depth: int
    _right: "DecisionTreeNode"
    _left: "DecisionTreeNode"

    def __init__(self, depth: int, samples: list[int]) -> None:
        """Initialize a new <DecisionTreeNode> as a leaf first. Since it is a leaf, it doesn't have right or left nodes,
        and it does not represent a feature from which a branching decision is made.
        <samples> represents a list of indices from the original training dataset that this node contains.
        <depth> is the depth at which this leaf belongs in the Decision Tree.
        """
        self._feature = None
        self._right = None
        self._left = None
        self._splitting_criteria = None
        self._depth = depth
        self._samples = [sample for sample in samples]
        self._num_samples = len(samples)

    @property
    def feature(self) -> str:
        return self._feature

    @property
    def splitting_criteria(self) -> float:
        return self._splitting_criteria

    @property
    def depth(self) -> int:
        return self._depth

    @property
    def right(self) -> "DecisionTreeNode":
        return self._right

    @property
    def left(self) -> "DecisionTreeNode":
        return self._left

    @property
    def samples(self) -> list[int]:
        return self._samples


class RegressionNode(DecisionTreeNode):
    """

    """

    _prediction: float
    _sum_squared_residuals: float

    def __int__(self, depth: int, samples: list[int], y: np.ndarray) -> None:
        """Initialize a <DecisionTreeNode> specifically to be used in Regression Decision Trees.
        <y> are the target variables for the observations of the samples at the indices determined by <samples>. <y> is
        used to determine the <self._prediction>, which is the average of the values in <y>, as well as the
        <self._sum_squared_residuals>, which is the sum of squares of the difference between the observed values (values
        in <y>) and the predicted (average of the values in <y>).
        """
        super().__init__(depth, samples)
        # Calculate the predicted output of this node
        self._prediction = float(np.average(y))
        # Calculate the sum of squared residuals of this node
        self._sum_squared_residuals = 0
        for observed in y:
            self._sum_squared_residuals = self._sum_squared_residuals + pow((observed - self._prediction), 2)

    @property
    def prediction(self) -> float:
        return self._prediction

    @property
    def sum_squared_residuals(self) -> float:
        return self._sum_squared_residuals


class ClassificationNode(DecisionTreeNode):
    """

    """

    _prediction: int
    _distribution: list[float]
    _gini_impurity: float
    _num_classes: int

    def __int__(self, depth: int, samples: list[int], num_classes: int, y: np.ndarray) -> None:
        """Initialize a <DecisionTreeNode> specifically to be used in Classification Decision Trees.
         <y> are the target variables for the observations of the samples at the indices determined by <samples>. <y> is
         used to determine <self._distribution> which represents the probability that the observations in this node
         belong to each of the classes, whose number is determined by <num_classes>, and from that <self._prediction>
         which is the class that has the highest probability. <y> is also used to determine <self._gini_impurity>, which
         is equal to 1 - (prob_class_0)^2 - (prob_class_1)^2 - etc.
        """
        super().__init__(depth, samples)
        self._num_classes = num_classes
        # Determine <self._distribution> and <self._gini_impurity> based on <y>
        self._distribution = []
        self._gini_impurity = 1
        for classification in range(num_classes):
            curr_class_probability = float((y == classification).sum() / y.size)
            self._distribution.append(curr_class_probability)
            self._gini_impurity = self._gini_impurity - pow(curr_class_probability, 2)
        # Determine <self._prediction> of node based on <self._distribution>
        self._prediction = self._distribution.index(max(self._distribution))

    @property
    def prediction(self) -> int:
        return self._prediction

    @property
    def probability_of_classes(self) -> list[float]:
        return self._distribution

    @property
    def gini_impurity(self) -> float:
        return self._gini_impurity
