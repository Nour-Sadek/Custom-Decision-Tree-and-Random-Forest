import numpy as np


class DecisionTreeNode:
    """

    """

    _feature: str
    _splitting_criteria: float
    _samples: np.ndarray
    _depth: int
    _right: "DecisionTreeNode"
    _left: "DecisionTreeNode"

    def __init__(self, depth: int, samples: np.ndarray = None) -> None:
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
        self._samples = samples

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
    def samples(self) -> np.ndarray:
        return self._samples

    @feature.setter
    def feature(self, feature_name: str) -> None:
        if not isinstance(feature_name, str):
            raise ValueError("feature must be a string")
        self._feature = feature_name

    @splitting_criteria.setter
    def splitting_criteria(self, splitting_criteria: float) -> None:
        if not isinstance(splitting_criteria, float):
            raise ValueError("splitting criteria must be a float")
        self._splitting_criteria = splitting_criteria

    @left.setter
    def left(self, left: "DecisionTreeNode") -> None:
        if not isinstance(left, DecisionTreeNode):
            raise ValueError("left node must be an instance from the class DecisionTreeNode")
        self._left = left

    @right.setter
    def right(self, right: "DecisionTreeNode") -> None:
        if not isinstance(right, DecisionTreeNode):
            raise ValueError("left node must be an instance from the class DecisionTreeNode")
        self._right = right

    @samples.setter
    def samples(self, samples: np.ndarray) -> None:
        if samples.ndim != 1 or not np.issubdtype(samples.dtype, np.integer):
            raise ValueError("samples should be a one-dimensional numpy array that only contains integers")
        self._samples = samples


class RegressionNode(DecisionTreeNode):
    """

    """

    _prediction: float
    _sum_squared_residuals: float

    def __int__(self, depth: int, samples: np.ndarray) -> None:
        """Initialize a <DecisionTreeNode> specifically to be used in Regression Decision Trees.
        <y> are the target variables for the observations of the samples at the indices determined by <samples>. <y> is
        used to determine the <self._prediction>, which is the average of the values in <y>, as well as the
        <self._sum_squared_residuals>, which is the sum of squares of the difference between the observed values (values
        in <y>) and the predicted (average of the values in <y>).
        """
        super().__init__(depth, samples)
        self._prediction = None
        self._sum_squared_residuals = None

    @property
    def prediction(self) -> float:
        return self._prediction

    @property
    def sum_squared_residuals(self) -> float:
        return self._sum_squared_residuals

    def set_properties(self, sum_squared_residuals: float, prediction: float) -> None:
        self._prediction = prediction
        self._sum_squared_residuals = sum_squared_residuals

    @classmethod
    def determine_sum_squared_residuals_and_prediction(cls, y: np.ndarray) -> tuple[float, float]:
        prediction = float(np.average(y))
        sum_squared_residuals = 0
        for observed in y:
            sum_squared_residuals = sum_squared_residuals + pow((observed - prediction), 2)
        return sum_squared_residuals, prediction


class ClassificationNode(DecisionTreeNode):
    """

    """

    _prediction: int
    _distribution: np.ndarray
    _gini_impurity: float

    def __int__(self, depth: int, samples: np.ndarray) -> None:
        """Initialize a <DecisionTreeNode> specifically to be used in Classification Decision Trees.
         <y> are the target variables for the observations of the samples at the indices determined by <samples>. <y> is
         used to determine <self._distribution> which represents the probability that the observations in this node
         belong to each of the classes, whose number is determined by <num_classes>, and from that <self._prediction>
         which is the class that has the highest probability. <y> is also used to determine <self._gini_impurity>, which
         is equal to 1 - (prob_class_0)^2 - (prob_class_1)^2 - etc.
        """
        self._prediction = None
        self._distribution = None
        self._gini_impurity = None
        super().__init__(depth, samples)

    @property
    def prediction(self) -> int:
        return self._prediction

    @property
    def distribution(self) -> np.ndarray:
        return self._distribution

    @property
    def gini_impurity(self) -> float:
        return self._gini_impurity

    @prediction.setter
    def prediction(self, prediction: int) -> None:
        if not isinstance(prediction, int):
            raise ValueError("the prediction must be an integer referring to a class")
        self._prediction = prediction

    @gini_impurity.setter
    def gini_impurity(self, gini_impurity: float) -> None:
        if not isinstance(gini_impurity, float):
            raise ValueError("gini_impurity must be a float")
        self._gini_impurity = gini_impurity

    @distribution.setter
    def distribution(self, distribution: np.ndarray) -> None:
        self._distribution = distribution

    def set_properties(self, gini_impurity: float, distribution: np.ndarray) -> None:
        self._prediction = int(np.argmax(distribution))
        self._gini_impurity = gini_impurity
        self._distribution = distribution

    @classmethod
    def determine_gini_impurity_and_distribution(cls, y: np.ndarray, num_classes: int) -> tuple[float, np.ndarray]:

        temp_distribution = []
        gini_impurity = 1
        for classification in range(num_classes):
            curr_class_probability = float((y == classification).sum() / y.size)
            temp_distribution.append(curr_class_probability)
            gini_impurity = gini_impurity - pow(curr_class_probability, 2)
        distribution = np.array(temp_distribution)

        return gini_impurity, distribution
