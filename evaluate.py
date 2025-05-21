from decision_tree import *
import random
from random_forest import *
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)


# Center the train and test sets
def center_data(X: np.ndarray) -> np.ndarray:
    """Return a modified numpy array where <X> is centered"""
    cols_mean = np.mean(X, axis=0)
    cols_mean_mat = cols_mean * np.ones((X.shape[0], X.shape[1]))
    centered_data = X - cols_mean_mat
    return centered_data


# Center the train and test datasets separately
X_train, X_test = center_data(X_train), center_data(X_test)


"""Output of the custom model using Classification Tree"""
custom_classification_tree = ClassificationTree()
custom_classification_tree.fit(X_train, y_train)

print("Custom Classification Decision Tree:\n")
custom_classification_tree.print_tree(custom_classification_tree.root)
print("\n\n")

# Using the trained model to predict classifications of the test dataset
custom_classification_tree_predictions = custom_classification_tree.predict(X_test)
comparison = (y_test == custom_classification_tree_predictions)
custom_accuracy_class = round(float(comparison.sum() / y_test.size) * 100, 2)
print(f"Accuracy of the Custom Classification Decision Tree on the test dataset: {custom_accuracy_class}%")

# Using the trained model to predict classifications of the train dataset
# Debugging step: accuracy of training dataset should be 100%
custom_classification_tree_predictions_train = custom_classification_tree.predict(X_train)
comparison_train = (y_train == custom_classification_tree_predictions_train)
custom_accuracy_class_train = round(float(comparison_train.sum() / y_train.size) * 100, 2)
print(f"Accuracy of the Custom Classification Decision Tree on the train dataset: {custom_accuracy_class_train}%\n")

"""Output of the custom model using Regression Tree"""
custom_regression_tree = RegressionTree(max_depth=100)
custom_regression_tree.fit(X_train, y_train)

# Print out the Regression Tree that was trained, which should fit the training data perfectly
print("Custom Regression Decision Tree:\n")
custom_regression_tree.print_tree(custom_regression_tree.root)
print("\n\n")

# Using the trained model to predict classifications of the test dataset
custom_regression_tree_predictions = custom_regression_tree.predict(X_test)
regress_comparison = (y_test == custom_regression_tree_predictions)
custom_accuracy_regress = round(float(regress_comparison.sum() / y_test.size) * 100, 2)
print(f"Accuracy of the Custom Regression Decision Tree on the test dataset: {custom_accuracy_regress}%")

# Using the trained model to predict classifications of the train dataset
# Debugging step: accuracy of training dataset should be 100%
custom_regression_tree_predictions_train = custom_regression_tree.predict(X_train)
regress_comparison_train = (y_train == custom_regression_tree_predictions_train)
custom_accuracy_regress_train = round(float(regress_comparison_train.sum() / y_train.size) * 100, 2)
print(f"Accuracy of the Custom Regression Decision Tree on the test dataset: {custom_accuracy_regress_train}%\n")

"""Output of the sklearn model using Classification Tree"""

sklearn_classification = tree.DecisionTreeClassifier(random_state=20)
sklearn_classification = sklearn_classification.fit(X_train, y_train)
score_class_sklearn_test = round(sklearn_classification.score(X_test, y_test) * 100, 2)
score_class_sklearn_train = round(sklearn_classification.score(X_train, y_train) * 100, 2)

print("sklearn Classification Decision Tree:\n")
print(f"Accuracy of sklearn Classification Decision Tree on the test dataset: {score_class_sklearn_test}%")
print(f"Accuracy of sklearn Classification Decision Tree on the train dataset: {score_class_sklearn_train}%\n")

fig = plt.figure(figsize=(20, 16))
tree.plot_tree(sklearn_classification)
plt.plot()

"""Output of the sklearn model using Regression Tree"""

sklearn_regression = tree.DecisionTreeRegressor(random_state=20)
sklearn_regression = sklearn_regression.fit(X_train, y_train)
score_regress_sklearn_test = round(sklearn_regression.score(X_test, y_test) * 100, 2)
score_regress_sklearn_train = round(sklearn_regression.score(X_train, y_train) * 100, 2)

print("sklearn Regression Decision Tree:\n")
print(f"Accuracy of sklearn Regression Decision Tree on the test dataset: {score_regress_sklearn_test}%")
print(f"Accuracy of sklearn Regression Decision Tree on the train dataset: {score_regress_sklearn_train}%\n")

fig = plt.figure(figsize=(20, 10))
tree.plot_tree(sklearn_regression)
plt.plot()

"""Output of the custom Random Forest model using an ensemble of Classification Trees"""

random.seed(20)

random_forest_custom = ClassificationRandomForest(6, num_trees=100)
random_forest_custom.fit(X_train, y_train)
random_forest_custom_pred = random_forest_custom.predict(X_test)
accuracy_custom = round(accuracy_score(y_test, random_forest_custom_pred) * 100, 2)
oob_error_custom = round(random_forest_custom.out_of_bag_error(X_train, y_train) * 100, 2)
print("Custom Random Forest Classifier:\n")
print(f"Accuracy of the custom Random Forest model using an ensemble of Classification Trees on the test dataset: {accuracy_custom}%")
print(f"Out-of-Bag Error of the custom Random Forest model using an ensemble of Classification Trees on the test dataset: {oob_error_custom}%\n")

"""Output of sklearn's Random Forest Classifier"""

random_forest_sklearn = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=20)
random_forest_sklearn.fit(X_train, y_train)
random_forest_sklearn_pred = random_forest_sklearn.predict(X_test)
accuracy = round(accuracy_score(y_test, random_forest_sklearn_pred) * 100)
oob_error_sklearn = round((1 - random_forest_sklearn.oob_score) * 100, 2)
print("sklearn Random Forest Classifier:\n")
print(f"Accuracy of sklearn's Random Forest Classifier on the test dataset: {accuracy}%")
print(f"Out-of-Bag Error of sklearn's Random Forest Classifier on the test dataset: {oob_error_sklearn}%\n")
