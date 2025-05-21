# Custom-Decision-Tree-and-Random-Forest

This repository compares custom implementations of Decision Trees and Random Forests for 
Classification and Regression problems to scikit-learn's implementations.

The comparison is done on the dummy wine dataset, which is made up of 13 features, 178 observations, and 3 classes, and 
is accessed through the sklearn.datasets package.

# Results

Run the evaluation script:

    python evaluate.py

### Data Preprocessing
The wine dataset has been split into train and test sets and each have been mean-centered.

### Training the wine dataset using the Custom implementation of Decision Trees

#### Classification Decision Tree
The custom Classification Decision Tree has been trained on the training portion of the wine dataset using the default settings:
- max depth = 30
- minimum samples required for splitting = 2
- minimum samples per leaf = 1

Since no regularization to prevent overfitting has been applied, the accuracy of predicting the labels of the training data 
should be 100%, which is what was obtained.
Accuracy of the model on the test set was 83.3%

This was the Classification Decision Tree that was generated:
![Image](https://github.com/user-attachments/assets/48ab0976-4c71-4083-8da4-8559ab410206)

#### Regression Decision Tree
Even though the predictions are class labels and so discrete and not continuous, fitting the data to a Regression Decision Tree 
would still work as a debugging strategy since the tree is made to overfit the data and so the mean at the leaf nodes will 
be equivalent to discrete labels and so an accuracy of 100% on training data should be expected, which was obtained. 
Accuracy on test data was 91.67%

The custom Regression Decision tree has been trained on the training portion of the wine dataset using the same default settings as the classification tree.

This was the Regression Decision Tree that was generated:
![Image](https://github.com/user-attachments/assets/7ed62d3b-ae7d-4390-8934-468389df27b2)

### Training the wine dataset using sciket-learn's implementation of Decision Trees

#### Classification Decision Tree
sklearn.tree.DecisionTreeClassifier was used to fit the model to the training data using the default settings, in addition 
to the parameter random_state=20.

The accuracy on the training dataset was indeed 100%, and on the test set it was 88.89%.

This was the Classification Decision Tree that was generated:
![Image](https://github.com/user-attachments/assets/2f471f2c-9d1d-40ff-934c-5a890589720b)

#### Regression Decision Tree
sklearn.tree.DecisionTreeRegressor was used to fit the model to the training data using the default settings, in addition 
to the parameter random_state=20.

The accuracy on the training dataset was indeed 100%, and on the test set it was 84.19%.

This was the Regression Decision Tree that was generated:
![Image](https://github.com/user-attachments/assets/8a508f2a-cc31-4950-bbf4-f045b106ff28)

### Training the wine dataset using the Custom implementation of Random Forest for Classification
The custom Random Forest for Classification has been trained on the training set of the wine dataset using the default 
settings for Decision Trees, as well as these two extra parameters:
- Max features to be considered for each Decision tree was set to 6
- Number of Decision Trees to be generated was set to 100

Accuracy of this model on the test set was 100%, however its Out-Of-Bag error was 11.27%.

### Training the wine dataset using the Custom implementation of Random Forest for Classification
sklearn.ensemble.RandomForestClassifier was used to fit the model to the training data, setting n_estimators = 100.

Accuracy of this model on the test set was 100%, and its Out-Of-Bag error was 0%.

# Repository Structure

This repository contains:

    decision_tree_node.py: Implementation of the RegressionNode and ClassificationNode classes that inherit from the custom 
    DecisionTreeNode parent class, which are used in building the custom Decision Tree classes
    
    decision_tree.py: Implementation of the RegressionTree and ClassificationTree classes that inherit from the custom DecisionTree
    parent class that can be used to fit training data on their own as well as used in building the custom Random Forest classes

    random_forest.py: Implementation of the RegressionRandomForest and ClassificationRandomForest classes that inherit from the 
    custom RandomForest parent class.

    evaluate.py: Main script for performing comparisons on the dummy wine dataset and generating plots

    requirements.txt: List of required Python packages

Python 3.12 version was used
