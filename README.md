# Random Forest Classifier from Scratch

Binary classification with a **Random Forest built from scratch in NumPy**. It covers Gini-based decision trees, bootstrap sampling, random feature selection at every split, and probability averaging across trees. The notebook runs as is on Kaggle or locally and is saved with all of its outputs.

## Notebook

[`Random_Forest_From_Scratch.ipynb`](Random_Forest_From_Scratch.ipynb)

## Pipeline

1. **Dataset**: 100 synthetic 2D points generated with NumPy (fixed seed). The label is 1 inside a noisy circle, so the classes are not linearly separable. No internet connection is needed.
2. **Preprocessing**: `PolynomialFeatures(degree=2)` gives `x1, x2, x1², x1·x2, x2²`, followed by a stratified 70/30 train/test split. Trees split on thresholds, so feature scaling is not needed.
3. **Decision Tree**: Recursive binary splits that minimize the weighted Gini impurity. A node stops splitting at `max_depth`, when it is pure, or when it has too few samples.
4. **Random Forest**: Each tree trains on a bootstrap sample and picks from a random subset of `max_features` features at each split. The forest's prediction is the mean of the trees' leaf probabilities.
5. **Evaluation**: Loss (Binary Cross-Entropy), Accuracy, Precision, Recall and F1 Score on the train and test sets, all computed from the confusion matrix.
6. **Visualization**: Loss against the number of trees, and the learned decision boundary.
7. **Sanity check**: The test metrics are compared against scikit-learn's `RandomForestClassifier` with the same hyperparameters.

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Number of trees | 50 |
| Max depth | 5 |
| Min samples to split | 2 |
| Max features per split | 2 (of 5) |

## Results

| Metric | Train | Test | Sklearn (Test) |
|--------|-------|------|----------------|
| Loss | 0.0494 | 0.1415 | 0.1228 |
| Accuracy | 1.0000 | 0.9667 | 0.9667 |
| Precision | 1.0000 | 1.0000 | 1.0000 |
| Recall | 1.0000 | 0.9091 | 0.9091 |
| F1 Score | 1.0000 | 0.9524 | 0.9524 |

The from-scratch forest matches scikit-learn on every classification metric. The loss falls sharply over the first few trees and then levels off, which shows how averaging many decorrelated trees reduces variance.

## Requirements

```
numpy
matplotlib
scikit-learn
```

## How to Run

Open the notebook in Kaggle, Colab or Jupyter and choose **Run All**.
