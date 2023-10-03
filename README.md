# gui_supervised_learing_trainer
This Python script provides a graphical user interface (GUI) for performing supervised learning tasks using various classifiers. It utilizes the tkinter library for creating the GUI, and scikit-learn, XGBoost, and imbalanced-learn for machine learning functionalities





## Features

- **Data Loading**: Open CSV files to load datasets for analysis.
- **Data Preprocessing**: Dummify categorical columns, perform feature importance analysis, and conduct feature selection based on importance thresholds.
- **SMOTE & Scaling**: Apply Synthetic Minority Over-sampling Technique (SMOTE) and scaling to handle imbalanced datasets.
- **Hyperparameter Tuning**: Customize hyperparameters for Decision Tree, Random Forest, SVM, k-NN, and XGBoost classifiers.
- **Cross-validation and Evaluation**: Evaluate classifiers using cross-validation and display results for Accuracy, F1 Score, Precision, Recall, and ROC AUC.
- **Train and Test Results**: View detailed performance metrics for training and testing sets.
- **Graphical Analysis**: Visualize Precision-Recall and ROC curves for selected classifiers.

## How to Use

1. Run the script to open the GUI.
2. Open a database file (CSV) using the "Open database" button.
3. Select the target/label column for analysis.
4. Proceed through the different frames to preprocess data, set hyperparameters, and analyze results.
5. Utilize the GUI to evaluate multiple classifiers and compare their performance.

## Requirements

- Python 3.x
- Required Python Libraries: pandas, numpy, matplotlib, scikit-learn, xgboost, imbalanced-learn, tkinter

## Usage

```bash
python gui_supervised_learning_trainer.py
