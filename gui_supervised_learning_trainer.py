import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc, precision_recall_curve)
from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import tkinter as tk
from tkinter import ttk, filedialog as fd, messagebox

# Create a GUI root
root = tk.Tk()

# Specify the title and dimensions to root
root.title('Supervised learning trainer')
root.geometry('1800x1000')

#create labelframes
label_frame_input = ttk.Labelframe(root, text='Inputs', width=300, height=350)
label_frame_input.grid(row=0, column=0, sticky='n')
label_frame_dummies = ttk.Labelframe(root, text='Dummies', width=300, height=350)
label_frame_dummies.grid(row=0, column=1, sticky='n')
label_frame_feature_importance = ttk.Labelframe(root, text='Feature importance', width=300, height=350)
label_frame_feature_importance.grid(row=0, column=2, sticky='n')
label_frame_feature_selection = ttk.Labelframe(root, text='Feature selection', width=300, height=350)
label_frame_feature_selection.grid(row=0, column=3, sticky='n')
label_frame_feature_SMOTE_scaling = ttk.Labelframe(root, text='SMOTE & scaling', width=300, height=350)
label_frame_feature_SMOTE_scaling.grid(row=0, column=4, sticky='n')
label_frame_hyperparameters = ttk.Labelframe(root, text='Hyperparameters', width=300, height=350)
label_frame_hyperparameters.grid(row=0, column=5, sticky='n')
label_frame_results = ttk.Labelframe(root, text='Results cross-validation', width=1200, height=350)
label_frame_results.grid(row=0, column=6, sticky='n')
label_frame_results_train = ttk.Labelframe(root, text='Results train', width=1200, height=350)
label_frame_results_train.grid(row=1, column=6, sticky='n')
label_frame_results_test = ttk.Labelframe(root, text='Results test', width=1200, height=350)
label_frame_results_test.grid(row=2, column=6, sticky='n')

# Create an open file button
open_button = tk.Button(label_frame_input, text='Open database', command=lambda: OpenFile())
open_button.grid(row=0, column=0, sticky='nw')

#create column selection option
def create_label_frame_input():
    """
    Create and configure the input frame with a label and listbox for selecting the target/label column.

    Returns:
        tk.Frame: The configured input frame.
        tk.StringVar: Variable to store the selected column.
        tk.Listbox: Listbox widget for displaying available columns.
    """
    global column_selection
    global listbox_columns
    # Label for selecting target/label column
    tk.Label(label_frame_input, font="none 7 bold", text="Select target/label column:").grid(row=3, column=0, sticky='w') # place widget with empty text, will be filled later o
    # Variable to store the selected column
    column_selection = tk.StringVar()
    column_selection.set([])
    # Listbox for displaying available columns
    listbox_columns = tk.Listbox(label_frame_input, listvariable=column_selection)
    listbox_columns.grid(row=4, column=0, sticky='nw', rowspan = 10)

def OpenFile():
    """
    Open a file dialog to choose a CSV file and read the data.

    Returns:
    - str: Location of the selected database file.
    - list: Columns of the database.
    """
    global name
    global data
    name = fd.askopenfilename(initialdir="", filetypes=(("Text File", "*.csv"), ("All Files", "*.*")), title="Choose a file.")
    data = pd.read_csv(name, error_bad_lines=False)
    data = data.sample(n=10000, random_state=42).copy()
    list(data.columns)
    column_selection.set(list(data.columns))
    tk.Button(label_frame_dummies, text='process', command=lambda: dummifying()).grid(row=0, column=0, sticky='w')



def clear_label_frame(name_label_frame):
    """
    Clear all label widgets in the specified label frame.

    Args:
    - name_label_frame (ttk.Labelframe): The label frame to be cleared.
    """
    for widget in name_label_frame.grid_slaves():
        if widget.winfo_class() == "Label":
            widget.destroy()

def dummifying():
    """
    Quick process of the database using a Chunk function and return some basic details of the selected column.

    Returns:
    - pd.DataFrame: Used database file.
    - str: Selected column.
    - float: Process time.
    - int: Rows in the database.
    - float: Average of the selected column.
    - float: Maximum value of the selected column.
    - float: Minimum value of the selected column.
    """
    clear_label_frame(label_frame_dummies)
    try:# checks if there was a column selected
        selection = listbox_columns.get(listbox_columns.curselection())
    except:
        tk.messagebox.showerror("warning", "Select column then Process database")
        return

    global X
    global y
    global X_encoded

    X = data.drop([selection], axis=1)
    y = data[selection]

    # Because we are trying to find the most significant correlations with another categorical variable ('Default'), it is very important to ensure we encode our categorical to ensure accurate feature selection.
    # One-hot encode all object (categorical) columns
    X_encoded = pd.get_dummies(X, columns=X.select_dtypes(include=['object']).columns, drop_first=True)
    total_rows = data.shape[0]
    # tk.Label(label_frame_dummies, font="none 7 bold", text="Used file: " + str(name)).grid(row=1, column=0, sticky='w') # place widget with empty text, will be filled later o
    tk.Label(label_frame_dummies, font="none 7 bold", text="Target column: " + str(selection)).grid(row=2, column=0, sticky='w') # place widget with empty text, will be filled later o
    tk.Label(label_frame_dummies, font="none 7 bold", text="Rows: " + str(total_rows)).grid(row=3, column=0, sticky='w')  # place widget with empty text, will be filled later
    tk.Label(label_frame_dummies, font="none 7 bold", text="Dummyfied columns:").grid(row=4, column=0, sticky='w')  # place widget with empty text, will be filled later
    row_number = 5

    for dummy in X.select_dtypes(include=['object']):
        tk.Label(label_frame_dummies, font="none 7", text=str(dummy)).grid(row=row_number, column=0,sticky='w') # place widget with empty text, will be filled later
        row_number = row_number + 1

    tk.Button(label_frame_feature_importance, text='process', command=lambda: feature_importance()).grid(row=0, column=0, sticky='w')

def feature_importance():
    """
    Calculate and display feature importances using RandomForestRegressor.
    """
    rf_regressor = RandomForestRegressor(n_estimators=10, random_state=42)
    rf_regressor.fit(X_encoded, y)
    feature_importances = rf_regressor.feature_importances_
    global importance_df
    importance_df = pd.DataFrame({'Feature': X_encoded.columns, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).reset_index()

    for index, row in importance_df.iterrows():
        feature = row["Feature"]
        importance = round(row["Importance"],3)
        tk.Label(label_frame_feature_importance, font="none 7", text=str(index) + " " + str(importance) + " " + feature).grid(row=index +1, column=0,sticky='w') # place widget with empty text, will be filled later
    # Create a StringVar to hold the default value
    tk.Button(label_frame_feature_selection, text='process', command=lambda: feature_selection()).grid(row=0, column=0, sticky='w')
    global default_value_selection
    default_value_selection = tk.StringVar()
    default_value_selection.set(0.01)  # Set the default value to "0.01"

    # Create an Entry widget and set its textvariable to the default value
    entry = tk.Entry(label_frame_feature_selection, textvariable=default_value_selection)
    entry.grid(row=1, column=0, sticky='w')


def feature_selection():
    """
    Perform feature selection based on the specified importance threshold.
    """
    value_selection = float(default_value_selection.get())
    clear_label_frame(label_frame_feature_selection)
    df_selection = importance_df[importance_df['Importance'] > value_selection]
    columns_to_filter= df_selection['Feature'].tolist()
    global X_encoded_filtered
    X_encoded_filtered = X_encoded.loc[:, columns_to_filter]
    tk.Label(label_frame_feature_selection, font="none 7 bold", text="Selected features:").grid(row=2, column=0, sticky='w')  # place widget with empty text, will be filled later o
    for index, row in df_selection.iterrows():
        feature = row["Feature"]
        importance = round(row["Importance"], 3)
        tk.Label(label_frame_feature_selection, font="none 7", text=str(index) + " " + str(importance) + " " + feature).grid(row=index + 3, column=0, sticky='w')  # place widget with empty text, will be filled later
    tk.Button(label_frame_feature_SMOTE_scaling, text='process', command=lambda: SMOTE_scaling()).grid(row=0, column=0, sticky='w')

    global checkbox_scaling_var
    checkbox_scaling_var = tk.BooleanVar(value=True)
    tk.Checkbutton(label_frame_feature_SMOTE_scaling, text="scaling", variable=checkbox_scaling_var).grid(row=2, column=0, sticky='w')
    global checkbox_SMOTE_var
    checkbox_SMOTE_var = tk.BooleanVar()
    tk.Checkbutton(label_frame_feature_SMOTE_scaling, text="SMOTE", variable=checkbox_SMOTE_var).grid(row=3, column=0, sticky='w')
    unique_features = y.unique()
    value_counts = y.value_counts()
    # Iterate over unique values and their counts using a for loop
    row = 4
    for value, count in value_counts.items():
        tk.Label(label_frame_feature_SMOTE_scaling, font="none 7", text=f"Feature value {value}: {count} occurrences").grid(row=row, column=0, sticky='w')
        row = row + 1


class Dropdown:
    """
    Create a dropdown menu in the specified label frame.

    Args:
    - label_frame (ttk.Labelframe): The label frame where the dropdown will be created.
    - options (tuple): Options for the dropdown.
    - default_value: Default value for the dropdown.
    - row (int): Row position in the label frame.
    - column (int): Column position in the label frame.
    - label (str, optional): Label for the dropdown.

    Attributes:
    - label_frame (ttk.Labelframe): The label frame where the dropdown is created.
    - options (tuple): Options for the dropdown.
    - default_value: Default value for the dropdown.
    - combo_var (tk.StringVar): StringVar to store the selected item.
    - combo (ttk.Combobox): Combobox widget.
    """
    def __init__(self, label_frame, options, default_value, row, column, label=None):
        self.label_frame = label_frame
        self.options = options
        self.default_value = default_value

        # Create a Label if specified
        if label:
            self.label = ttk.Label(label_frame, text=label)
            self.label.grid(row=row, column=column, padx=5, pady=5)

        # Create a StringVar to store the selected item
        self.combo_var = tk.StringVar()

        # Create a Combobox widget
        self.combo = ttk.Combobox(label_frame, textvariable=self.combo_var)
        self.combo['values'] = options
        self.combo.set(default_value)
        self.combo.grid(row=row, column=column + 1, padx=5, pady=5)

    def get_selected_value(self):
        return self.combo_var.get()

def SMOTE_scaling():
    """
    Apply SMOTE and scaling to the selected features.
    """
    global X_scaled
    global y_resampled
    if checkbox_SMOTE_var.get():
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_encoded_filtered, y)
        y_resampled.value_counts(normalize=True)
    else:
        X_resampled, y_resampled = X_encoded_filtered, y
    if checkbox_scaling_var.get():
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_resampled)
    else:
        X_scaled = X_resampled

    #Hyperparameters Decision Tree
    tk.Button(label_frame_hyperparameters, text='process & show results', command=lambda: run_models()).grid(row=0, column=0, sticky='w')
    tk.Label(label_frame_hyperparameters, font="none 7 bold", text="Decision Tree").grid(row=1, column=0, sticky='w')
    global dropdown_1, dropdown_2, dropdown_3, dropdown_4, dropdown_5, dropdown_6, dropdown_7, dropdown_8

    # Options for the dropdowns
    options_1 = ("None", 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
    # Create instances of the Dropdown class in different LabelFrames
    dropdown_1 = Dropdown(label_frame_hyperparameters, options_1, options_1[4], row=2, column=0, label="MaxDepth:")
    #Hyperparameters Random Forest
    tk.Label(label_frame_hyperparameters, font="none 7 bold", text="Random Forest (E)").grid(row=3, column=0, sticky='w')
    options_2 = ("None", 40, 30, 20, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
    dropdown_2 = Dropdown(label_frame_hyperparameters, options_2, options_2[3], row=4, column=0, label="N_estimators:")
    options_3 = ("None", 10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
    dropdown_3 = Dropdown(label_frame_hyperparameters, options_3, options_3[4], row=5, column=0, label="MaxDepth:")
    #Hyperparameters SVM
    tk.Label(label_frame_hyperparameters, font="none 7 bold", text="SVM").grid(row=6, column=0, sticky='w')
    options_4 = (0.001, 0.01, 1, 10, 100, 1000)
    dropdown_4 = Dropdown(label_frame_hyperparameters, options_4, options_4[2], row=7, column=0, label="C:")
    options_5 = ("linear", "poly", "rbf")
    dropdown_5 = Dropdown(label_frame_hyperparameters, options_5, options_5[0], row=8, column=0, label="Kernel:")
    #Hyperparameters k-NN
    tk.Label(label_frame_hyperparameters, font="none 7 bold", text="k-NN").grid(row=9, column=0, sticky='w')
    options_6 = (10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
    dropdown_6 = Dropdown(label_frame_hyperparameters, options_6, options_6[0], row=10, column=0, label="N_neighbors:")
    #Hyperparameters X-boost
    tk.Label(label_frame_hyperparameters, font="none 7 bold", text="XGBoost (Ensemble & Gradiet Boosting)").grid(row=11, column=0, sticky='w')
    options_7 = (50, 75, 100, 250, 500, 1000, 2000)
    dropdown_7 = Dropdown(label_frame_hyperparameters, options_7, options_7[0], row=12, column=0, label="N_estimators:")
    #Hyperparameters Cross-validation
    tk.Label(label_frame_hyperparameters, font="none 7 bold", text="Cross-validation").grid(row=13, column=0, sticky='w')
    options_8 = (10, 9, 8, 7, 6, 5, 4, 3, 2, 1)
    dropdown_8 = Dropdown(label_frame_hyperparameters, options_8, options_8[5], row=14, column=0, label="CV:")


    tk.Label(label_frame_hyperparameters, font="none 7 bold", text="Graphs (computational power intensive)").grid(row=15, column=0,sticky='w')
    global checkbox_precision_recall_var
    checkbox_precision_recall_var = tk.BooleanVar(value=False)
    tk.Checkbutton(label_frame_hyperparameters, text="Precision/Recall graph", variable=checkbox_precision_recall_var).grid(row=16, column=0, sticky='w')
    global checkbox_ROC_var
    checkbox_ROC_var = tk.BooleanVar(value=False)
    tk.Checkbutton(label_frame_hyperparameters, text="ROC graph", variable=checkbox_ROC_var).grid(row=17, column=0, sticky='w')


def run_models():
    """
    Run multiple classifiers and display the evaluation metrics.
    """
    selected_value_1 = int(dropdown_1.get_selected_value()) if dropdown_1.get_selected_value() != "None" else None
    selected_value_2 = int(dropdown_2.get_selected_value()) if dropdown_2.get_selected_value() != "None" else None
    selected_value_3 = int(dropdown_3.get_selected_value()) if dropdown_3.get_selected_value() != "None" else None
    selected_value_4 = float(dropdown_4.get_selected_value())
    selected_value_5 = dropdown_5.get_selected_value()
    selected_value_6 = int(dropdown_6.get_selected_value()) if dropdown_6.get_selected_value() != "None" else None
    selected_value_7 = int(dropdown_7.get_selected_value()) if dropdown_7.get_selected_value() != "None" else None
    selected_value_8 = int(dropdown_8.get_selected_value()) if dropdown_8.get_selected_value() != "None" else None


    classifiers = {
        'Decision Tree': DecisionTreeClassifier(max_depth=selected_value_1, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=selected_value_2, max_depth=selected_value_3 ,random_state=42),
        'SVM': SVC(kernel= selected_value_5, C=selected_value_4, random_state=42, probability=True),  # Enable probability estimates
        'k-NN': KNeighborsClassifier(n_neighbors=selected_value_6),
        'Naive Bayes': GaussianNB(),
        'XGBoost': xgb.XGBClassifier(n_estimators=selected_value_7, random_state=42),

    }

    # Initialize dictionaries to store evaluation metric results
    results = {
        'Classifier': [],
        'Accuracy': [],
        'F1 Score': [],
        'Precision': [],
        'Recall': [],
        'ROC AUC': []
    }

    # Loop through classifiers and calculate various evaluation metrics using cross-validation
    for classifier_name, classifier in classifiers.items():

        accuracy_scores = cross_val_score(classifier, X_scaled, y_resampled, cv=selected_value_8, scoring='accuracy')
        f1_scores = cross_val_score(classifier, X_scaled, y_resampled, cv=selected_value_8, scoring='f1')
        precision_scores = cross_val_score(classifier, X_scaled, y_resampled, cv=selected_value_8, scoring='precision')
        recall_scores = cross_val_score(classifier, X_scaled, y_resampled, cv=selected_value_8, scoring='recall')
        roc_auc_scores = cross_val_score(classifier, X_scaled, y_resampled, cv=selected_value_8, scoring='roc_auc')

        # Take the mean of cross-validation scores
        accuracy_mean = round(np.mean(accuracy_scores),3)
        f1_mean = round(np.mean(f1_scores),3)
        precision_mean = round(np.mean(precision_scores),3)
        recall_mean = round(np.mean(recall_scores),3)
        roc_auc_mean = round(np.mean(roc_auc_scores),3)

        results['Classifier'].append(classifier_name)
        results['Accuracy'].append(accuracy_mean)
        results['F1 Score'].append(f1_mean)
        results['Precision'].append(precision_mean)
        results['Recall'].append(recall_mean)
        results['ROC AUC'].append(roc_auc_mean)

    # Transpose the data to have classifiers as column names
    tree = ttk.Treeview(label_frame_results, show="headings")

    # Include 'Classifier' in the list of columns
    columns = tuple(results.keys())
    tree["columns"] = columns

    # Configure columns using grid
    for i, metric in enumerate(columns):
        tree.column(metric, anchor="w", width=100)
        tree.heading(metric, text=metric)
        tree.grid(row=0, column=i, sticky="nsew")

    # Insert data into the Treeview
    for i in range(len(results['Classifier'])):
        classifier = results['Classifier'][i]
        row_data = [results[metric][i] for metric in columns]
        tree.insert("", "end", values=tuple(row_data))

    # Configure treeview to expand with the window
    tree.grid(row=1, column=0, sticky="nsew")

    # Configure the labeled frame to expand with the window
    label_frame_results.grid_rowconfigure(1, weight=1)
    label_frame_results.grid_columnconfigure(0, weight=1)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_resampled, test_size=0.2, random_state=42)
    # Initialize dictionaries to store evaluation metric results
    results_train = {
        'Classifier': [],
        'Accuracy': [],
        'F1 Score': [],
        'Precision': [],
        'Recall': [],
        'ROC AUC': []
    }

    results_test = {
        'Classifier': [],
        'Accuracy': [],
        'F1 Score': [],
        'Precision': [],
        'Recall': [],
        'ROC AUC': []
    }

    # Loop through classifiers and calculate various evaluation metrics using for training and testing
    for classifier_name, classifier in classifiers.items():
        # Train the model on the training set
        classifier.fit(X_train, y_train)

        # Evaluate the model on the training set
        y_train_pred = classifier.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred)
        train_recall = recall_score(y_train, y_train_pred)
        train_roc_auc = roc_auc_score(y_train, y_train_pred)

        # Make predictions on the test set
        y_test_pred = classifier.predict(X_test)

        # Evaluate the model on the test set
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_recall = recall_score(y_test, y_test_pred)
        test_roc_auc = roc_auc_score(y_test, y_test_pred)

        # Store the results for both training and test sets
        results_train['Classifier'].append(classifier_name)
        results_train['Accuracy'].append(train_accuracy)
        results_train['F1 Score'].append(train_f1)
        results_train['Precision'].append(train_precision)
        results_train['Recall'].append(train_recall)
        results_train['ROC AUC'].append(train_roc_auc)

        results_test['Classifier'].append(classifier_name)
        results_test['Accuracy'].append(test_accuracy)
        results_test['F1 Score'].append(test_f1)
        results_test['Precision'].append(test_precision)
        results_test['Recall'].append(test_recall)
        results_test['ROC AUC'].append(test_roc_auc)

        tree2 = ttk.Treeview(label_frame_results_train, show="headings")

        # Include 'Classifier' in the list of columns
        columns = tuple(results_train.keys())
        tree2["columns"] = columns

        # Configure columns using grid
        for i, metric in enumerate(columns):
            tree2.column(metric, anchor="w", width=100)
            tree2.heading(metric, text=metric)
            tree2.grid(row=0, column=i, sticky="nsew")

        # Insert data into the Treeview
        for i in range(len(results_train['Classifier'])):
            classifier = results_train['Classifier'][i]
            row_data = [results_train[metric][i] for metric in columns]
            tree2.insert("", "end", values=tuple(row_data))

        # Configure treeview to expand with the window
        tree2.grid(row=1, column=0, sticky="nsew")

        #fill test:
        tree3 = ttk.Treeview(label_frame_results_test, show="headings")

        # Include 'Classifier' in the list of columns
        columns = tuple(results_test.keys())
        tree3["columns"] = columns

        # Configure columns using grid
        for i, metric in enumerate(columns):
            tree3.column(metric, anchor="w", width=100)
            tree3.heading(metric, text=metric)
            tree3.grid(row=0, column=i, sticky="nsew")

        # Insert data into the Treeview
        for i in range(len(results_test['Classifier'])):
            classifier = results_test['Classifier'][i]
            row_data = [results_test[metric][i] for metric in columns]
            tree3.insert("", "end", values=tuple(row_data))

        # Configure treeview to expand with the window
        tree3.grid(row=1, column=0, sticky="nsew")


    tk.Label(label_frame_results, font="none 7 bold", text="Accuracy: The proportion of correctly classified instances(Number of Correct Prediction/Total Number of Predictions).").grid(row=2, column=0, sticky='w')
    tk.Label(label_frame_results, font="none 7 bold", text="Precision: The proportion of true positive predictions among all positive predictions (True Positives/True Positives+False Positives)").grid(row=3, column=0, sticky='w')
    tk.Label(label_frame_results, font="none 7 bold", text="Recall (Sensitivity): The proportion of true positive predictions among all actual positive (True Positives/True Positives+False Negatives)").grid(row=4, column=0, sticky='w')
    tk.Label(label_frame_results, font="none 7 bold", text="F1 Score: The harmonic mean of precision and recall. It balances precision and recall.").grid(row=5, column=0, sticky='w')
    tk.Label(label_frame_results, font="none 7 bold", text="ROC: The ability of the model to discriminate between positive and negative instances.").grid(row=6, column=0, sticky='w')
    tk.Label(label_frame_results, font="none 7 bold", text="ROC Interpretation Guidelines:").grid(row=7, column=0, sticky='w')
    tk.Label(label_frame_results, font="none 7 bold", text="ROC:0.5 to 0.7: Poor discrimination. The model is not effectively separating the classes.").grid(row=8, column=0, sticky='w')
    tk.Label(label_frame_results, font="none 7 bold", text="ROC:0.7 to 0.8: Acceptable discrimination.").grid(row=9, column=0, sticky='w')
    tk.Label(label_frame_results, font="none 7 bold", text="ROC:0.8 to 0.9: Good discrimination.").grid(row=10, column=0, sticky='w')
    tk.Label(label_frame_results, font="none 7 bold", text="ROC:Above 0.9: Excellent discrimination.").grid(row=11, column=0, sticky='w')

    if checkbox_precision_recall_var.get() or checkbox_ROC_var.get():
        # Initialize figures
        plt.figure(figsize=(10, 5))

    if checkbox_precision_recall_var.get():
        # Loop through classifiers and calculate various evaluation metrics using cross-validation
        for classifier_name, classifier in classifiers.items():
            # Fit the classifier
            classifier.fit(X_scaled, y_resampled)

            # Cross-validated predicted probabilities
            y_scores = cross_val_predict(classifier, X_scaled, y_resampled, cv=selected_value_8, method='predict_proba')[:, 1]


            # Compute precision-recall curve
            precision, recall, _ = precision_recall_curve(y_resampled, y_scores)
            # Plot Precision-Recall curve
            plt.subplot(1, 2, 1)
            plt.plot(recall, precision, label=classifier_name)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend()

    if  checkbox_ROC_var.get():
        for classifier_name, classifier in classifiers.items():
            # Fit the classifier
            classifier.fit(X_scaled, y_resampled)

            # Cross-validated predicted probabilities
            y_scores = cross_val_predict(classifier, X_scaled, y_resampled, cv=selected_value_8, method='predict_proba')[:, 1]

            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_resampled, y_scores)

            # Compute AUC for ROC
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve
            plt.subplot(1, 2, 2)
            plt.plot(fpr, tpr, label=f'{classifier_name} (AUC = {roc_auc:.2f})')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend()

    if checkbox_precision_recall_var.get() or checkbox_ROC_var.get():
        plt.tight_layout()
        plt.show()

create_label_frame_input()
root.mainloop()