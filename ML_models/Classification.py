##############################
#   Author:: Adriana Galáns
#   Music Taste Project
############################

# Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, RocCurveDisplay, \
    classification_report


# Data loading
def load_data(data_path, train_test=False):
    # CSVs reading and training - test data generation
    data = pd.read_csv(data_path)
    y = data.iloc[:, 0]
    X = data.iloc[:, 1:]
    if train_test:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    else:
        return X, y, data


# Cross Validation for three ML classification models, Support Vector Machine, Decision Tree &
# Multiple Logistic Regression
def cross_valid_models(data_path, modelML):
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=2)
    X, y, data = load_data(data_path)
    # Enumerate splits
    outer_results = list()
    for train, test in cv_outer.split(data):
        # Split data
        X_train, X_test = X[train, :], X[test, :]
        y_train, y_test = y[train], y[test]

        # Configure the cross-validation procedure
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)

        # Model selection and search space definition
        model = ''
        space = ''
        if modelML == 'Multiple Logistic Regression':
            # Model selection
            model = LogisticRegression()
            # Search Variables definition
            space = [
                {'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']},
                {'penalty': ['none', 'l2']},
                {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
            ]
        elif modelML == 'Support Vector Machine':
            # Model selection
            model = SVC(random_state=1)
            # Search Variables definition
            space = [
                {'kernel': ['rbf', 'linear', 'poly', 'sigmoid']},
                {'gamma': ['scale', 'auto']},
                {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
            ]
        elif modelML == 'Decision Tree':
            # Model selection
            model = DecisionTreeClassifier(random_state=1)
            # Search Variables definition
            space = [
                {'criterion': ['gini', 'entropy', 'log_loss']},
                {'splitter': ['best', 'random']},
                {'max_leaf_nodes': list(range(2, 100))},
                {'min_samples_split': [2, 3, 4]}
            ]

        # Search definition
        search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)

        # Search
        result = search.fit(X_train, y_train)

        # Save the best model
        best_model = result.best_estimator_

        # Model Evaluation
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        if modelML == 'Multiple Logistic Regression' or 'Support Vector Machine':
            # Confusion Matrix
            cnf_matrix = confusion_matrix(y_test, y_pred)
            labels = ['Alternative', 'Pop', 'Techno', 'Dance', 'Rock', 'Classical']
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            tick_marks = np.arange(len(labels))
            ax1.set_xticks(tick_marks, labels)
            ax1.set_yticks(tick_marks, labels)
            # Heatmap
            sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu", fmt='g', ax=ax1)
            ax1.xaxis.set_label_position("top")
            ax1.title.set_text('Confusion Matrix')

            RocCurveDisplay.from_estimator(search, X_test, y_test, ax=ax2)
        elif modelML == 'Decision Tree':
            fig = plt.figure(figsize=(15, 20))
            _ = tree.plot_tree(best_model,
                               feature_names=['Tempo', 'Beats per song', 'Danceability', 'Loudness (dB)', 'Energy (dB)',
                                              'Spectral Rolloff', 'Spectral Centroid'],
                               class_names=['Alternative', 'Pop', 'Techno', 'Dance', 'Rock', 'Classical'],
                               filled=True)

        # Store result for the outer one
        outer_results.append({'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1_score': f1})

        plt.show()

        # Report progress
        report = classification_report(y_test, y_pred)
        print(f'>acc={acc:.3f}, est={result.best_score_:.3f}, cfg={result.best_params_}')
        print(f'>precision={precision:.3f}, rll={recall:.3f}, f1={f1:.3f}')

        return
