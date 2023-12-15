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
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, RocCurveDisplay, \
    classification_report, auc, roc_curve, roc_auc_score
from sklearn.preprocessing import label_binarize
from itertools import cycle


# Data loading
def load_data(data_path, train_test=False, plots=False):
    # CSVs reading and training - test data generation
    data = pd.read_csv(data_path, sep=';', decimal=",", index_col=None)
    y = data.iloc[:, 0]
    X = data.iloc[:, 2:6].apply(pd.to_numeric, errors='coerce')

    # Represent data
    if plots:
        color_map = {'Alternative': 'red', 'Pop': 'blue', 'Classical': 'green', 'Rock': 'purple', 'Dance': 'orange',
                     'Techno': 'brown'}
        marker_map = {'Alternative': 'o', 'Pop': 's', 'Classical': '^', 'Rock': 'v', 'Dance': 'p', 'Techno': '*'}
        for col in X.columns:
            plt.figure(figsize=(10, 8))
            plt.clf()
            for label in y.unique():
                subset = X.loc[y == label, col].round(3)
                plt.scatter(subset, y[y == label], label=f'{label}',
                           color=color_map[label], marker=marker_map[label])
            plt.title(f'{col}')
            plt.xlabel('Values')
            plt.ylabel('Genres')
            plt.legend()
            plt.savefig(f'ML_models/new_data/{col}.png')

    if train_test:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    else:
        return X, y, data


# QDA model
def qda_model(data_path):
    results = list()
    X_train, X_test, y_train, y_test = load_data(data_path, train_test=True)

    # One-hot encode the categorical variable
    y_train_reshaped = y_train.values.reshape(-1, 1)
    y_test_reshaped = y_test.values.reshape(-1, 1)
    encoder = OneHotEncoder()
    y_train_encoded = np.argmax(encoder.fit_transform(y_train_reshaped).toarray(), axis=1)
    y_test_encoded = np.argmax(encoder.transform(y_test_reshaped).toarray(), axis=1)

    # Create and train the QDA model
    model = QuadraticDiscriminantAnalysis()
    model.fit(X_train, y_train_encoded)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model and store the result
    acc = accuracy_score(y_test_encoded, y_pred)
    precision = precision_score(y_test_encoded, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test_encoded, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_test_encoded, y_pred, average='macro', zero_division=1)

    results.append({'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1_score': f1})
    print(f'>acc={acc:.3f}, precision={precision:.3f}, rll={recall:.3f}, f1={f1:.3f}')

    cm = confusion_matrix(y_test_encoded, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=y_train.unique(), yticklabels=y_train.unique())
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f"ML_models/new_data/QDA/Confusion_Matrix.png")

    return results


# Cross Validation for two Random Forest & Decision Tree
def cross_valid_models(data_path, modelML):
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=2)
    X, y, data = load_data(data_path)
    # Enumerate splits
    outer_results = list()
    for train, test in cv_outer.split(data.iloc[:, 0:6]):
        # Split data
        X_train, X_test = X.iloc[train, 1:6], X.iloc[test, 1:6]
        y_train, y_test = y.iloc[train], y.iloc[test]

        # Configure the cross-validation procedure
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)

        # Model selection and search space definition
        model = ''
        space = ''
        if modelML == 'Random Forest':
            # Model selection
            model = RandomForestClassifier(random_state=1)
            # Search Variables definition
            space = {
                'n_estimators': [50, 100, 150],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
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
        precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)

        # Store result for the outer one
        outer_results.append({'Accuracy': acc, 'Precision': precision, 'Recall': recall, 'F1_score': f1})

        if modelML == 'Random Forest':
            # Feature Importance
            feature_importance = best_model.feature_importances_
            feature_names = best_model.feature_names_in_

            # Plot Feature Importance
            plt.figure(figsize=(20, 12))
            ax = sns.barplot(x=feature_importance, y=feature_names, palette="viridis")
            plt.title('Feature Importance in Random Forest', fontweight='bold')
            plt.xlabel('Importance', fontweight='bold')
            plt.ylabel('Features', fontweight='bold')
            for i in range(len(feature_importance)):
                ax.text(feature_importance[i], i, f'{feature_names[i]}: {feature_importance[i]:.4f}', ha='left',
                        va='center')
            plt.tight_layout()
            sns.set_style("whitegrid")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.set_xlim(0, max(feature_importance) * 1.1)
            plt.savefig(f"ML_models/new_data/Forest/feature_importance.png")

            for tree_index, tree_estimator in enumerate(best_model.estimators_):
                plt.figure(figsize=(192, 108))
                plot_tree(tree_estimator, feature_names=best_model.feature_names_in_,
                          class_names=['Alternative', 'Pop', 'Techno', 'Dance', 'Rock', 'Classical'],
                          filled=True, rounded=True)
                plt.title(f"Decision Tree {tree_index}", fontweight='bold')
                plt.savefig(f"ML_models/new_data/Forest/{tree_index}.png")

        elif modelML == 'Decision Tree':
            fig = plt.figure(figsize=(192, 108))
            _ = tree.plot_tree(best_model,
                               feature_names=['Beats per song', 'Danceability', 'Loudness (dB)',
                                              'Spectral Rolloff', 'Spectral Centroid'],
                               class_names=['Alternative', 'Pop', 'Techno', 'Dance', 'Rock', 'Classical'],
                               filled=True)
            plt.savefig(f"ML_models/new_data/Tree/{best_model}.png")

        # Report progress
        report = classification_report(y_test, y_pred)
        print(f'>acc={acc:.3f}, est={result.best_score_:.3f}, cfg={result.best_params_}')
        print(f'>precision={precision:.3f}, rll={recall:.3f}, f1={f1:.3f}')

    return outer_results
