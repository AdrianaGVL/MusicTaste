##############################
#    Author:: Adriana Galán
#   Music Taste Project
############################

# Libraries
import pandas as pd
from numpy import mean
from numpy import std
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns

# Class variables for models evaluation
m_mse = list()
m_r2 = list()


def load_data(data_path, train_test=False):
    # CSVs reading and training - test data generation
    data = pd.read_csv(data_path, sep=';', decimal=",", index_col=None)
    y = data.iloc[:, -1]
    X = data.iloc[:, 6:-1]
    if train_test:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    else:
        return X, y, data


# Ridge Regression
def ridge_regress(data_path):
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
    X, y, data = load_data(data_path)
    # Enumerate splits
    outer_results = list()
    outer_r2 = list()
    # Outer loop
    for train, test in cv_outer.split(data):
        # Split data
        X_train, X_test = X.iloc[train, 7:-1], X.iloc[test, 7:-1]
        y_train, y_test = y.iloc[train], y.iloc[test]

        # Model selection
        model = RidgeCV(alphas=[0.1, 0.3, 0.5, 0.7, 1])

        # Search
        result = model.fit(X_train, y_train)

        # Model Evaluation
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Store result
        outer_results.append(mse)
        outer_r2.append(r2)

        # Plot results
        plt.clf()
        plt.scatter(y_test, y_pred, color='red')
        plt.xlabel('Real Values')
        plt.ylabel('Predicted Values')
        plt.title('Scatter Matrix - Ridge Regression')
        plt.savefig(f'ML_models/new_data/Ridge/Alpha_{r2}.png')

        # # Report progress
        print('- mse=%.3f,r2=%.3f, alpha=%.3f,cfg=%s' % (mse, r2, model.alpha_, [model.coef_, model.intercept_]))

    # Summarize the estimated performance of the model
    print('MSE: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))
    print('R2: %.3f (%.3f)' % (mean(outer_r2), std(outer_r2)))
    m_mse.append(mean(outer_results))
    m_r2.append(mean(outer_r2))

    return m_mse, m_r2


# KNN & Decision Tree Regression
def knn_regress(data_path, modelML):
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
    X, y, data = load_data(data_path)
    # Enumerate splits
    outer_results = list()
    outer_r2 = list()
    # Outer loop
    for train, test in cv_outer.split(X):
        # Split data
        X_train, X_test = X.iloc[train, 7:-1], X.iloc[test, 7:-1]
        y_train, y_test = y.iloc[train], y.iloc[test]

        # Configure the cross-validation procedure
        cv_inner = KFold(n_splits=3, shuffle=True, random_state=1)

        # Model selection and search space definition
        model = ''
        space = dict()
        if modelML == 'KNN':
            # Model selection
            model = KNeighborsRegressor()
            # Search Variables definition
            space['n_neighbors'] = list(range(2, 100))
        elif modelML == 'Decision Tree':
            # Model selection
            model = DecisionTreeRegressor()
            # Search Variables definition
            space['max_leaf_nodes'] = list(range(2, 100))
            space['min_samples_split'] = list(range(2, 20))

        # Search definition
        search = GridSearchCV(model, space, cv=cv_inner, refit=True)

        # Search
        result = search.fit(X_train, y_train)

        # Save the best model
        best_model = result.best_estimator_

        # Model Evaluation
        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Store result for the outer one
        outer_results.append(mse)
        outer_r2.append(r2)

        # Plot results
        if modelML == 'KNN':
            plt.clf()
            plt.scatter(y_test, y_pred, color='red')
            plt.savefig(f'ML_models/new_data/KNN/{model.n_neighbors}_Neighbours.png')
        elif modelML == 'Decision Tree':
            fig = plt.figure(figsize=(25, 20))
            _ = tree.plot_tree(best_model,
                               feature_names=data.columns,
                               filled=True)

        # Report progress
        print('- mse=%.3f,r2=%.3f, K_neightbours=%s' % (mse, r2, result.best_params_))

    # Summarize the estimated performance of the model
    print('MSE: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))
    print('R2: %.3f (%.3f)' % (mean(outer_r2), std(outer_r2)))
    m_mse.append(mean(outer_results))
    m_r2.append(mean(outer_r2))

    return m_mse, m_r2


def models_results(mses, mr2s):
    compare = pd.DataFrame(columns=['Model', 'MSE', 'R2'])
    compare['Model'] = ['Ridge', 'KNN', 'Decision Tree']
    compare['MSE'] = mses
    compare['R2'] = mr2s

    print(compare)
    sns.scatterplot(data=compare, x="MSE", y="R2", hue='Model', style="Model", s=150).set_title(
        'Regression models compared')

    return