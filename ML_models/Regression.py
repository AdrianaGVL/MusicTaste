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


def load_data(data_path, train_test=False):
    # CSVs reading and training - test data generation
    data = pd.read_csv(data_path, sep=';', decimal=",", index_col=None)
    y = data.iloc[:, 6:7]
    X = data.iloc[:, 7:28]
    if train_test:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    else:
        return X, y, data


# Ridge Regression
def ridge_regress(data_path, plots=False):
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
    X, y, data = load_data(data_path)
    # Enumerate splits
    outer_results = list()
    # Outer loop
    for train, test in cv_outer.split(data.iloc[:, 6:28]):
        # Split data
        X_train, X_test = X.iloc[train, 0:28], X.iloc[test, 0:28]
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
        outer_results.append({'Model': 'Ridge', 'MSE': mse, 'R2': r2})

        # Plot results
        if plots:
            plt.clf()
            plt.plot([min(np.array(y_test)), max(np.array(y_test))], [min(y_pred), max(y_pred)], linestyle='--',
                     color='black',
                     label='Perfect Prediction')
            plt.scatter(y_test, y_pred, color='purple')
            plt.xlabel('Real Values')
            plt.ylabel('Predicted Values')
            plt.title('Scatter Matrix - Ridge Regression')
            plt.savefig(f'ML_models/new_data/Ridge/Alpha_{r2}.png')

        # # Report progress
        print('- mse=%.3f,r2=%.3f, alpha=%.3f' % (mse, r2, model.alpha_,))

    # Summarize the estimated performance of the model
    mse_values = [item['MSE'] for item in outer_results]
    r2_values = [item['R2'] for item in outer_results]
    print('MSE: %.3f (%.3f)' % (mean(mse_values), std(mse_values)))
    print('R2: %.3f (%.3f)' % (mean(r2_values), std(r2_values)))

    return outer_results


# KNN & Decision Tree Regression
def cross_validation(data_path, modelML, plots=False):
    num_knn = 0
    cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
    X, y, data = load_data(data_path)
    # Enumerate splits
    outer_results = list()
    # Outer loop
    for train, test in cv_outer.split(data.iloc[:, 6:28]):
        # Split data
        X_train, X_test = X.iloc[train, 0:20].astype(float), X.iloc[test, 0:20].astype(float)
        a1 = X_train.dtypes
        a2 = X_test.dtypes
        y_train, y_test = y.iloc[train].astype(float), y.iloc[test].astype(float)
        a3 = y_train.dtypes
        a4 = y_test.dtypes

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
        elif modelML == 'Regression Tree':
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
        outer_results.append({'Model': modelML, 'MSE': mse, 'R2': r2})

        # Plot results
        if modelML == 'KNN':
            if plots:
                num_knn += 1
                plt.clf()
                plt.plot([min(np.array(y_test)), max(np.array(y_test))], [min(y_pred), max(y_pred)], linestyle='--', color='black',
                         label='Perfect Prediction')
                plt.scatter(y_test, y_pred, color='blue')
                plt.xlabel('Real Values')
                plt.ylabel('Predicted Values')
                plt.title('Scatter Matrix - KNN')
                plt.savefig(f'ML_models/new_data/KNN/{best_model}_{num_knn}.png')

            # Report progress
            print('- mse=%.3f,r2=%.3f, K_neightbours=%s' % (mse, r2, result.best_params_))

        elif modelML == 'Regression Tree':
            if plots:
                fig = plt.figure(figsize=(25, 20))
                _ = tree.plot_tree(best_model,
                                   feature_names=X.iloc[:, 0:20].columns,
                                   filled=True)
                plt.savefig(f'ML_models/new_data/RegressTree/{best_model}.png')

            # Report progress
            print('- mse=%.3f,r2=%.3f, Tree_Params=%s' % (mse, r2, result.best_params_))

    # Summarize the estimated performance of the model
    mse_values = [item['MSE'] for item in outer_results]
    r2_values = [item['R2'] for item in outer_results]
    print('MSE: %.3f (%.3f)' % (mean(mse_values), std(mse_values)))
    print('R2: %.3f (%.3f)' % (mean(r2_values), std(r2_values)))

    return outer_results