import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import sys

if __name__ == "__main__":

    # Load the data with specified delimiter and strip whitespace from column names
    data = pd.read_csv('dataset/data/dataset_r3.csv', delimiter=';')
    data.columns = data.columns.str.strip()  # Remove leading/trailing whitespace from column names

    target_values = ['ee_x', 'ee_y', 'ee_qw', 'ee_qz']

    # Split the data into X and Y
    X = data.drop(columns=target_values)    
    Y = data[target_values]

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1979543)

    validation_data = pd.read_csv('dataset/validation/r3.csv', delimiter=';')
    validation_data.columns = validation_data.columns.str.strip()  # Remove leading/trailing whitespace from column names

    X_validation = validation_data.drop(columns=target_values)
    Y_validation = validation_data[target_values]

    '''
    # try to find the best hyperparameters but with fixed epsilon=0.03 and kernel=rbf
    svr = SVR(kernel='rbf', epsilon=0.03)
    multi_target_svr = MultiOutputRegressor(svr)
    
    # define the grid search parameters
    param_grid = {
        'estimator__C': [0.1, 0.25, 0.5, 1, 5],
        'estimator__gamma': ['scale', 'auto'],
    }

    
    # define the random search
    clf = RandomizedSearchCV(multi_target_svr, param_distributions=param_grid, n_iter=10, cv=3, n_jobs=-1, verbose=3)
    clf.fit(X_train, Y_train)
    
    print("Best parameters found: ", clf.best_params_)
    '''
    
    svr = SVR(kernel='rbf', C=5, epsilon=0.03, gamma='scale')
    clf = MultiOutputRegressor(svr)
    clf.fit(X_train, Y_train)
    
    # Predict
    Y_pred = clf.predict(X_test)
    Y_pred = np.round(Y_pred, 3)

    # Evaluate the model on test set
    RMSE = root_mean_squared_error(Y_test, Y_pred)  # Calculate root mean squared error
    MAE = mean_absolute_error(Y_test, Y_pred)

    print("TEST:")
    print('Root Mean Squared Error:', RMSE)
    print('Mean Absolute Error:', MAE)

    # Predict on validation set
    Y_pred = clf.predict(X_validation)
    Y_pred = np.round(Y_pred, 3)

    # Evaluate the model on validation set
    RMSE = root_mean_squared_error(Y_validation, Y_pred)  # Calculate root mean squared error
    MAE = mean_absolute_error(Y_validation, Y_pred)

    print("VALIDATION:")
    print('Root Mean Squared Error:', RMSE)
    print('Mean Absolute Error:', MAE)

    sys.exit(0)
