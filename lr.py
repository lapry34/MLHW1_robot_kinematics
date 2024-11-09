import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import sys

robot = 'r5' # r3 or r5


validation_path = 'dataset/validation/' + robot + '.csv'
dataset_path = 'dataset/data/dataset_' + robot + '.csv'

target_values = list()
trigonometric_values = list()

if robot == 'r2':
    target_values = ['ee_x', 'ee_y', 'ee_qw', 'ee_qz']
    trigonometric_values = ['cos(j0)', 'sin(j0)', 'cos(j1)', 'sin(j1)']
elif robot == 'r3':
    target_values = ['ee_x', 'ee_y', 'ee_qw', 'ee_qz']
    trigonometric_values = ['cos(j0)', 'sin(j0)', 'cos(j1)', 'sin(j1)', 'cos(j2)', 'sin(j2)']
elif robot == 'r5':
    target_values = ['ee_x', 'ee_y', 'ee_z', 'ee_qw', 'ee_qx', 'ee_qy', 'ee_qz']
    trigonometric_values = ['cos(j0)', 'sin(j0)', 'cos(j1)', 'sin(j1)', 'cos(j2)', 'sin(j2)', 'cos(j3)', 'sin(j3)', 'cos(j4)', 'sin(j4)']


if __name__ == "__main__":

    # Load the data with specified delimiter and strip whitespace from column names
    data = pd.read_csv(dataset_path, delimiter=';')
    data.columns = data.columns.str.strip()  # Remove leading/trailing whitespace from column names

    # Split the data into X and Y
    X = data.drop(columns=target_values)    
    Y = data[target_values]

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1979543)

    validation_data = pd.read_csv(validation_path, delimiter=';')
    validation_data.columns = validation_data.columns.str.strip()  # Remove leading/trailing whitespace from column names

    X_val= validation_data.drop(columns=target_values)
    Y_val = validation_data[target_values]


    #Define the parameter grid
    param_grid = {
        'estimator__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'estimator__max_iter': [1000, 2000, 3000],
        'estimator__tol': [1e-3, 1e-4, 1e-5],
        'estimator__penalty': ['l1', 'l2', 'elasticnet'],
        'estimator__learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
        'estimator__eta0': [0.01, 0.1, 1.0],
        'estimator__power_t': [0.25, 0.5, 0.75]
    }

    #{'estimator__alpha': 0.0001, 'estimator__eta0': 0.1, 'estimator__learning_rate': 'adaptive', 'estimator__max_iter': 1000, 'estimator__penalty': 'elasticnet', 'estimator__power_t': 0.25, 'estimator__tol': 0.001}

    # Create a Linear Regression model
    base_model = linear_model.SGDRegressor(
        alpha=0.0001,
        max_iter=1000,
        tol=1e-3,
        penalty='elasticnet',
        learning_rate='adaptive',
        eta0=0.1,
        power_t=0.25
    )

    # Create a MultiOutput Regressor
    model = MultiOutputRegressor(base_model)

    # Create a GridSearchCV object
    #model = GridSearchCV(model, param_grid, cv=5, verbose=2, n_jobs=-1)
    
    #fit the model
    model.fit(X_train, Y_train)

    #print("Best parameters found: ", model.best_params_)

    # Predict
    Y_pred = model.predict(X_test)
    Y_pred = np.round(Y_pred, 3)

    # Evaluate the model on test set
    RMSE = root_mean_squared_error(Y_test, Y_pred)  # Calculate root mean squared error
    MAE = mean_absolute_error(Y_test, Y_pred)

    print("TEST:")
    print('Root Mean Squared Error:', RMSE)
    print('Mean Absolute Error:', MAE)

    # Predict on validation set
    Y_pred = model.predict(X_val)
    Y_pred = np.round(Y_pred, 3)

    # Evaluate the model on validation set
    RMSE = root_mean_squared_error(Y_val, Y_pred)  # Calculate root mean squared error
    MAE = mean_absolute_error(Y_val, Y_pred)

    print("VALIDATION:")
    print('Root Mean Squared Error:', RMSE)
    print('Mean Absolute Error:', MAE)


    sys.exit(0)