import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from MAPE import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import sys

#force tensorflow to use CPU
tf.config.set_visible_devices([], 'GPU')

robot = 'r3' # r3 or r5
tuned = False

# Paths
model_path = ''

#choose the model path
if tuned:
    model_path = 'models/NN/model_' + robot + '_tuned.keras'
else:
    model_path = 'models/NN/model_' + robot + '.keras'

# validation path for the robot
validation_path = 'dataset/validation/' + robot + '.csv'

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
    
    model = keras.models.load_model(model_path)
    print(model.summary())

    data = pd.read_csv(validation_path, delimiter=';')
    data.columns = data.columns.str.strip()  # Remove leading/trailing whitespace from column names


    # Split the data into X and Y
    X = data.drop(columns=target_values)
    #X = X.drop(columns=trigonometric_values)  # Remove the cos/sin columns
    Y = data[target_values]

    # evaluate model performance

    Y_pred = model.predict(X)
    Y_pred = np.round(Y_pred, 3)

    # Evaluate the model on test set
    RMSE = root_mean_squared_error(Y, Y_pred)  # Calculate mean squared error
    MAE = mean_absolute_error(Y, Y_pred)  # Calculate mean absolute error
    MAPE = mean_absolute_percentage_error(Y, Y_pred)  # Calculate mean absolute percentage error

    print("TEST:")

    # Print the RMSE
    print('Root Mean Squared Error:', RMSE)

    # Print the MAE
    print('Mean Absolute Error:', MAE)

    # Print the MAPE
    print('Mean Absolute Percentage Error: %.3f%%' % (MAPE))

    sys.exit(0)