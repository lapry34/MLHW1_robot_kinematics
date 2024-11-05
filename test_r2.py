import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import sys

if __name__ == "__main__":
    
    model = keras.models.load_model('model_r2.keras')
    print(model.summary())

    data = pd.read_csv('dataset/validation/r2.csv', delimiter=';')
    data.columns = data.columns.str.strip()  # Remove leading/trailing whitespace from column names

    target_values = ['ee_x', 'ee_y', 'ee_qw', 'ee_qz']

    # Split the data into X and Y
    X = data.drop(columns=target_values)
    Y = data[target_values]

    # evaluate model performance

    Y_pred = model.predict(X)
    Y_pred = np.round(Y_pred, 3)

    # Print the MAE
    print('Mean Absolute Error:', mean_absolute_error(Y, Y_pred))

    # Print the RMSE
    print('Root Mean Squared Error:', root_mean_squared_error(Y, Y_pred))

    sys.exit(0)