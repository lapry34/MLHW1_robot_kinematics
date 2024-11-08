import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import sys

#force tensorflow to use CPU
tf.config.set_visible_devices([], 'GPU')

robot = 'r5' # r3 or r5

# Paths
model_path = 'model_' + robot + '.keras'
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
    #X = X.drop(columns=trigonometric_values)  # Remove the cos/sin columns
    Y = data[target_values]


    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1979543)

    regularization_factor = 10e-6 #L2 regularization factor

    # Create the model
    model = keras.Sequential([
        keras.layers.Dense(64, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=keras.regularizers.l2(regularization_factor)),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(regularization_factor)),
        #keras.layers.Dropout(0.2),
        keras.layers.Dense(len(target_values))
    ])

    # print summary
    print(model.summary())

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, Y_train, epochs=15)

    # Evaluate the model
    test_loss = model.evaluate(X_test, Y_test)

    #predict
    Y_pred = model.predict(X_test)
    #round Y_pred to 3 decimal values
    Y_pred = np.round(Y_pred, 3)

    # Save the model
    model.save(model_path)

    print("TEST:")

    # Print the test loss
    print('Mean Squared Error:', test_loss) #mse impreciso valori < 1???
    print('Root Mean Squared Error:', np.sqrt(test_loss)) 

    # Print the MAE
    print('Mean Absolute Error:', np.mean(np.abs(Y_test - Y_pred)))

    sys.exit(0)
