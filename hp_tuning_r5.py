import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras_tuner import Hyperband
import sys
import os

def build_model(hp):
    # Hyperparameters to tune
    hp_units_1 = hp.Int('units_1', min_value=8, max_value=64, step=8)
    hp_units_2 = hp.Int('units_2', min_value=8, max_value=64, step=8)
    hp_activation = hp.Choice('activation', values=['relu', 'sigmoid', 'tanh'])
    hp_learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')
    hp_regularization_factor = hp.Float('regularization_factor', min_value=1e-6, max_value=1e-2, sampling='log')

    # Create the model
    model = keras.Sequential([
        keras.layers.Dense(hp_units_1, input_shape=(X_train.shape[1],), activation=hp_activation,
                           kernel_regularizer=keras.regularizers.l2(hp_regularization_factor)),
        keras.layers.Dense(hp_units_2, activation=hp_activation,
                           kernel_regularizer=keras.regularizers.l2(hp_regularization_factor)),
        keras.layers.Dense(len(target_values))
    ])

    # Compile the model
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss='mean_squared_error')

    return model

if __name__ == "__main__":

    # Load the data with specified delimiter and strip whitespace from column names
    data = pd.read_csv('dataset/data/dataset_r5.csv', delimiter=';')
    data.columns = data.columns.str.strip()  # Remove leading/trailing whitespace from column names
    target_values=['ee_x', 'ee_y', 'ee_z', 'ee_qw', 'ee_qx', 'ee_qy', 'ee_qz']

    # Split the data into X and Y
    X = data.drop(columns=target_values)
    # X = X.drop(columns=['cos(j0)', 'sin(j0)', 'cos(j1)', 'sin(j1)', 'cos(j2)', 'sin(j2)'])  # Remove the cos/sin columns
    Y = data[target_values]

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1979543)

    # Function to build model for Keras Tuner

    # Remove previous tuner results if they exist
    if os.path.exists('hyperband_tuning/robotic_arm_regression'):
        import shutil
        shutil.rmtree('hyperband_tuning/robotic_arm_regression')

    # Instantiate the tuner
    # Using Hyperband for efficient hyperparameter search
    tuner = Hyperband(build_model,
                    objective='val_loss',
                    max_epochs=20,
                    factor=3,
                    directory='hyperband_tuning',
                    project_name='robotic_arm_regression')

    # Run the hyperparameter search
    tuner.search(X_train, Y_train, validation_split=0.2, epochs=20)

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=10)[0]

    print(f"The optimal number of units in the first layer is {best_hps.get('units_1')}")
    print(f"The optimal number of units in the second layer is {best_hps.get('units_2')}")
    print(f"The optimal activation function is {best_hps.get('activation')}")
    print(f"The optimal learning rate is {best_hps.get('learning_rate')}")
    print(f"The optimal regularization factor is {best_hps.get('regularization_factor')}")

    # Train the model with the optimal hyperparameters
    model = tuner.hypermodel.build(best_hps)
    model.fit(X_train, Y_train, validation_split=0.2, epochs=20)

    # Evaluate the model
    test_loss = model.evaluate(X_test, Y_test)

    # Predict
    y_pred = model.predict(X_test)
    # Round y_pred to 3 decimal values
    y_pred = np.round(y_pred, 3)

    # Save the model
    model.save('model_r5_tuned.keras')

    print("TEST:")

    # Print the test loss
    print('Mean Squared Error:', test_loss)
    print('Root Mean Squared Error:', np.sqrt(test_loss))

    # Print the MAE
    print('Mean Absolute Error:', np.mean(np.abs(Y_test - y_pred)))

    sys.exit(0)
