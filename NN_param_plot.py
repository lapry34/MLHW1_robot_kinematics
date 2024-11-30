import os
import numpy as np
import pandas as pd
from MAPE import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Force TensorFlow to use CPU
tf.config.set_visible_devices([], 'GPU')

robot = 'r5'  # Options: 'r2', 'r3', 'r5'

# Paths
model_base_path = 'models/NN/param/'
plot_path = 'plots/'
dataset_path = f'dataset/data/dataset_{robot}.csv'
validation_path = f'dataset/validation/{robot}.csv'

# Ensure output directories exist
os.makedirs(model_base_path, exist_ok=True)
os.makedirs(plot_path, exist_ok=True)

# Target values for each robot type
if robot == 'r2':
    target_values = ['ee_x', 'ee_y', 'ee_qw', 'ee_qz']
elif robot == 'r3':
    target_values = ['ee_x', 'ee_y', 'ee_qw', 'ee_qz']
elif robot == 'r5':
    target_values = ['ee_x', 'ee_y', 'ee_z', 'ee_qw', 'ee_qx', 'ee_qy', 'ee_qz']

if __name__ == "__main__":
    # Load the dataset
    data = pd.read_csv(dataset_path, delimiter=';')
    data.columns = data.columns.str.strip()

    # Split the data into X (features) and Y (targets)
    X = data.drop(columns=target_values)
    Y = data[target_values]

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1979543)

    # Load the validation data
    validation_data = pd.read_csv(validation_path, delimiter=';')
    validation_data.columns = validation_data.columns.str.strip()

    X_val = validation_data.drop(columns=target_values)
    Y_val = validation_data[target_values]

    # List of neurons to test
    neurons_list = [10, 16, 20, 32, 40, 48, 56, 64]

    # Lists to store results for plotting
    num_parameters = []
    val_mape = []

    for neurons in neurons_list:
        # Create the model
        regularization_factor = 1e-5  # L2 regularization factor
        model = keras.Sequential([
            keras.layers.Dense(neurons, input_shape=(X_train.shape[1],), activation='relu',
                               kernel_regularizer=keras.regularizers.l2(regularization_factor)),
            keras.layers.Dense(neurons, activation='relu', kernel_regularizer=keras.regularizers.l2(regularization_factor)),
            keras.layers.Dense(len(target_values))
        ])

        # Get the number of trainable parameters
        num_params = model.count_params()
        num_parameters.append(num_params)

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

        # Train the model
        model.fit(X_train, Y_train, epochs=15, verbose=0, validation_data=(X_test, Y_test))

        # Predict on the validation set
        Y_pred_val = model.predict(X_val)

        # Evaluate the model on the validation set
        MAPE_val = mean_absolute_percentage_error(Y_val, Y_pred_val)
        val_mape.append(MAPE_val)

        print(f"NEURONS: {neurons}, PARAMETERS: {num_params}, VALIDATION MAPE: {MAPE_val:.4f}")

        # Save the model for this configuration
        model_path = os.path.join(model_base_path, f'nn_{robot}_{neurons}_neurons.keras')
        model.save(model_path)

    # Plot Parameters vs. Validation MAPE
    plt.figure(figsize=(10, 6))
    plt.plot(num_parameters, val_mape, marker='o', label='Validation MAPE', linestyle='-')
    plt.title(f'Validation MAPE vs Number of Parameters ({robot.upper()})')
    plt.xlabel('Number of Trainable Parameters')
    plt.ylabel('Mean Absolute Percentage Error')
    plt.grid(True)
    plt.legend()

    # Save the Parameters vs MAPE plot
    mape_plot_path = os.path.join(plot_path, f'nn_{robot}_params_vs_mape.png')
    plt.savefig(mape_plot_path)
    print(f"Parameters vs MAPE plot saved to {mape_plot_path}")
    plt.close()