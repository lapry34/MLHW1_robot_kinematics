import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from MAPE import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from jacobian_NN import jacobian, reduce_J, analytical_jacobian

# Force TensorFlow to use CPU
tf.config.set_visible_devices([], 'GPU')

robot = 'r3'  # Options: 'r2', 'r3', 'r5'

tuned = True

n1, n2 = (None, None) 

if tuned == False:
    n1, n2 = (64, 64)
if robot == 'r2':       
    n1, n2 = (64, 24)
elif robot == 'r3':
    n1, n2 = (64, 40)
elif robot == 'r5':
    n1, n2 = (48, 64)

# Paths
model_base_path = 'models/NN/perc/'
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

    # Percentages of training data to use
    percentages = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

    # Lists to store results for plotting
    test_mape = []
    val_mape = []
    val_jacobian = []

    # Fixed model parameters
    regularization_factor = 1e-5  # L2 regularization factor
    model = keras.Sequential([
        keras.layers.Dense(n1, input_shape=(X_train.shape[1],), activation='relu', 
                           kernel_regularizer=keras.regularizers.l2(regularization_factor)),
        keras.layers.Dense(n2, activation='relu', kernel_regularizer=keras.regularizers.l2(regularization_factor)),
        keras.layers.Dense(len(target_values))
    ])

    # Get the number of trainable parameters
    num_params = model.count_params()

    for perc in percentages:
        # Determine the subset of training data
        subset_size = int(len(X_train) * perc)
        X_train_subset = X_train.iloc[:subset_size]
        Y_train_subset = Y_train.iloc[:subset_size]

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

        # Train the model
        model.fit(X_train_subset, Y_train_subset, epochs=15, verbose=0, validation_data=(X_test, Y_test))

        # Predict on the test set
        Y_pred_test = model.predict(X_test)

        # Evaluate the model on the test set
        MAPE_test = mean_absolute_percentage_error(Y_test, Y_pred_test)
        test_mape.append(MAPE_test)

        # Predict on the validation set
        Y_pred_val = model.predict(X_val)

        J_num_list = []
        J_analytical_list = []
        # Compute the Jacobian of the model with respect to input X
        for x in X_val.values:
            J_num = jacobian(model, x, x.shape[0], len(target_values)).numpy()
            J_analytical = analytical_jacobian(x, robot)
            J_num_reduced = reduce_J(J_num, robot)
            
            # Flatten and save the Jacobian data
            J_num_list.append(J_num_reduced.flatten())
            J_analytical_list.append(J_analytical.flatten())

        # Compute the difference between the numerical and analytical Jacobians
        MAE_jac = mean_absolute_error(J_analytical_list, J_num_list)
        print(f"Jacobian MAE: {MAE_jac:.4f}")
        val_jacobian.append(MAE_jac)

        # Evaluate the model on the validation set
        MAPE_val = mean_absolute_percentage_error(Y_val, Y_pred_val)
        val_mape.append(MAPE_val)

        print(f"TRAINING SET {int(perc * 100)}%:")
        print(f"(Test) MAPE: {MAPE_test:.4f}")
        print(f"(Validation) MAPE: {MAPE_val:.4f}")

        # Save the model for this percentage
        model_path = os.path.join(model_base_path, f'nn_{robot}_{int(perc * 100)}.keras')
        model.save(model_path)

    # Plot MAPE
    plt.figure(figsize=(10, 6))
    plt.plot([p * 100 for p in percentages], test_mape, marker='o', label='Test MAPE', linestyle='-')
    plt.plot([p * 100 for p in percentages], val_mape, marker='o', label='Validation MAPE', linestyle='--')
    plt.title(f'MAPE for Neural Network Model ({robot.upper()})')
    plt.xlabel('Percentage of Training Set Used (%)')
    plt.ylabel('Mean Absolute Percentage Error')
    plt.legend()
    plt.grid(True)

    # Add the number of parameters as a label
    plt.text(50, max(max(test_mape), max(val_mape)) * 0.9, f'Trainable Parameters: {num_params}', 
             fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

    # Save the MAPE plot
    mape_plot_path = os.path.join(plot_path, f'nn_{robot}_mape.png')
    plt.savefig(mape_plot_path)
    print(f"MAPE plot saved to {mape_plot_path}")
    plt.close()

    # Plot Jacobian error
    plt.figure(figsize=(10, 6))
    plt.plot([p * 100 for p in percentages], val_jacobian, marker='o', label='Validation Jacobian Error', linestyle='-')
    plt.title(f'Jacobian Error for Neural Network Model ({robot.upper()})')
    plt.xlabel('Percentage of Training Set Used (%)')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True)

    # Save the Jacobian plot
    jacobian_plot_path = os.path.join(plot_path, f'nn_{robot}_jacobian.png')
    plt.savefig(jacobian_plot_path)
    print(f"Jacobian plot saved to {jacobian_plot_path}")
    plt.close()