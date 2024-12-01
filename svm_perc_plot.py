import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from joblib import dump, load
from MAPE import mean_absolute_percentage_error
from jacobian_svm import compute_jacobian, reduce_J, analytical_jacobian
from sklearn.metrics import mean_absolute_error

robot = 'r2'  # Options: 'r2', 'r3', 'r5'

# Paths
model_base_path = 'models/svm/perc/'
plot_path = 'plots/'
validation_path = f'dataset/validation/{robot}.csv'
dataset_path = f'dataset/data/dataset_{robot}.csv'

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
    validation_data.columns = data.columns.str.strip()

    X_val = validation_data.drop(columns=target_values)
    Y_val = validation_data[target_values]

    # Percentages of training data to use
    percentages = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

    # Lists to store results for plotting
    val_mape = []
    val_jacobian = []
    test_mape = []
    support_vectors = []

    for perc in percentages:
        # Determine the subset of training data
        subset_size = int(len(X_train) * perc)
        model_path = os.path.join(model_base_path, f'svm_{robot}_{int(perc * 100)}.joblib')

        if os.path.exists(model_path):
            print(f"Model for {int(perc * 100)}% already exists. Loading...")
            model = load(model_path)
        else:
            print(f"Model for {int(perc * 100)}% not found. Training...")
            X_train_subset = X_train.iloc[:subset_size]
            Y_train_subset = Y_train.iloc[:subset_size]

            # Define and train a new model
            svr = SVR(kernel='rbf', C=5, epsilon=0.03, gamma='scale')
            model = MultiOutputRegressor(svr)
            model.fit(X_train_subset, Y_train_subset)

            # Save the model for this percentage
            dump(model, model_path)

        # Predict on the validation set
        Y_pred_val = model.predict(X_val)


        J_num_list = []
        J_analytical_list = []

        # Compute the Jacobian of the model with respect to input X
        i = 0
        for x in X_val.values:
            J_num = compute_jacobian(model, x)
            J_analytical = analytical_jacobian(x, robot)
            J_num_reduced = reduce_J(J_num, robot)
            
            # Flatten and save the Jacobian data
            J_num_list.append(J_num_reduced.flatten())
            J_analytical_list.append(J_analytical.flatten())
            if i % 100 == 0:
                print(f"Jacobian {i} computed.")
            if i == 1000:
                break
            i += 1

        # Compute the difference between the numerical and analytical Jacobians
        MAE_jac = mean_absolute_error(J_analytical_list, J_num_list)
        print(f"Jacobian MAE: {MAE_jac:.4f}")
        val_jacobian.append(MAE_jac)

        # Evaluate the model on the validation set
        MAPE_val = mean_absolute_percentage_error(Y_val, Y_pred_val)
        val_mape.append(MAPE_val)

        # Predict on the test set
        Y_pred_test = model.predict(X_test)

        # Evaluate the model on the test set
        MAPE_test = mean_absolute_percentage_error(Y_test, Y_pred_test)
        test_mape.append(MAPE_test)

        # Calculate the total number of support vectors
        sv = sum(estimator.n_support_ for estimator in model.estimators_)
        support_vectors.append(sv)

        print(f"TRAINING SET {int(perc * 100)}%:")
        print(f"(Validation) MAPE: {MAPE_val:.4f}")
        print(f"(Test) MAPE: {MAPE_test:.4f}")
        print(f"Total Support Vectors: {sv}")

    # Plot MAPE with support vectors above points
    plt.figure(figsize=(10, 6))
    percentages_scaled = [p * 100 for p in percentages]
    plt.plot(percentages_scaled, test_mape, marker='o', label='Test MAPE', linestyle='-')
    plt.plot(percentages_scaled, val_mape, marker='o', label='Validation MAPE', linestyle='--')
    
    # Add the number of support vectors as labels above the validation points
    for i, perc in enumerate(percentages_scaled):
        plt.text(perc, val_mape[i] + 0.5, f"{support_vectors[i]}", ha='center', fontsize=10)

    plt.title(f'MAPE for SVM Model ({robot.upper()})')
    plt.xlabel('Percentage of Training Set Used (%)')
    plt.ylabel('Mean Absolute Percentage Error')
    plt.legend()
    plt.grid(True)

    # Save the plot
    mape_plot_path = os.path.join(plot_path, f'svm_{robot}_mape_with_sv.png')
    plt.savefig(mape_plot_path)
    print(f"MAPE plot saved to {mape_plot_path}")

    # Plot Jacobian error
    plt.figure(figsize=(10, 6))
    plt.plot([p * 100 for p in percentages], val_jacobian, marker='o', label='Validation Jacobian Error', linestyle='-')
    plt.title(f'Jacobian Error for Neural Network Model ({robot.upper()})')
    plt.xlabel('Percentage of Training Set Used (%)')
    plt.ylabel('Mean Absolute Error')
    plt.grid(True)

    # Plot Jacobian MAE
    jacobian_plot_path = os.path.join(plot_path, f'svm_{robot}_jacobian.png')
    plt.savefig(jacobian_plot_path)
    print(f"Jacobian plot saved to {jacobian_plot_path}")
    plt.close()