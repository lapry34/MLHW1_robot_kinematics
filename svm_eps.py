import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from MAPE import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from joblib import dump, load

robot = 'r5'  # r3 or r5

# Paths
model_path_template = 'models/svm/eps/svm_' + robot + '_epsilon_{}.joblib'
validation_path = 'dataset/validation/' + robot + '.csv'
dataset_path = 'dataset/data/dataset_' + robot + '.csv'

target_values = []
if robot == 'r2':
    target_values = ['ee_x', 'ee_y', 'ee_qw', 'ee_qz']
    trigonometric_values = ['cos(j0)', 'sin(j0)', 'cos(j1)', 'sin(j1)']
elif robot == 'r3':
    target_values = ['ee_x', 'ee_y', 'ee_qw', 'ee_qz']
    trigonometric_values = ['cos(j0)', 'sin(j0)', 'cos(j1)', 'sin(j1)', 'cos(j2)', 'sin(j2)']
elif robot == 'r5':
    target_values = ['ee_x', 'ee_y', 'ee_z', 'ee_qw', 'ee_qx', 'ee_qy', 'ee_qz']
    trigonometric_values = ['cos(j0)', 'sin(j0)', 'cos(j1)', 'sin(j1)', 'cos(j2)', 'sin(j2)', 'cos(j3)', 'sin(j3)', 'cos(j4)', 'sin(j4)']

# Load dataset
data = pd.read_csv(dataset_path, delimiter=';')
data.columns = data.columns.str.strip()

# Split data
X = data.drop(columns=target_values)
Y = data[target_values]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1979543)

validation_data = pd.read_csv(validation_path, delimiter=';')
validation_data.columns = validation_data.columns.str.strip()
X_val = validation_data.drop(columns=target_values)
Y_val = validation_data[target_values]

# Epsilon values to try
epsilon_values = [0.08, 0.06, 0.05, 0.04, 0.03, 0.025, 0.02]
results = []

# Train and evaluate models for different epsilon values
for epsilon in epsilon_values:
    model_path = model_path_template.format(epsilon)

    # Check if model already exists
    if os.path.exists(model_path):
        print(f"Model for epsilon={epsilon} found. Loading...")
        model = load(model_path)
    else:
        print(f"Model for epsilon={epsilon} not found. Training...")
        svr = SVR(kernel='rbf', C=5, gamma='scale', epsilon=epsilon)
        model = MultiOutputRegressor(svr)
        model.fit(X_train, Y_train)
        dump(model, model_path)

    # Evaluate on validation set
    Y_pred = model.predict(X_val)
    RMSE = np.sqrt(mean_squared_error(Y_val, Y_pred))
    MAE = mean_absolute_error(Y_val, Y_pred)
    MAPE = mean_absolute_percentage_error(Y_val, Y_pred)

    # Count total support vectors
    total_sv = sum([estimator.n_support_ for estimator in model.estimators_])

    # Store results
    results.append({'epsilon': epsilon, 'support_vectors': total_sv, 'MAPE': MAPE})

    print(f"Epsilon: {epsilon}")
    print(f"Root Mean Squared Error: {RMSE}")
    print(f"Mean Absolute Error: {MAE}")
    print(f"Mean Absolute Percentage Error: {MAPE:.3f}%")
    print(f"Total Support Vectors: {total_sv}")
    print("-" * 40)

# Sort results by epsilon in descending order
results_df = pd.DataFrame(results).sort_values(by='epsilon', ascending=False)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(results_df['epsilon'], results_df['support_vectors'], marker='o', label='Support Vectors')

# Add labels for MAPE over each point
for i in range(len(results_df)):
    epsilon = results_df.iloc[i]['epsilon']
    sv = results_df.iloc[i]['support_vectors']
    mape = results_df.iloc[i]['MAPE']
    plt.text(epsilon, sv + 23, f"{mape:.2f}%", ha='center', fontsize=10)

plt.xlabel('Epsilon')
plt.ylabel('Number of Support Vectors')
plt.title('Epsilon vs Support Vectors (with MAPE) - ' + robot.upper())
plt.grid()
plt.legend()

# Save the plot
os.makedirs('plots', exist_ok=True)
plt.savefig(f'plots/svm_{robot}_epsilon_vs_sv.png')
plt.show()