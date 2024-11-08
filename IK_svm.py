import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import sys
from joblib import load
import warnings

# Suppress specific UserWarning
warnings.filterwarnings("ignore", message="X does not have valid feature names")


robot = 'r3' # r3 or r5

# Paths
model_path = 'svm_' + robot + '.joblib'

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


def compute_jacobian(model, X, epsilon=1e-5):
    """
    Compute the Jacobian of a MultiOutputRegressor with respect to input X using finite differences.
    
    Parameters:
    - model: Trained MultiOutputRegressor.
    - X: Input data point (1D array) for which the Jacobian is computed.
    - epsilon: Small number to compute the finite difference.
    
    Returns:
    - jacobian: 2D numpy array with shape (n_outputs, n_features), representing
                the Jacobian matrix of partial derivatives.
    """
    n_outputs = len(model.estimators_)
    n_features = X.shape[0]
    jacobian = np.zeros((n_outputs, n_features))
    
    # For each feature, compute the partial derivative for each output using central differences
    for i in range(n_features):
        # Perturb the i-th feature by epsilon in both directions
        X_perturbed_plus = X.copy()
        X_perturbed_minus = X.copy()
        X_perturbed_plus[i] += epsilon
        X_perturbed_minus[i] -= epsilon
        
        # Predict with the perturbed inputs
        perturbed_output_plus = model.predict(X_perturbed_plus.reshape(1, -1)).flatten()
        perturbed_output_minus = model.predict(X_perturbed_minus.reshape(1, -1)).flatten()
        
        # Calculate partial derivatives (finite central differences)
        jacobian[:, i] = (perturbed_output_plus - perturbed_output_minus) / (2 * epsilon)
    
    return jacobian

if __name__ == "__main__":

    # Load the validation data
    val_data = pd.read_csv(validation_path, delimiter=';')
    val_data.columns = val_data.columns.str.strip()  # Remove leading/trailing whitespace from column names

    X_val = val_data.drop(columns=target_values)
    Y_val = val_data[target_values]

    X_val = X_val.to_numpy()
    Y_val = Y_val.to_numpy()

    # Load the trained model
    model = load(model_path)

    
    # get a random data point from the validation set
    idx = np.random.randint(0, X_val.shape[0])
    X = X_val[idx]
    Y = Y_val[idx]

    max_iter = 10000
    eta = 0.01

    # get the initial X
    X_i = X_val[0]
    
    # find iteratively X that provides the same Y
    for i in range(max_iter):

        # Compute the Jacobian of the model with respect to input X
        J = compute_jacobian(model, X_i)
        
        # Compute the difference between predicted and actual values
        Y_pred = model.predict(X_i.reshape(1, -1)).flatten()
        error = Y_pred - Y 

        if np.linalg.norm(error) < 10e-4:
            print("Converged after ", i, " iterations.")
            break


        if i % 100 == 0:
            print("Iteration: ", i, " Error: ", np.linalg.norm(error))

        #check if the Jacobian has full rank
        if np.linalg.matrix_rank(J) < J.shape[0]:
            X_i = X_i - eta * np.dot(J.T, error) # Gradient descent
        else:
            X_i = X_i -  eta * np.linalg.pinv(J) @ error # Newton's method




    #compare the predicted and actual values
    print("X: ", X)
    print("X_i: ", X_i)

    np.set_printoptions(suppress=True)
    print("Predicted Y in X: ", model.predict(X.reshape(1, -1)).flatten())
    print("Predicted Y in X_i: ", model.predict(X_i.reshape(1, -1)).flatten())
    print("Y: ", Y)

    print("Error: %.5f " % np.linalg.norm(model.predict(X_i.reshape(1, -1)).flatten() - Y))

    sys.exit(0)