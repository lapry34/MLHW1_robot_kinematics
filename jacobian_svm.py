import numpy as np
import pandas as pd
from scipy import optimize
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

#analytical jacobian of the robots

def analytical_jacobian(X):
    if robot == 'r2':

        theta_1 = X[0]
        theta_2 = X[1]

        l1 = l2 = 0.1

        J = np.array([[-l1*np.sin(theta_1) - l2*np.sin(theta_1 + theta_2), -l2*np.sin(theta_1 + theta_2)],
                      [l1*np.cos(theta_1) + l2*np.cos(theta_1 + theta_2), l2*np.cos(theta_1 + theta_2)]])
        return J

    if robot == 'r3':

        theta_1 = X[0]
        theta_2 = X[1]
        theta_3 = X[2]

        l1 = l2 = l3 = 0.1

        J = np.array([
            [-l1*np.sin(theta_1) - l2*np.sin(theta_1 + theta_2) - l3*np.sin(theta_1 + theta_2 + theta_3), -l2*np.sin(theta_1 + theta_2) - l3*np.sin(theta_1 + theta_2 + theta_3), -l3*np.sin(theta_1 + theta_2 + theta_3)],
            [l1*np.cos(theta_1) + l2*np.cos(theta_1 + theta_2) + l3*np.cos(theta_1 + theta_2 + theta_3), l2*np.cos(theta_1 + theta_2) + l3*np.cos(theta_1 + theta_2 + theta_3), l3*np.cos(theta_1 + theta_2 + theta_3)]
        ])
        return J
    
    if robot == 'r5':

        theta = X[:5]
        l = 0.1

        J = np.zeros(3,5)

        #TODO 

        return J


def compute_jacobian(model, X, epsilon=1e-6):

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

def reduce_J(J):
    if robot == 'r2':
        return J[:2, :2]
    elif robot == 'r3':
        return J[:2, :3]
    elif robot == 'r5':
        return J[:3, :5]
    
def print_all(X, Y, X_sol, Y_pred):
    np.set_printoptions(precision=3, suppress=True)
    print("X: ", np.round(X, 3))
    print("X_sol: ", np.round(X_sol, 3))
    print("Y: ", np.round(Y, 3))
    print("Y_pred: ", np.round(Y_pred, 3))
    print("Error: ", np.round(np.linalg.norm(Y - Y_pred), 3))
    print("JACOBIANS:")
    J_analytical_sol = analytical_jacobian(X_sol)
    J_num_sol = reduce_J(compute_jacobian(model, X_sol))
    print("J_analytical in X_sol: ", np.round(J_analytical_sol, 3))
    print("J_num in X_sol: ", np.round(J_num_sol, 3))
    print("J_analytical in X: ", np.round(analytical_jacobian(X), 3))
    print("J_num in X: ", np.round(reduce_J(compute_jacobian(model, X)), 3))
    print("Error in J(X_sol): ", np.round(np.linalg.norm(J_analytical_sol - J_num_sol), 3))
    return

def IK(model, X_i, Y, max_iter=10000, eta=0.05, method='newton', lambda_=0.005): #lambda used if method is levenberg
    
    # Iterate to find X such that model(X) ≈ Y
    for i in range(max_iter):

        # Compute the Jacobian of the model with respect to input X
        J = compute_jacobian(model, X_i)

        # Compute the difference between predicted and actual values
        Y_pred = model.predict(X_i.reshape(1, -1)).flatten()
        error = Y_pred - Y 

        # Check convergence
        if np.linalg.norm(error) < 1e-4:
            print("Converged after", i, "iterations.")
            break

        # Log progress
        if i % 100 == 0:
            print("Iteration:", i, "Error:", np.linalg.norm(error))

        # Update rule for different methods
        if method == 'newton':
            # Check if the Jacobian has full rank
            if np.linalg.matrix_rank(J) < J.shape[0]:
                X_i = X_i - eta * (J.T @ error)  # Gradient descent fallback
            else:
                X_i = X_i - eta * np.linalg.pinv(J) @ error  # Newton's method

        elif method == 'levenberg':
            # Levenberg-Marquardt update
            J_levenberg = np.linalg.inv(J.T @ J + lambda_* np.eye(J.shape[1])) @ J.T
            X_i = X_i - eta * J_levenberg @ error

    return X_i

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

    # Print the number of support vectors
    sv = 0
    for estimator in model.estimators_:
        sv += estimator.n_support_
    print("Total support vectors: ", sum(sv))
    
    # get a random data point from the validation set
    idx = np.random.randint(0, X_val.shape[0])
    X = X_val[idx]
    Y = Y_val[idx]

    idx = np.random.randint(0, X_val.shape[0])
    X_i = X_val[idx]
   
    X_sol = IK(model, X_i, Y, method='newton')
    print("Using our method (Newton's method)")
    print_all(X, Y, X_sol, model.predict(X_sol.reshape(1, -1)).flatten())
    print("-----------------")
    print("Using our method (Levenberg-Marquardt)")
    X_sol = IK(model, X_i, Y, method='levenberg')
    print_all(X, Y, X_sol, model.predict(X_sol.reshape(1, -1)).flatten())
    print("-----------------")
    print("Using Scipy optimization BFGS")
    def error(X, model, Y):
        return np.linalg.norm(model.predict(X.reshape(1, -1)).flatten() - Y)
    
    res = optimize.minimize(error, X_i, args=(model, Y), method='BFGS', options={'disp': False})
    X_sol = res.x
    print_all(X, Y, X_sol, model.predict(X_sol.reshape(1, -1)).flatten())

    sys.exit(0)