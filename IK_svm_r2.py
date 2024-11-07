import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
import sys
from joblib import load

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

    # Compute output for the original input
    original_output = model.predict(X.reshape(1, -1)).flatten()
    
    # For each feature, compute the partial derivative for each output
    for i in range(n_features):
        # Perturb the i-th feature by epsilon
        X_perturbed = X.copy()
        X_perturbed[i] += epsilon
        
        # Predict with the perturbed input
        perturbed_output = model.predict(X_perturbed.reshape(1, -1)).flatten()
        
        # Calculate partial derivatives (finite differences)
        jacobian[:, i] = (perturbed_output - original_output) / epsilon
    
    return jacobian

if __name__ == "__main__":

    # Load the validation data
    val_data = pd.read_csv('dataset/validation/r2.csv', delimiter=';')
    val_data.columns = val_data.columns.str.strip()  # Remove leading/trailing whitespace from column names

    target_values = ['ee_x', 'ee_y', 'ee_qw', 'ee_qz']
    X_val = val_data.drop(columns=target_values)
    Y_val = val_data[target_values]

    X_val = X_val.to_numpy()
    Y_val = Y_val.to_numpy()

    # Load the trained model
    clf = load('svm_r2.joblib')

    
    # get a random data point from the validation set
    idx = np.random.randint(0, X_val.shape[0])
    X = X_val[idx]
    Y = Y_val[idx]

    max_iter = 10000
    learning_rate = 0.01

    # get the initial X
    X_0 = X_val[0]
    
    # find iteratively X that provides the same Y
    for i in range(max_iter):

        # Compute the Jacobian of the model with respect to input X
        J = compute_jacobian(clf, X_0)
        
        # Compute the difference between predicted and actual values
        Y_pred = clf.predict(X_0.reshape(1, -1)).flatten()
        error = Y_pred - Y 

        if np.linalg.norm(error) < 10e-4:
            print("Converged after ", i, " iterations.")
            break


        if i % 100 == 0:
            print("Iteration: ", i, " Error: ", np.linalg.norm(error))

        #check if the Jacobian is singular
        if np.linalg.matrix_rank(J) < J.shape[0]:
            X_0 = X_0 - learning_rate * np.dot(J.T, error) # Gradient descent
        else:
            X_0 = X_0 -  learning_rate*np.linalg.pinv(J) @ error # Newton's method




    #compare the predicted and actual values
    print("X: ", X)
    print("X_inv: ", X_0)

    np.set_printoptions(suppress=True)
    print("Predicted Y in X: ", clf.predict(X.reshape(1, -1)).flatten())
    print("Predicted Y in X_inv: ", clf.predict(X_0.reshape(1, -1)).flatten())
    print("Y: ", Y)

    print("Error: %.5f " % np.linalg.norm(clf.predict(X_0.reshape(1, -1)).flatten() - Y))

    sys.exit(0)