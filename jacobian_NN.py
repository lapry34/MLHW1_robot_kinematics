import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import sys

#force tensorflow to use CPU
tf.config.set_visible_devices([], 'GPU')

robot = 'r3' # r3 or r5
tuned = False

# Paths
model_path = ''

#choose the model path
if tuned:
    model_path = 'models/NN/model_' + robot + '_tuned.keras'
else:
    model_path = 'models/NN/model_' + robot + '.keras'

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

input_dim = None
output_dim = None

def FK(model, theta, i_dim=None, o_dim=None):

    #get the input and output dimensions
    if i_dim is None or o_dim is None:
        i_dim = input_dim
        o_dim = output_dim
        
    #reshape to batch size 1
    t = tf.reshape(theta, (1, i_dim)) 
    #predict
    out = model(t)
    #reshape to 1D vector
    out = tf.reshape(out, (o_dim,)) 
    return out

@tf.function
def jacobian(model, x, i_dim=None, o_dim=None):

    #get the input and output dimensions
    if i_dim is None or o_dim is None:
        i_dim = input_dim
        o_dim = output_dim

    with tf.GradientTape() as tape:
        tape.watch(x)
        y = FK(model, x, i_dim, o_dim)
    return tape.jacobian(y, x)

def reduce_J(J, robot=robot):
    if robot == 'r2':
        return J[:2, :2]
    elif robot == 'r3':
        return J[:2, :3]
    elif robot == 'r5':
        return J[:3, :5]


#analytical jacobian of the robots
def analytical_jacobian(X, robot=robot):
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

        J = np.zeros(shape=(3, 5))

        #TODO 
        return J


def print_all(X, Y, X_sol, Y_pred):
    np.set_printoptions(precision=3, suppress=True)
    print("X: ", np.round(X, 3))
    print("X_sol: ", np.round(X_sol, 3))
    print("Y: ", np.round(Y, 3))
    print("Y_pred: ", np.round(Y_pred, 3))
    print("Error: ", np.round(np.linalg.norm(Y - Y_pred), 3))
    print("JACOBIANS:")
    J_analytical_sol = analytical_jacobian(X_sol)
    J_num_sol = reduce_J(jacobian(model, tf.constant(X_sol, dtype=tf.float32)).numpy())
    print("J_analytical in X_sol: ", np.round(J_analytical_sol, 3))
    print("J_num in X_sol: ", np.round(J_num_sol, 3))
    print("J_analytical in X: ", np.round(analytical_jacobian(X), 3))
    print("J_num in X: ", np.round(reduce_J(jacobian(model, tf.constant(X, dtype=tf.float32)).numpy()), 3))
    print("Error in J(X_sol): ", np.round(np.linalg.norm(J_analytical_sol - J_num_sol), 3))
    return

def IK(model, X_i, Y, max_iter=10000, eta=0.05, method='newton', lambda_=0.005, orientation=True, verbose=False): #lambda used if method is levenberg

    X_i = np.float32(X_i)
    Y = np.float32(Y)

    #row vectors
    i_dim = X_i.shape[0]
    o_dim = Y.shape[0]

    for i in range(max_iter):

        #convert X_i to tensor
        X_i_tf = tf.convert_to_tensor(X_i, dtype=tf.float32)

        # Predict the output for the current X_i
        Y_pred = FK(model, X_i_tf, i_dim, o_dim).numpy().flatten()

        # Compute the Jacobian of the model with respect to input X
        J = jacobian(model, X_i_tf, i_dim, o_dim).numpy()

        # Compute the difference between predicted and actual values
        error = Y_pred - Y  # Convert Y_pred to numpy for error calculation

        if orientation == False: #if we are not interested in the orientation
            n_joints = int(robot[1])
            #put the orientation error values to zero
            error[n_joints:] = 0

        if np.linalg.norm(error) < 10e-4:
            if verbose:
                print("Converged after ", i, " iterations.")
            return X_i

        if i % 100 == 0 and verbose:
            print("Iteration: ", i, " Error: ", np.linalg.norm(error))

        if method == 'newton':
            #check if the Jacobian has full rank
            if np.linalg.matrix_rank(J) < J.shape[0]:
                X_i = X_i - eta * (J.T @ error) # Gradient descent
            else:
                X_i = X_i -  eta * np.linalg.pinv(J) @ error # Newton's method
        
        elif method == 'levenberg':
            # Levenberg-Marquardt update
            J_levenberg = np.linalg.inv(J.T @ J + lambda_* np.eye(J.shape[1])) @ J.T
            X_i = X_i - eta * J_levenberg @ error

    #warning!
    print("Warning: IK Did not converge within the maximum number of iterations.")
    return X_i 

if __name__ == "__main__":
    # Load the trained model
    model = keras.models.load_model(model_path)

    #get the input and output dimensions
    input_dim = model.input_shape[1]
    output_dim = model.output_shape[1]

    # Load the data and prepare it as in the original script
    data = pd.read_csv(validation_path, delimiter=';')
    data.columns = data.columns.str.strip()  # Remove leading/trailing whitespace from column names

    # Split the data into X and Y
    X_val = data.drop(columns=target_values)
    Y_val = data[target_values]

    max_iter = 10000
    eta = 0.05

    # get a random data point from the validation set
    idx = np.random.randint(0, X_val.shape[0])
    X_gen = X_val.iloc[idx].values
    Y = Y_val.iloc[idx].values

    #initialize X_i randomly
    idx = np.random.randint(0, X_val.shape[0])
    X_i = X_val.iloc[idx].values

    # find iteratively X that provides the same Y
    print("Newton Method")
    X_sol = IK(model, X_i, Y, max_iter, eta, method='newton', verbose=True)
    Y_sol = model.predict(X_sol.reshape(1, -1), verbose=0).flatten()
    print_all(X_gen, Y, X_i, Y_sol)
    print("---------------------------------")
    print("Levenberg-Marquardt Method")
    X_sol = IK(model, X_i, Y, max_iter, eta, method='levenberg', verbose=True)
    Y_sol = model.predict(X_sol.reshape(1, -1), verbose=0).flatten()
    print_all(X_gen, Y, X_i, Y_sol)
    
    sys.exit(0)
