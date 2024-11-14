import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
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
    model_path = 'model_' + robot + '_tuned.keras'
else:
    model_path = 'model_' + robot + '.keras'

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

def FK(model, theta):
    #reshape to batch size 1
    t = tf.reshape(theta, (1, input_dim)) 
    #predict
    out = model(t)
    #reshape to 1D vector
    out = tf.reshape(out, (output_dim,)) 
    return out

@tf.function
def jacobian(model, x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = FK(model, x)
    return tape.jacobian(y, x)

def reduce_J(J):
    if robot == 'r2':
        return J[:2, :2]
    elif robot == 'r3':
        return J[:2, :3]
    elif robot == 'r5':
        return J[:3, :5]

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
    for i in range(max_iter):

        # Compute the Jacobian of the model with respect to input X
        J = jacobian(model, tf.constant(X_i, dtype=tf.float32))
        
        #convert J to numpy
        J = J.numpy()

        # Predict the output for the current X_i
        X_i_tf = tf.constant(X_i.reshape(1, -1), dtype=tf.float32)
        Y_pred = model.predict(X_i_tf, verbose=0).flatten()

        # Compute the difference between predicted and actual values
        error = Y_pred - Y 

        if np.linalg.norm(error) < 10e-4:
            print("Converged after ", i, " iterations.")
            break

        if i % 100 == 0:
            print("Iteration: ", i, " Error: ", np.linalg.norm(error))

        #check if the Jacobian has full rank
        if np.linalg.matrix_rank(J) < J.shape[0]:
            X_i = X_i - eta * (J.T @ error) # Gradient descent
        else:
            X_i = X_i -  eta * np.linalg.pinv(J) @ error # Newton's method




    #compare the predicted and actual values
    print("X: ", X_val.iloc[0].values)
    print("X_i: ", X_i)

    X_gen_tf = tf.constant(X_gen.reshape(1, -1), dtype=tf.float32)
    X_i_tf = tf.constant(X_i.reshape(1, -1), dtype=tf.float32)

    np.set_printoptions(suppress=True)
    print("Predicted Y in X: ", model.predict(X_gen_tf, verbose=0).flatten())
    print("Predicted Y in X_i: ", model.predict(X_i_tf, verbose=0).flatten())
    print("Y: ", Y)

    print("Error: %.5f " % np.linalg.norm(model.predict(X_i_tf, verbose=0) - Y))

    J_analytical = analytical_jacobian(X_gen)
    print("Analytical Jacobian: ", J_analytical)
    J_num = jacobian(model, X_gen).numpy()
    J_num = reduce_J(J_num)

    print("Numerical Jacobian: ", J_num)

    #error of computation between the J(acobian
    J_err = mean_absolute_error(J_analytical, J_num)
    print("J Error: %.4f " % J_err)

    sys.exit(0)
