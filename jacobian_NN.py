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
    t = tf.reshape(theta, (1, input_dim)) #6 is the number of joints
    out = model(t)
    #reshape to 1D vector
    out = tf.reshape(out, (output_dim,)) #4 is the number of end effector position variables
    return out

@tf.function
def jacobian(model, x):
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = FK(model, x)
    return tape.jacobian(y, x)


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
    X = data.drop(columns=target_values)
    Y = data[target_values]

    # Compute the jacobian in a random point in the dataset taking only the first two values
    X_sample = X.sample(1)

    J_nn = jacobian(model, tf.constant(X_sample.values, dtype=tf.float32))

    print(J_nn.numpy())



    sys.exit(0)
