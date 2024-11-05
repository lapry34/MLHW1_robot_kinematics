import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import sys


#input and output dimensions
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
    model = keras.models.load_model('model_r2.keras')

    #get the input and output dimensions
    input_dim = model.input_shape[1]
    output_dim = model.output_shape[1]

    # Load the data and prepare it as in the original script
    data = pd.read_csv('dataset/data/dataset_r2.csv', delimiter=';')
    data.columns = data.columns.str.strip()  # Remove leading/trailing whitespace from column names

    target_values = ['ee_x', 'ee_y', 'ee_qw', 'ee_qz']

    # Split the data into X and Y
    X = data.drop(columns=target_values)
    Y = data[target_values]

    # Compute the jacobian in a random point in the dataset taking only the first two values
    X_sample = X.sample(1)

    J_nn = jacobian(model, tf.constant(X_sample.values, dtype=tf.float32))

    print(J_nn.numpy())

    '''
    #compute  the jacobian for the whole dataset one by one and print the percentage of completion

    # -l1s2 -l2s12             -l2s12
    # l1c1 l2c12                l2c12
    # 1 1

    #extract from X_sample the values of the joints

    q1, q2 = X_sample.values[0][0], X_sample.values[0][1]
    l1, l2 = 1, 1

    J_analytical = np.array([[-l1*np.sin(q1) - l2*np.sin(q1+q2), -l2*np.sin(q1+q2)], [l1*np.cos(q1) + l2*np.cos(q1+q2), l2*np.cos(q1+q2)], [1, 1]])


    print(J_analytical)
    '''
    '''
    jacobians = []
    for i, x in enumerate(X.values):
        J = jacobian(model, tf.constant([x], dtype=tf.float32))
        jacobians.append(J)

        if i % 1000 == 0:
            print(f'{i}/{len(X)}')
              
    '''


    sys.exit(0)
