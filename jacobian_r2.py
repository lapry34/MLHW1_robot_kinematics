import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import sys


def FK(model, theta):
    #reshape to batch size 1
    t = tf.reshape(theta, (1, 6)) #6 is the number of input variables
    out = model(t)
    #reshape to 1D vector
    out = tf.reshape(out, (4,)) #4 is the number of output variables
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

    # Load the data and prepare it as in the original script
    data = pd.read_csv('dataset/data/dataset_r2.csv', delimiter=';')
    data.columns = data.columns.str.strip()  # Remove leading/trailing whitespace from column names

    target_values = ['ee_x', 'ee_y', 'ee_qw', 'ee_qz']

    # Split the data into X and Y
    X = data.drop(columns=target_values)
    Y = data[target_values]

    # Compute the jacobian in a random point in the dataset
    X_sample = X.sample(1)

    J = jacobian(model, tf.constant(X_sample.values, dtype=tf.float32))

    print(J)

    #compute  the jacobian for the whole dataset one by one and print the percentage of completion

    jacobians = []
    for i, x in enumerate(X.values):
        J = jacobian(model, tf.constant([x], dtype=tf.float32))
        jacobians.append(J)

        if i % 1000 == 0:
            print(f'{i}/{len(X)}')
              
    # Save the jacobians to a file
    np.save('jacobians_r2.npy', np.array(jacobians))

    sys.exit(0)
