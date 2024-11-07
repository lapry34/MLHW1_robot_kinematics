import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
import sys

if __name__ == "__main__":

    # Load the data with specified delimiter and strip whitespace from column names
    data = pd.read_csv('dataset/data/dataset_r2.csv', delimiter=';')
    data.columns = data.columns.str.strip()  # Remove leading/trailing whitespace from column names

    target_values=['ee_x', 'ee_y', 'ee_qw', 'ee_qz']

    # Split the data into X and Y
    X = data.drop(columns=target_values)    
    #X = X.drop(columns=['j0','j1']) #remove the joint values
    X = X.drop(columns=['cos(j0)', 'sin(j0)', 'cos(j1)', 'sin(j1)']) #remove the cos/sin
    Y = data[target_values]

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1979543)

    regularization_factor = 10e-6 #L2 regularization factor

    # Create the model
    model = keras.Sequential([
        keras.layers.Dense(64, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=keras.regularizers.l2(regularization_factor)),
        keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(regularization_factor)),
        #keras.layers.Dense(16, activation='relu', kernel_regularizer=keras.regularizers.l2(regularization_factor)),
        #keras.layers.Dropout(0.2),
        keras.layers.Dense(len(target_values))
    ])

    # print summary
    print(model.summary())

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, Y_train, epochs=20)
    # Evaluate the model
    test_loss = model.evaluate(X_test, Y_test)

    #predict
    Y_pred = model.predict(X_test)
    #round Y_pred to 3 decimal values
    Y_pred = np.round(Y_pred, 3)

    # Save the model
    model.save('model_r3.keras')

    print("TEST:")

    # Print the test loss
    print('Mean Squared Error:', test_loss) #mse impreciso valori < 1???
    print('Root Mean Squared Error:', np.sqrt(test_loss)) 

    # Print the MAE
    print('Mean Absolute Error:', np.mean(np.abs(Y_test - Y_pred)))

    sys.exit(0)
