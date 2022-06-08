from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense


# Note: 'weights' is ignored and just present for compatibility with other networks
def MELMAN_LCS_20_2(classes, input_shape, weights=None):
    # Defining the architecture of the NN and compiling the NN
    classifier = Sequential()  # The Sequential model is a linear stack of layers.
    # First Hidden Layer
    classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=input_shape))
    # Second  Hidden Layer
    classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal'))
    # Output Layer
    classifier.add(Dense(classes, activation='sigmoid', kernel_initializer='random_normal'))
    # Compiling the neural network
    classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return classifier

