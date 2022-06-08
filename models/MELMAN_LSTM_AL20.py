from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Bidirectional


# Note: 'weights' is ignored and just present for compatibility with other networks
def MELMAN_LSTM_AL20(classes, input_shape, weights=None):
    model = Sequential()
    model.add(LSTM(100, input_shape=input_shape))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
