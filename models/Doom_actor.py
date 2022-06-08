from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Convolution2D, Dense, BatchNormalization, Flatten, Input
from tensorflow.python.keras.optimizers import Adam


def Doom_actor(classes, input_shape, weights=None):
    inputs = Input(shape=input_shape)
    x = Convolution2D(32, kernel_size=3, strides=2, activation="relu", padding="same")(inputs)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = Convolution2D(64, kernel_size=3, strides=2, activation="relu", padding="same")(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = Convolution2D(128, kernel_size=3, strides=2, activation="relu", padding="same")(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = Convolution2D(256, kernel_size=3, strides=2, activation="relu", padding="same")(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = Convolution2D(512, kernel_size=3, strides=2, activation="relu", padding="same")(x)
    x = BatchNormalization(epsilon=1e-5)(x)
    x = Flatten()(x)
    outputs = Dense(classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(.001), loss='mse')
    model.summary()
    return model
