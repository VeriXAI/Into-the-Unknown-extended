# https://www.kaggle.com/micajoumathematics/fine-tuning-efficientnetb0-on-cifar-100

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
import efficientnet.keras as efn
from tensorflow.python.keras.applications.vgg16 import VGG16


def CIFAR_EffNet(classes, input_shape, weights=None):
    efnb0 = efn.EfficientNetB0(weights='imagenet', include_top=False, classes=classes, input_shape=(224, 224, 3))
    model = Sequential()
    model.add(efnb0)
    #for layer in model.layers:
    #    layer.trainable = False
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.2))
    model.add(Dense(classes, activation='softmax'))

    # parameters for optimizers
    lr = 1e-3

    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    return model
