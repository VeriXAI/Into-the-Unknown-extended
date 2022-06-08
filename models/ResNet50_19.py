from tensorflow.python.keras.applications.resnet import ResNet50
from tensorflow.keras.optimizers import Adam


# Note: 'weights' is ignored and just present for compatibility with other networks
def ResNet50_19(classes, input_shape, weights=None):
    model = ResNet50(weights=weights, classes=classes, input_shape=input_shape)
    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model
