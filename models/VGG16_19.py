from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.keras.optimizers import SGD


# Note: 'weights' is ignored and just present for compatibility with other networks
def VGG16_19(classes, input_shape, weights=None):
    model = VGG16(
        weights=weights,
        include_top=True,
        classes=classes,
        input_shape=input_shape
    )

    model.summary()
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy']
    )

    return model
