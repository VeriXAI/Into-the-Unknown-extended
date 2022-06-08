import foolbox as fb
import tensorflow as tf
import matplotlib.pyplot as plt

model = tf.keras.applications.MobileNetV2(weights="imagenet")

preprocessing = dict()
bounds = (-1, 1)
fmodel = fb.TensorFlowModel(model, bounds=bounds, preprocessing=preprocessing)

fmodel = fmodel.transform_bounds((0, 1))
assert fmodel.bounds == (0, 1)

images, labels = fb.utils.samples(fmodel, dataset='imagenet', batchsize=16)

print(fb.utils.accuracy(fmodel, images, labels))
print(type(images), images.shape)
print(type(labels), labels.shape)

attack = fb.attacks.LinfDeepFoolAttack()
raw, clipped, is_adv = attack(fmodel, images, labels, epsilons=0.03)
print(is_adv)

fb.plot.images(images)
fb.plot.images(raw)
fb.plot.images(raw - images, n=4, bounds=(-1, 1), scale=4.)
plt.show()
