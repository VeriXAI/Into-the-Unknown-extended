import foolbox as fb
import tensorflow as tf
import matplotlib.pyplot as plt
import eagerpy as ep
import numpy as np

from run.experiment_helper import *
from trainers import StandardTrainer
from data import *
from utils import *

tf.compat.v1.enable_eager_execution()


def plot_monochrome(images):
    images = images.numpy()
    n = images.shape[0]
    s1 = images.shape[1]
    s2 = images.shape[2]
    fig, ax = plt.subplots(ncols=n)
    for i, image in enumerate(images):
        ax[i].imshow(image.reshape(s1, s2), cmap="gray")


def run_foolbox(ignore_misclassifications=False, plot_images=True, n_images=10, n_classes=9, instance=instance_CIFAR10,
                epsilons=0.03, bounds=(0, 255), print_every=None):
    # load instance
    model_name, data_name, stored_network_name, total_classes = instance()
    data_train_model, data_test_model, data_train_monitor, data_test_monitor, data_run, model_path, _ = \
        load_instance(n_classes, total_classes, stored_network_name)
    class_label_map, all_labels = get_data_loader(data_name)(
        data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
        data_test_monitor=data_test_monitor, data_run=data_run)
    model_trainer = StandardTrainer()
    statistics = Statistics()
    n_epochs = batch_size = -1
    model, history_model = get_model(model_name=model_name, model_path=model_path, data_train=data_train_model,
                                     data_test=data_test_model, class_label_map=class_label_map,
                                     model_trainer=model_trainer, n_epochs=n_epochs, batch_size=batch_size,
                                     statistics=statistics)

    fmodel = fb.TensorFlowModel(model, bounds=bounds)

    # filter images
    if n_images != math.inf:
        data_run.filter(filter=[i for i in range(n_images)], copy=False)

    attack = fb.attacks.LinfDeepFoolAttack()
    original_images = []
    attacked_images = []
    original_labels = []
    # TODO CS: probably have to map the labels/classes
    for i, (image, label) in enumerate(zip(data_run.inputs(), data_run.ground_truths())):
        if print_every is not None and i % print_every == 0:
            print("iteration {:d}".format(i))
        # convert to tensors
        image_tensor = ep.from_numpy(fmodel.dummy, np.stack([image])).raw
        label_tensor = ep.from_numpy(fmodel.dummy, np.stack([label]))

        # classification accuracy
        if ignore_misclassifications:
            correct_classification = fb.utils.accuracy(fmodel, image_tensor, label_tensor) == 1.0
            if not correct_classification:
                continue

        attacked_image_unbounded, attacked_image_bounded, attack_successful = attack(fmodel, image_tensor, label_tensor,
                                                                                     epsilons=epsilons)
        attack_successful_bool = attack_successful.numpy()[0]  # convert to bool
        if not attack_successful_bool:
            continue
        # sanity check: network should misclassify
        assert fb.utils.accuracy(fmodel, attacked_image_bounded, label_tensor) == 0.0
        original_images.append(image_tensor)
        attacked_images.append(attacked_image_bounded)
        original_labels.append(label)

    # convert lists to tensors
    original_images = tf.concat(original_images, 0)
    attacked_images = tf.concat(attacked_images, 0)

    if plot_images:
        if data_run.inputs()[0].shape[2] == 1:
            plot_monochrome(original_images)
            plot_monochrome(attacked_images)
            plot_monochrome(attacked_images - original_images)
        else:
            fb.plot.images(original_images)
            fb.plot.images(attacked_images)
            fb.plot.images(attacked_images - original_images)
        plt.show()

    return original_images.numpy(), attacked_images.numpy(), original_labels


if __name__ == "__main__":
    run_foolbox()
