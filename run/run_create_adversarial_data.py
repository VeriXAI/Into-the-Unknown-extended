import tensorflow  # necessary to prevent a crash
from pickle import dump

from run.run_foolbox_integrated import run_foolbox
from run.experiment_helper import *


def create_adversarial_dataset_from_function(instance):
    folder_name_infix, instance, n_classes, bounds, epsilons = instance()
    return create_adversarial_dataset(folder_name_infix, instance, n_classes, bounds, epsilons)


def create_adversarial_dataset(folder_name_infix, instance, n_classes, bounds, epsilons):
    # instance options
    n_images = math.inf
    print_every = 100

    # obtain adversarial examples
    _, adversarial_examples, labels = run_foolbox(ignore_misclassifications=True, plot_images=False, n_images=n_images,
                                                  n_classes=n_classes, instance=instance, epsilons=epsilons,
                                                  bounds=bounds, print_every=print_every)

    # store dataset
    storage = {"data": adversarial_examples, "labels": labels}
    file_name = DATA_PATH + "{}/adversarial_0-{:d}".format(folder_name_infix, n_classes - 1)
    with open(file_name, mode='wb') as file:
        dump(obj=storage, file=file)


if __name__ == "__main__":
    #create_adversarial_dataset_from_function(MNIST_adversarial_parameters)
    #create_adversarial_dataset_from_function(F_MNIST_adversarial_parameters)
    #create_adversarial_dataset_from_function(CIFAR10_adversarial_parameters)
    #create_adversarial_dataset_from_function(GTSRB_adversarial_parameters)
    create_adversarial_dataset_from_function(Doom_adversarial_parameters)
