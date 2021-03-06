from utils import *
from trainers import *
from monitoring import *
from monitoring.online import *
from run.experiment_helper import *


def run_script(classes=None, seed=0, epochs=10, model_path=None, *args, **kwargs):
    """
    Parameters:
    classes - a list of class indices; can also be:
       * None (interpreted as 'all classes')
       * an integer k (interpreted as 'k random classes')
    """

    data_name = "CIFAR10"
    model_name = "VGG_CIFAR10"#"CIFAR_EffNet"
    n_epochs = epochs
    batch_size = 32
    plot_name = "!"  # None = no plots, "" = show plots, "!" = store plots
    n_classes_total = 10

    if classes is None:
        classes = [k for k in range(n_classes_total)]
    elif isinstance(classes, int):
        classes = get_random_classes(classes, n_classes_total)


    data_train_model = DataSpec(randomize=False, classes=classes)
    data_test_model = DataSpec(randomize=False, classes=classes)
    classes_string = classes2string(classes)
    if model_path is None:
        model_path = "VGG_CIFAR10_{}.h5".format(classes_string)
        if plot_name == "!":
            plot_name_current = "VGG_CIFAR10_{}".format(classes_string)
        else:
            plot_name_current = plot_name

    run_training(
        data_name=data_name, data_train_model=data_train_model, data_test_model=data_test_model,
        model_name=model_name, model_path=model_path, n_epochs=n_epochs, batch_size=batch_size,
        seed=seed, plot_name=None)



#
# def run_script(classes=None, seed=0,  epochs=50):
#     """
#     Parameters:
#     classes - a list of class indices; can also be:
#        * None (interpreted as 'all classes')
#        * an integer k (interpreted as 'k random classes')
#     """
#     data_name = "CIFAR10"
#     model_name = "VGG_CIFAR10"#"CIFAR_EffNet"
#     n_epochs = epochs
#     batch_size = 32
#     plot_name = "!"  # None = no plots, "" = show plots, "!" = store plots
#     n_classes_total = 5
#
#     if classes is None:
#         classes = [k for k in range(n_classes_total)]
#     elif isinstance(classes, int):
#         classes = get_random_classes(classes, n_classes_total)
#
#     data_train_model = DataSpec(randomize=False, classes=classes)
#     data_test_model = DataSpec(randomize=False, classes=classes)
#     classes_string = classes2string(classes)
#     model_path = "VGG_CIFAR10_{}.h5".format(classes_string)
#     if plot_name == "!":
#         plot_name_current = "VGG_CIFAR10_{}".format(classes_string)
#     else:
#         plot_name_current = plot_name
#
#     run_training(
#         data_name=data_name, data_train_model=data_train_model, data_test_model=data_test_model,
#         model_name=model_name, model_path=model_path, n_epochs=n_epochs, batch_size=batch_size,
#         seed=seed, plot_name=plot_name_current)


if __name__ == "__main__":
    run_script()
