from utils import *
from trainers import *
from monitoring import *
from monitoring.online import *
from run.experiment_helper import *


def run_script(classes=None, seed=0, epochs=10, epochs_ratio=0.2, model_path=None, experiment_name="EX",
               accuracy_threshold=0.90, base=10, batch_size=128,  freeze=0.5, plot=False, *args, **kwargs):
    """
    Parameters:
    classes - a list of class indices; can also be:
       * None (interpreted as 'all classes')
       * an integer k (interpreted as 'k random classes')
    """
    data_name = "MNIST"
    model_name = "MNIST"
    n_epochs = round(epochs*epochs_ratio)
    epochs_distance = epochs-n_epochs
    plot_name = ""  # None = no plots, "" = show plots, "!" = store plots
    n_classes_total = 10

    if classes is None:
        classes = [k for k in range(n_classes_total)]
    elif isinstance(classes, int):
        classes = get_random_classes(classes, n_classes_total)

    data_train_model = DataSpec(randomize=False, classes=classes)
    data_test_model = DataSpec(randomize=False, classes=classes)
    classes_string = classes2string(classes)
    if model_path is None:
        model_path = "AT_CNY19id1_MNIST_{}.h5".format(classes_string)
        if plot_name == "!":
            plot_name_current = "AT_CNY19id1_MNIST_{}".format(classes_string)
        else:
            plot_name_current = plot_name

    raw_monitor = box_abstraction_MNIST()
    distance_fun = box_distance_parameter
    distance_monitor = DistanceMonitor(monitor=raw_monitor, distance_fun=distance_fun)
    clustering_threshold = 0.07
    layer = 8
    layer2n_components = {layer: 10}
    monitor_manager = MonitorManager([distance_monitor], clustering_threshold=clustering_threshold,
                                     skip_confidence=False, fit_distribution_method=None, layer2n_components=layer2n_components)

    trainer = AbstractionTrainer(monitor_manager, epochs_distance=epochs_distance, accuracy_threshold=accuracy_threshold, base=base,
                                 experiment_name=experiment_name, freeze=freeze, plot=plot)
    run_training(
        data_name=data_name, data_train_model=data_train_model, data_test_model=data_test_model,
        model_name=model_name, model_path=model_path, n_epochs=n_epochs, batch_size=batch_size,
        seed=seed, plot_name=None, model_trainer=trainer)

if __name__ == "__main__":
    run_script([0, 1])
