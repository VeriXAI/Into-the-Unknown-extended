from utils import *
from trainers import *


def run_script():
    seed = 0
    data_name = "CIFAR100"
    model_name = "CIFAR_100"
    n_epochs = {2:10,10:250,20:250,50:250,80:250,90:250,98:250,100:250}
    batch_size = 128
    plot_name = "!"  # None = no plots, "" = show plots, "!" = store plots

    for n_classes in [50, 100]:
        classes = [k for k in range(n_classes)]
        data_train_model = DataSpec(randomize=False, classes=classes)
        data_test_model = DataSpec(randomize=False, classes=classes)
        classes_string = classes2string(classes)
        model_path = "CNY19id2_CIFAR100_{}.h5".format(classes_string)
        if plot_name == "!":
            plot_name_current = "CNY19id2_CIFAR100_{}".format(classes_string)
        else:
            plot_name_current = plot_name

        run_training(
            data_name=data_name, data_train_model=data_train_model, data_test_model=data_test_model,
            model_name=model_name, model_path=model_path, n_epochs=n_epochs[n_classes], batch_size=batch_size,
            seed=seed, plot_name=plot_name_current)


if __name__ == "__main__":
    run_script()
