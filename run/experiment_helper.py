from utils import *
from monitoring import *
from abstractions import *
from tensorflow.keras import optimizers


def load_instance(classes, total_classes, stored_network_name):
    """
    Parameters:
    classes - a list of class indices; can also be an integer k (interpreted as 'the first k classes')
    """
    Monitor.reset_ids()
    if isinstance(classes, int):
        classes = [k for k in range(classes)]
    data_train_model = DataSpec(randomize=False, classes=classes)
    data_test_model = DataSpec(randomize=False, classes=classes)
    data_train_monitor = DataSpec(randomize=False, classes=classes)
    data_test_monitor = DataSpec(randomize=False, classes=classes)
    data_run = DataSpec(randomize=False, classes=[k for k in range(0, total_classes)])
    classes_string = classes2string(classes)
    model_path = "{}_{}.h5".format(stored_network_name, classes_string)
    transfer_model_path = "{}_trans_{}.h5".format(stored_network_name, classes_string)

    return data_train_model, data_test_model, data_train_monitor, data_test_monitor, data_run, model_path, \
           transfer_model_path, classes_string


def print_general_statistics(statistics, data_train_monitor, data_run, show_training=True, show_running=True):
    print("overall statistics")
    if show_training:
        print("{} seconds for extracting {:d} values during monitor training".format(
            float_printer(statistics.time_training_monitor_value_extraction), data_train_monitor.n()))
        print("{} seconds for clustering during monitor training".format(
            float_printer(statistics.time_training_monitor_clustering)))
        print("{} seconds for monitor training on {:d} samples".format(
            float_printer(statistics.time_training_monitor_tweaking), data_train_monitor.n()))
    if show_running:
        print("{} seconds for extracting {:d} values during running the monitored session".format(
            float_printer(statistics.time_running_monitor_value_extraction), data_run.n()))
        print("{} seconds for running the monitored session on {:d} samples".format(
            float_printer(statistics.time_running_monitor_classification), data_run.n()))


def print_monitor_statistics_single(monitor, statistics, data_train_monitor, data_run):
    m_id = monitor.id()
    print("\nprinting statistics for monitor {:d} with abstraction structure {}".format(m_id, monitor.short_str()))
    time_training = statistics.time_tweaking_each_monitor[m_id]
    print("{} seconds for training the monitor on {:d} samples".format(
        float_printer(time_training), data_train_monitor.n()))
    time_running = statistics.time_running_each_monitor[m_id]
    print("{} seconds for running the monitor on {:d} samples".format(
        float_printer(time_running), data_run.n()))
    return time_training, time_running


def print_monitor_statistics(monitors, statistics, data_train_monitor, data_run):
    for monitor in monitors:
        print_monitor_statistics_single(monitor, statistics, data_train_monitor, data_run)


# modifies: storage_monitors
def print_and_store_monitor_statistics(storage_monitors, monitors, statistics, history_run, novelty_wrapper_run,
                                       data_train_monitor, data_run):
    for monitor in monitors:
        m_id = monitor.id()
        time_training, time_running = print_monitor_statistics_single(monitor, statistics, data_train_monitor, data_run)
        history_run.update_statistics(m_id)
        fn = history_run.false_negatives()
        fp = history_run.false_positives()
        tp = history_run.true_positives()
        tn = history_run.true_negatives()
        novelty_results = novelty_wrapper_run.evaluate_detection(m_id)
        storage = CoreStatistics(fn=fn, fp=fp, tp=tp, tn=tn,
                                 novelties_detected=len(novelty_results["detected"]),
                                 novelties_undetected=len(novelty_results["undetected"]),
                                 time_training=time_training, time_running=time_running)
        storage_monitors[m_id - 1].append(storage)

def instance_AT_MNIST(transfer=False):
    model_name = "MNIST"
    data_name = "MNIST"
    if not transfer:
        stored_network_name = "AT_CNY19id1_MNIST"
    else:
        stored_network_name = "AT_CNY19id1_MNIST_transfer"
    total_classes = 10
    flatten_layer = 3
    optimizer = optimizers.Adam(lr=0.001)
    return model_name, data_name, stored_network_name, total_classes, flatten_layer, optimizer

def instance_MNIST(transfer=False):
    model_name = "MNIST"
    data_name = "MNIST"
    if not transfer:
        stored_network_name = "CNY19id1_MNIST"
    else:
        stored_network_name = "CNY19id1_MNIST_transfer"
    total_classes = 10
    flatten_layer = 3
    optimizer = optimizers.Adam(lr=0.001)
    return model_name, data_name, stored_network_name, total_classes, flatten_layer, optimizer


def instance_F_MNIST(transfer=False):
    model_name = "F_MNIST"
    data_name = "F_MNIST"
    if not transfer:
        stored_network_name = "CNY19id1_F_MNIST"
    else:
        stored_network_name = "CNY19id1_F_MNIST_transfer"
    total_classes = 10
    flatten_layer = 4
    optimizer = optimizers.Adam(lr=0.001)
    return model_name, data_name, stored_network_name, total_classes, flatten_layer, optimizer

def instance_AT_F_MNIST(transfer=False):
    model_name = "F_MNIST"
    data_name = "F_MNIST"
    if not transfer:
        stored_network_name = "AT_CNY19id1_F_MNIST"
    else:
        stored_network_name = "AT_CNY19id1_F_MNIST_transfer"
    total_classes = 10
    flatten_layer = 4
    optimizer = optimizers.Adam(lr=0.001)
    return model_name, data_name, stored_network_name, total_classes, flatten_layer, optimizer


def instance_EMNIST(transfer=False):
    model_name = "EMNIST"
    data_name = "EMNIST"
    if not transfer:
        stored_network_name = "CNY19id1_EMNIST"
    else:
        stored_network_name = "CNY19id1_EMNIST_transfer"
    total_classes = 47
    flatten_layer = 4
    optimizer = optimizers.Adam(lr=0.001)
    return model_name, data_name, stored_network_name, total_classes, flatten_layer, optimizer

def instance_AT_EMNIST(transfer=False):
    model_name = "EMNIST"
    data_name = "EMNIST"
    if not transfer:
        stored_network_name = "AT_CNY19id1_EMNIST"
    else:
        stored_network_name = "AT_CNY19id1_EMNIST_transfer"
    total_classes = 47
    flatten_layer = 4
    optimizer = optimizers.Adam(lr=0.001)
    return model_name, data_name, stored_network_name, total_classes, flatten_layer, optimizer


def instance_CIFAR10(transfer=False):
    model_name = "VGG_CIFAR10"
    data_name = "CIFAR10"
    if not transfer:
        stored_network_name = "VGG_CIFAR10"#"CNY19id2_CIFAR10"
    else:
        stored_network_name = "VGG_CIFAR10_transfer"#"CNY19id2_CIFAR10_transfer"
    total_classes = 10
    flatten_layer = 9#8
    optimizer = optimizers.SGD(lr=0.001, momentum=0.9)#optimizers.RMSprop(lr=0.001)
    return model_name, data_name, stored_network_name, total_classes, flatten_layer, optimizer

def instance_AT_CIFAR10(transfer=False):
    model_name = "VGG_CIFAR10"
    data_name = "CIFAR10"
    if not transfer:
        stored_network_name = "AT_VGG_CIFAR10"#"CNY19id2_CIFAR10"
    else:
        stored_network_name = "AT_VGG_CIFAR10_transfer"#"CNY19id2_CIFAR10_transfer"
    total_classes = 10
    flatten_layer = 9#8
    optimizer = optimizers.SGD(lr=0.001, momentum=0.9)#optimizers.RMSprop(lr=0.001)
    return model_name, data_name, stored_network_name, total_classes, flatten_layer, optimizer



def instance_CIFAR100(transfer=False):
    model_name = "CIFAR_100"
    data_name = "CIFAR100"
    if not transfer:
        stored_network_name = "CNY19id2_CIFAR100"
    else:
        stored_network_name = "CNY19id2_CIFAR100_transfer"
    total_classes = 100
    flatten_layer = 8
    optimizer = optimizers.RMSprop(lr=0.001)
    return model_name, data_name, stored_network_name, total_classes, flatten_layer, optimizer


def instance_CIFAR10_100(transfer=False):
    model_name = "CIFAR10_100"
    data_name = "CIFAR100"
    if not transfer:
        stored_network_name = "cifar10vgg"
    else:
        stored_network_name = "cifar10vgg_transfer"
    total_classes = 100
    flatten_layer = -7
    learning_rate = 0.1
    lr_decay = 1e-6
    # optimization details
    optimizer = optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    return model_name, data_name, stored_network_name, total_classes, flatten_layer, optimizer


def instance_GTSRB(transfer=False):
    model_name = "GTSRB"
    data_name = "GTSRB"
    if not transfer:
        stored_network_name = "CNY19id2_GTSRB"
    else:
        stored_network_name = "CNY19id2_GTSRB_transfer"
    total_classes = 43
    flatten_layer = 8
    optimizer = optimizers.RMSprop(lr=0.001)
    return model_name, data_name, stored_network_name, total_classes, flatten_layer, optimizer

def instance_AT_GTSRB(transfer=False):
    model_name = "GTSRB"
    data_name = "GTSRB"
    if not transfer:
        stored_network_name = "AT_CNY19id2_GTSRB"
    else:
        stored_network_name = "AT_CNY19id2_GTSRB_transfer"
    total_classes = 43
    flatten_layer = 8
    optimizer = optimizers.RMSprop(lr=0.001)
    return model_name, data_name, stored_network_name, total_classes, flatten_layer, optimizer


def instance_Doom():
    model_name = "Doom"
    data_name = "Doom"
    stored_network_name = "amodel_BasicDoom"
    total_classes = 3
    return model_name, data_name, stored_network_name, total_classes


def instance_MELMAN(rolling=True, multi=True, transfer=False):
    if multi:
        model_name = "MELMAN_2"
    else:
        model_name = "MELMAN"
    data_name = "MELMAN"
    if rolling:
        stored_network_name = "melman_2017_rolling"
    else:
        stored_network_name = "melman_2017"
    if multi:
        stored_network_name = "melman_2017_multi"
    if transfer:
        stored_network_name = "melman_2017_transfer"
    total_classes = 3
    return model_name, data_name, stored_network_name, total_classes


def instance_MELMAN_LSTM(transfer=False):
    model_name = "MELMAN_LSTM"
    data_name = "MELMAN"
    if transfer:
        stored_network_name = "melman_2017_LSTM_transfer"
    else:
        stored_network_name = "melman_2017_LSTM"
    flatten_layer = -2
    total_classes = 3
    optimizer = optimizers.Adam(lr=0.001)
    return model_name, data_name, stored_network_name, total_classes, flatten_layer, optimizer


def box_abstraction_given_layers(layers, epsilon=0.):
    layer2abstraction = dict()
    for layer in layers:
        layer2abstraction[layer] = BoxAbstraction(euclidean_mean_distance, epsilon=epsilon)
    return layer2abstraction


def meanball_abstraction_given_layers(layers, epsilon=0.):
    layer2abstraction = dict()
    for layer in layers:
        layer2abstraction[layer] = MeanBallAbstraction(euclidean_mean_distance, epsilon=epsilon)
    return layer2abstraction


def box_abstraction_MNIST(epsilon=0., learn_from_test_data=False):
    layer2abstraction = box_abstraction_given_layers(layers=[-2], epsilon=epsilon)
    return Monitor(layer2abstraction=layer2abstraction, learn_from_test_data=learn_from_test_data)


def euclidean_abstraction_MNIST(epsilon=0., learn_from_test_data=False):
    layer2abstraction = meanball_abstraction_given_layers(layers=[-2], epsilon=epsilon)
    return Monitor(layer2abstraction=layer2abstraction, learn_from_test_data=learn_from_test_data)


def box_abstraction_F_MNIST(epsilon=0., learn_from_test_data=False):
    layer2abstraction = box_abstraction_given_layers(layers=[-2], epsilon=epsilon)
    return Monitor(layer2abstraction=layer2abstraction, learn_from_test_data=learn_from_test_data)


def box_abstraction_EMNIST(epsilon=0., learn_from_test_data=False):
    layer2abstraction = box_abstraction_given_layers(layers=[-2], epsilon=epsilon)
    return Monitor(layer2abstraction=layer2abstraction, learn_from_test_data=learn_from_test_data)


def box_abstraction_CIFAR10(epsilon=0., learn_from_test_data=False):
    layer2abstraction = box_abstraction_given_layers(layers=[-3], epsilon=epsilon)
    return Monitor(layer2abstraction=layer2abstraction, learn_from_test_data=learn_from_test_data)


def box_abstraction_CIFAR100(epsilon=0., learn_from_test_data=False):
    layer2abstraction = box_abstraction_given_layers(layers=[-2], epsilon=epsilon)
    return Monitor(layer2abstraction=layer2abstraction, learn_from_test_data=learn_from_test_data)


def box_abstraction_CIFAR10_100(epsilon=0., learn_from_test_data=False):
    layer2abstraction = box_abstraction_given_layers(layers=[-6], epsilon=epsilon)
    return Monitor(layer2abstraction=layer2abstraction, learn_from_test_data=learn_from_test_data)


def box_abstraction_GTSRB(epsilon=0., learn_from_test_data=False):
    layer2abstraction = box_abstraction_given_layers(layers=[-2], epsilon=epsilon)
    print(layer2abstraction, "XXX")
    return Monitor(layer2abstraction=layer2abstraction, learn_from_test_data=learn_from_test_data)


def box_abstraction_Doom(epsilon=0., learn_from_test_data=False):
    layer2abstraction = box_abstraction_given_layers(layers=[-2], epsilon=epsilon)
    return Monitor(layer2abstraction=layer2abstraction, learn_from_test_data=learn_from_test_data)


def box_abstraction_MELMAN(epsilon=0., learn_from_test_data=False):
    layer2abstraction = box_abstraction_given_layers(layers=[-2], epsilon=epsilon)
    return Monitor(layer2abstraction=layer2abstraction, learn_from_test_data=learn_from_test_data)


def octagon_abstraction_given_layers(layers, epsilon=0.):
    layer2abstraction = dict()
    for layer in layers:
        layer2abstraction[layer] = OctagonAbstraction(euclidean_mean_distance, epsilon=epsilon)
    return layer2abstraction


def octagon_abstraction_MNIST(epsilon=0., learn_from_test_data=False):
    layer2abstraction = octagon_abstraction_given_layers(layers=[-1, -2, -3, -4], epsilon=epsilon)
    return Monitor(layer2abstraction=layer2abstraction, learn_from_test_data=learn_from_test_data)


def octagon_abstraction_F_MNIST(epsilon=0., learn_from_test_data=False):
    layer2abstraction = octagon_abstraction_given_layers(layers=[-1, -2, -3, -4], epsilon=epsilon)
    return Monitor(layer2abstraction=layer2abstraction, learn_from_test_data=learn_from_test_data)


def octagon_abstraction_CIFAR10(epsilon=0., learn_from_test_data=False):
    layer2abstraction = octagon_abstraction_given_layers(layers=[-1, -2, -3, -4], epsilon=epsilon)
    return Monitor(layer2abstraction=layer2abstraction, learn_from_test_data=learn_from_test_data)


def octagon_abstraction_GTSRB(epsilon=0., learn_from_test_data=False):
    layer2abstraction = octagon_abstraction_given_layers(layers=[-2], epsilon=epsilon)
    return Monitor(layer2abstraction=layer2abstraction, learn_from_test_data=learn_from_test_data)


def chull_abstraction_given_layers(layers, epsilon=0.):
    layer2abstraction = dict()
    for layer in layers:
        layer2abstraction[layer] = ConvexHullAbstraction(euclidean_mean_distance, epsilon=epsilon)
    return layer2abstraction


def chull_abstraction_MNIST(epsilon=0., learn_from_test_data=False):
    layer2abstraction = chull_abstraction_given_layers(layers=[-2], epsilon=epsilon)
    return Monitor(layer2abstraction=layer2abstraction, learn_from_test_data=learn_from_test_data)


def chull_abstraction_F_MNIST(epsilon=0., learn_from_test_data=False):
    layer2abstraction = chull_abstraction_given_layers(layers=[-1, -2, -3, -4], epsilon=epsilon)
    return Monitor(layer2abstraction=layer2abstraction, learn_from_test_data=learn_from_test_data)


def chull_abstraction_CIFAR10(epsilon=0., learn_from_test_data=False):
    layer2abstraction = chull_abstraction_given_layers(layers=[-2], epsilon=epsilon)
    return Monitor(layer2abstraction=layer2abstraction, learn_from_test_data=learn_from_test_data)


def chull_abstraction_GTSRB(epsilon=0., learn_from_test_data=False):
    layer2abstraction = chull_abstraction_given_layers(layers=[-2], epsilon=epsilon)
    return Monitor(layer2abstraction=layer2abstraction, learn_from_test_data=learn_from_test_data)


def chull_abstraction_MELMAN(epsilon=0., learn_from_test_data=False):
    layer2abstraction = chull_abstraction_given_layers(layers=[0, -1, -2], epsilon=epsilon)
    return Monitor(layer2abstraction=layer2abstraction, learn_from_test_data=learn_from_test_data)


def MNIST_adversarial_parameters():
    folder_name_infix = "MNIST"
    instance = instance_MNIST
    n_classes = 9
    bounds = (0, 255)
    epsilons = 0.2
    return folder_name_infix, instance, n_classes, bounds, epsilons


def F_MNIST_adversarial_parameters():
    folder_name_infix = "Fashion_MNIST"
    instance = instance_F_MNIST
    n_classes = 9
    bounds = (0, 1)
    epsilons = 0.2
    return folder_name_infix, instance, n_classes, bounds, epsilons


def CIFAR10_adversarial_parameters():
    folder_name_infix = "cifar-10-python/cifar-10-batches-py"
    instance = instance_CIFAR10
    n_classes = 9
    bounds = (0, 255)
    epsilons = 0.03
    return folder_name_infix, instance, n_classes, bounds, epsilons


def GTSRB_adversarial_parameters():
    folder_name_infix = "GTSRB"
    instance = instance_GTSRB
    n_classes = 20
    bounds = (0, 255)
    epsilons = 0.03
    return folder_name_infix, instance, n_classes, bounds, epsilons


def Doom_adversarial_parameters():
    folder_name_infix = "Doom"
    instance = instance_Doom
    n_classes = 5
    bounds = (0, 255)
    epsilons = 0.05
    return folder_name_infix, instance, n_classes, bounds, epsilons


def MELMAN_adversarial_parameters():
    folder_name_infix = "MELMAN"
    instance = instance_MELMAN
    n_classes = 2
    bounds = (0, 255)  # TODO: what to do with sequential data?
    epsilons = 0.05
    return folder_name_infix, instance, n_classes, bounds, epsilons
