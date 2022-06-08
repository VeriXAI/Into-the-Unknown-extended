from utils import *
from abstractions import *
from trainers import *
from run.Runner import run
from monitoring import *


def run_script():
    # options
    seed = 0
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    data_name = "CIFAR10"
    data_train_model = DataSpec(randomize=False, classes=classes)
    data_test_model = DataSpec(randomize=False, classes=classes)
    data_train_monitor = DataSpec(randomize=False, classes=classes)
    data_test_monitor = DataSpec(randomize=False, classes=classes)
    data_run = DataSpec(randomize=False, classes=[x for x in range(0, 10)])
    model_name = "CIFAR_CNY19"#"GTSRB"
    model_path = "CIFAR_CNY19-model.h5"
    n_epochs = 30
    batch_size = 128
    score_fun = F1Score()
    confidence_thresholds = uniform_bins(100)

    # model trainer
    model_trainer = StandardTrainer()

    # abstractions
    epsilon = 0.0
    confidence_fun = euclidean_mean_distance
    layer2abstraction1 = {-2: BoxAbstraction(confidence_fun, epsilon=epsilon)}
    layer2abstraction2 = {-2: ZoneAbstraction(confidence_fun, epsilon=epsilon)}
    layer2abstraction3 = {-2: ConvexHullAbstraction(confidence_fun)}
    layer2abstraction4 = {-2: MeanBallAbstraction(confidence_fun, epsilon=epsilon)}
    layer2abstraction5 = {-2: BooleanAbstraction(gamma=2)}
    layer2abstraction6 = {-3: BoxAbstraction(confidence_fun, epsilon=epsilon),
                          -2: BoxAbstraction(confidence_fun, epsilon=epsilon)}
    layer2abstraction7 = {-2: PartitionBasedAbstraction(1, partition=uniform_partition(40, 2),
                                                        abstractions=ConvexHullAbstraction(confidence_fun,
                                                                                           remove_redundancies=True))}
    layer2abstraction8 = {-2: PartitionBasedAbstraction(1, partition=uniform_partition(40, 2),
                                                        abstractions=MeanBallAbstraction(confidence_fun,
                                                                                         epsilon=epsilon))}
    layer2dimensions = {-3: [0, 1], -2: [0, 1]}
    monitors = [
        # Monitor(layer2abstraction1, score_fun, layer2dimensions),
        # Monitor(layer2abstraction2, score_fun, layer2dimensions),
        # Monitor(layer2abstraction3, score_fun, layer2dimensions),
        # Monitor(layer2abstraction4, score_fun, layer2dimensions),
        # Monitor(layer2abstraction5, score_fun, layer2dimensions),
        # Monitor(layer2abstraction6, score_fun, layer2dimensions),
        # Monitor(layer2abstraction7, score_fun, layer2dimensions),
        # Monitor(layer2abstraction8, score_fun, layer2dimensions),
    ]
    monitor_manager = MonitorManager(monitors, clustering_threshold=0.02, n_clusters=5)

    # general run script
    evaluate_combination(seed=seed, data_name=data_name, data_train_model=data_train_model,
                         data_test_model=data_test_model,
                         data_train_monitor=data_train_monitor, data_test_monitor=data_test_monitor, data_run=data_run,
                         model_trainer=model_trainer, model_name=model_name, model_path=model_path, n_epochs=n_epochs,
                         batch_size=batch_size, monitor_manager=monitor_manager,
                         confidence_thresholds=confidence_thresholds,
                         skip_image_plotting=True, alpha=0.95)


if __name__ == "__main__":
    run_script()
