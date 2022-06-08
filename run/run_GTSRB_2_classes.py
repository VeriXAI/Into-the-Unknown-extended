from utils import *
from abstractions import *
from trainers import *
from run.Runner import run
from monitoring import *


def run_script():
    # options
    seed = 0
    classes = [0, 1]
    data_name = "GTSRB"
    data_train_model = DataSpec(randomize=False, classes=classes)
    data_test_model = DataSpec(randomize=False, classes=classes)
    data_train_monitor = DataSpec(randomize=False, classes=classes)
    data_test_monitor = DataSpec(randomize=False, classes=classes)
    data_run = DataSpec(randomize=False, classes=classes)
    model_name = "GTSRB"
    model_path = "GTSRB_CNY19_2-model.h5"
    n_epochs = 20
    batch_size = 128
    score_fun = F1Score()

    # model trainer
    model_trainer = StandardTrainer()

    # abstractions
    confidence_fun = euclidean_mean_distance
    layer2abstraction = {-2: MeanBallAbstraction(confidence_fun, 10, 0.0)}
    layer2dimensions = {-2: [0, 1]}
    monitors = [Monitor(layer2abstraction, score_fun, layer2dimensions)]
    monitor_manager = MonitorManager(monitors)

    # general run script
    run(seed=seed, data_name=data_name, data_train_model=data_train_model, data_test_model=data_test_model,
        data_train_monitor=data_train_monitor, data_test_monitor=data_test_monitor, data_run=data_run,
        model_trainer=model_trainer, model_name=model_name, model_path=model_path, n_epochs=n_epochs,
        batch_size=batch_size, monitor_manager=monitor_manager)


if __name__ == "__main__":
    run_script()
