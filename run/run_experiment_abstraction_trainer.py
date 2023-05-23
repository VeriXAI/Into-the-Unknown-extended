import random
from utils.AbstractionTrainerAnalysis import analyse
from data import *
from monitoring.online import DistanceMonitor, box_distance_parameter, MonitorWrapper
from run.experiment_helper import *
from run.experiment_collection import *
from trainers import *
from utils.Options import MODEL_INSTANCE_PATH
import os
import json
import time
import shutil
import re
from pprint import pprint


def translate_name(name):
    DATASET_NAME = [
        ("GTSRB", "GTSRB"),
        ("FMNIST", "FMNIST"),
        ("MNIST", "MNIST")]

    EXPERIMENT_NAME = {
        "01-2": "1a",
        "01-2345": "1b",
        "01-23456789": "1c",
        "01234-5": "2a",
        "01234-56789": "2c",
        "012345678-9": "3c",
        "01-2345678910111213141516171819202122": "1b",
        "01-23456789101112131415161718192021222324252627282930313233343536373839404142": "1c",
        "0123456789101112131415161718192021-22": "2a",
        "0123456789101112131415161718192021-222324252627282930313233343536373839404142": "2c",
        "01234567891011121314151617181920212223242526272829303132333435363738394041-42": "3c",
    }
    dataset = "Unknown"
    for k, v in DATASET_NAME:
        if k in name:
            dataset = v
            break
    experiment = EXPERIMENT_NAME[re.findall("\d+-\d+", name)[0]]
    key = name.split("_")[0]
    suffix = "." + name.split(".")[-1]
    additional = ""
    if key == "network":
        if "AT_" in name:
            additional += "A"
        else:
            additional += "B"
    if key == "network" or key == "figure-projection":
        additional += re.findall("_n\d+_s\d+", name)[0]
    experiment += additional
    return key, dataset, experiment, suffix


def generate_output_dir(directory):
    base_path = directory
    path = "results"
    DIR_MAP = {
        "data": os.path.join(base_path, "data"),
        "figure-projection": os.path.join(base_path, "figure-projection"),
        "network": os.path.join(base_path, "network"),
        "training-data": os.path.join(base_path, "training-data"),
    }
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    os.mkdir(base_path)
    for i in DIR_MAP.values():
        os.mkdir(i)
    for name in os.listdir(path):
        key, dataset, experiment, suffix = translate_name(name)
        if not os.path.exists(os.path.join(base_path, key, dataset)):
            os.mkdir(os.path.join(base_path, key, dataset))
        shutil.copy(os.path.join(path, name), os.path.join(base_path, key, dataset, experiment) + suffix)


def run_all_experiments():
    if os.path.exists("results"):
        shutil.rmtree("results")
    os.mkdir("results")
    number_of_experiments = 5
    for ex in [MNIST_experiments(), F_MNIST_experiments(), GTSRB_experiments()]:
        run_experiment(experiments=ex, ex_name="Abstraction_Trainer_Experiment", n=number_of_experiments, plot=True)
    data_dir = "experiment_data"
    generate_output_dir(data_dir)
    analyse(data_dir)


def train_models():
    if os.path.exists("results"):
        shutil.rmtree("results")
    os.mkdir("results")
    number_of_experiments = 5
    for ex in [MNIST_experiments(), F_MNIST_experiments(), GTSRB_experiments()]:
        run_experiment(experiments=ex, ex_name="Abstraction_Trainer_Experiment", n=number_of_experiments, plot=True)
    data_dir = "experiment_data"
    generate_output_dir(data_dir)
    shutil.rmtree("results")


def evaluate_data():
    data_dir = "experiment_data"
    if not os.path.exists(data_dir):
        train_models()
    analyse(data_dir)


def run_experiment(experiments, n=2, start_seed=0, ex_name='EX', plot=False):
    for experiment in experiments:
        results = {"abstraction": [], "base": [], "training_log": []}
        random.seed(start_seed)
        name, abstraction_experiment, base_experiment = experiment()
        experiment_name = '{}_n{}_{}'.format(ex_name, n, name)
        # ds = name.split("_")[0]
        for i in range(n):
            seed = random.randint(0, 200)
            print('')
            print('#' * 100)
            print('EXPERIMENT: {} (Run {}, seed {})'.format(experiment_name, i, seed))
            print('#' * 100)

            try:
                print('')
                print('#'*100)
                print('EXPERIMENT: {} (Run {}, seed {})'.format(experiment_name, i, seed))
                print('#'*100)
                s_experiment_name = '{}_n{}_s{}'.format(experiment_name, i, seed)
                # seed = start_seed

                result_ae = run_single_experiment(*abstraction_experiment, seed=seed, exp_name=s_experiment_name, plot=plot)
                result_be = run_single_experiment(*base_experiment, seed=seed, exp_name=s_experiment_name, plot=plot)
                results["abstraction"].append(result_ae)
                results["base"].append(result_be)
                with open('results/training-data_{}.json'.format(s_experiment_name), 'r') as f:
                    results["training_log"].append((result_ae[0], json.load(f)))
                with open('results/data_{}.json'.format(experiment_name), 'w') as f:
                    json.dump(results, f)
            except Exception as e:
                    with open('results/error_{}.txt'.format(experiment_name), 'a') as f:
                        f.write('EXPERIMENT: {0} (Run {1}, seed {2}) -> {3}\n'.format(experiment_name, i, seed, e))
                    print("ERROR:", 'EXPERIMENT: {0} (Run {1}, seed {2} -> {3})'.format(experiment_name, i, seed, e))


def remove_model(model_path):
    model_path = os.path.join(MODEL_INSTANCE_PATH, model_path)
    if os.path.exists(model_path):
        os.remove(model_path)


def move_model(model_path):
    src_path = os.path.join(MODEL_INSTANCE_PATH, model_path)
    dst_path = os.path.join("results", "network_" + model_path)
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)


def run_single_experiment(trainer, instance, known_classes, all_classes, interesting_classes, epochs,
                          accuracy_threshold, base, batch_size, freeze, seed=0, exp_name='EX', plot=False, *args, **kwargs):
    start_time = time.time()
    model_name, data_name, stored_network_name, total_classes, flatten_layer, optimizer = instance(transfer=False)
    classes_string = classes2string(known_classes)
    all_classes_string = classes2string(all_classes)
    if "trans" in stored_network_name:
        model_path = "{}_{}_{}.h5".format(exp_name, stored_network_name, all_classes_string)
    else:
        model_path = "{}_{}_{}.h5".format(exp_name, stored_network_name, all_classes_string)
    remove_model(model_path)
    start_time_train = time.time()
    trainer(known_classes, seed=seed, epochs=epochs, model_path=model_path, experiment_name=exp_name,
            accuracy_threshold=accuracy_threshold, base=base, batch_size=batch_size, freeze=freeze, plot=plot)
    end_time_train = time.time()

    data_train_model = DataSpec(randomize=False, classes=known_classes)
    data_test_model = DataSpec(randomize=False, classes=known_classes)
    data_train_monitor = DataSpec(randomize=False, classes=known_classes)
    data_test_monitor = DataSpec(randomize=False, classes=known_classes)
    data_run = DataSpec(randomize=False, classes=interesting_classes)
    clustering_threshold = 0.07
    layer = 8
    layer2n_components = {layer: 10}  # dimension reduction; use 'None' to deactivate

    class_label_map, all_labels = get_data_loader(data_name)(
        data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
        data_test_monitor=data_test_monitor, data_run=data_run)

    model, _ = get_model(model_name=model_name, model_path=model_path, data_train=data_train_model,
                         data_test=data_test_model, class_label_map=class_label_map, model_trainer=StandardTrainer(),
                         n_epochs=epochs, batch_size=batch_size, statistics=Statistics())

    print_data_information(data_train_monitor, data_test_monitor, data_run)
    # create monitor
    layer2abstraction = {layer: BoxAbstraction(euclidean_mean_distance)}
    # monitor = Monitor(layer2abstraction=layer2abstraction)
    monitor = Monitor(layer2abstraction=layer2abstraction)
    # monitor = DistanceMonitor(monitor, box_distance_parameter)
    monitor_manager = MonitorManager([monitor], clustering_threshold=clustering_threshold, skip_confidence=False,
                                     layer2n_components=layer2n_components,
                                     fit_distribution_method=None)
    monitor_wrapper = MonitorWrapper(monitor_manager=monitor_manager)
    monitor_wrapper.score_thresholds = {0: 1.0, 1: 1.0, 2: 1.0}  # , 3: 1.0, 4: 1.0, 5: 1.0}

    # run instance
    start_time_monitor = time.time()
    monitor_manager.normalize_and_initialize(model, class_label_map=class_label_map, n_classes_total=total_classes)
    monitor_manager.train(model=model, data_train=data_train_monitor, data_test=data_test_monitor,
                          statistics=Statistics())
    end_time_monitor = time.time()

    # training data
    history = History()
    history.set_ground_truths(data_train_monitor.ground_truths())
    layer2values, _, _ = obtain_predictions(model=model, data=data_train_monitor, class_label_map=class_label_map,
                                            layers=[layer])
    if layer2n_components is not None:
        layer2values, layer2components = \
            reduce_dimension(layer2data=layer2values, layers=monitor_manager.layers(),
                             layer2n_components=layer2n_components,
                             layer2components=monitor_manager.layer2components)
        plot_title = " training with reduced data"
    else:
        plot_title = " training with original data"
    history.set_layer2values(layer2values)

    start_time_monitor_eval = time.time()
    history_run_all = monitor_manager.run(model=model, data=data_run, statistics=Statistics())
    end_time_monitor_eval = time.time()

    history_run_all.update_statistics(monitor.id())
    end_time = time.time()
    data_test_tf = tf.data.Dataset.from_tensor_slices((data_test_model.inputs(), data_test_model.categoricals()))
    data_test_tf = data_test_tf.batch(batch_size)
    eval_results = model.evaluate(data_test_tf, batch_size=batch_size)

    res = {"monitor": {"fn": history_run_all._fn,
            "fp": history_run_all._fp,
            "tn": history_run_all._tn,
            "tp": history_run_all._tp},
           "network": {"accuracy": eval_results},
           "time": {"all":[start_time, end_time],
                    "network_train": [start_time_train, end_time_train],
                    "monitor_train": [start_time_monitor, end_time_monitor],
                    "monitor_eval": [start_time_monitor_eval, end_time_monitor_eval]}}

    outside_the_box = []
    for i, (prediction_i, ground_truth_i) in enumerate(zip(history_run_all.predictions, history_run_all.ground_truths)):
        is_outlier = monitor_wrapper.update_history_result(i=i, monitor=monitor, prediction_i=prediction_i,
                                                           ground_truth_i=ground_truth_i, history=history_run_all)
        if is_outlier:
            outside_the_box.append(i)

            labels = ['label' + str(i) for i in range(max(data_run.classes) + 1)]
            warnings_full = [history_run_all.warnings(monitor=monitor, data=data_run)[len(outside_the_box)-1]]
            monitor_results = history_run_all.monitor2results[monitor.id()][i]
            # print(monitor_results.distance(), monitor_results.prediction(), monitor_results.suggestion())
            # plot_images(images=warnings_full, labels=labels, classes=data_run.classes,
            #            iswarning=True,
            #            monitor_id=monitor.id(),
            #
            #            c_suggested=[monitor_results.suggestion()])
    if plot:
        try:
            plot_2d_projection(history=history_run_all, monitor=monitor, layer=layer, all_classes=all_classes,
                               class_label_map=class_label_map,
                               category_title=model_name + plot_title,
                               dimensions=[0, 1], distance_thresholds=monitor_wrapper.score_thresholds,
                               additional_point_lists=[history_run_all.layer2values[layer][outside_the_box]],#[[history_run_all.layer2values[layer][i]]],
                               distances=[])#[monitor_results.distance()])
            plt.savefig('results/figure-projection_{}.png'.format(exp_name), dpi=300, bbox_inches='tight', transparent=True)
        except Exception as e:
            text = 'EXPERIMENT: {0} plot "results/figure-projection_{0}.png" could not be drawn -> {1}) \n'.format(exp_name, e)
            with open('results/error_{}.txt'.format(exp_name), 'a') as f:
                f.write(text)
            print("ERROR:", text)
    move_model(model_path)
    return model_path, res


if __name__ == "__main__":
    run_all_experiments()
