from data import *
from monitoring.online import DistanceMonitor, box_distance_parameter, MonitorWrapper
from run.experiment_helper import *
from trainers import *


def run_script():
    model_name, data_name, stored_network_name, total_classes, flatten_layer, optimizer = instance_AT_MNIST(transfer=False)
    print(model_name, data_name, stored_network_name, total_classes, flatten_layer, optimizer)
    known_classes = [0, 1]
    all_classes = [0, 1, 2]
    interesting_classes = [0, 1, 2]  # [2, 3, 9]
    classes_string = classes2string(known_classes)
    all_classes_string = classes2string(all_classes)
    if "trans" in stored_network_name:
        model_path = "{}_{}.h5".format(stored_network_name, all_classes_string)
    else:
        model_path = "{}_{}.h5".format(stored_network_name, classes_string)
    data_train_model = DataSpec(randomize=False, classes=known_classes)
    data_test_model = DataSpec(randomize=False, classes=known_classes)
    data_train_monitor = DataSpec(randomize=False, classes=known_classes)
    data_test_monitor = DataSpec(randomize=False, classes=known_classes)
    data_run = DataSpec(randomize=False, classes=interesting_classes)
    layer = 8
    layer2n_components = None  #{layer: 10}  # dimension reduction; use 'None' to deactivate

    class_label_map, all_labels = get_data_loader(data_name)(
        data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
        data_test_monitor=data_test_monitor, data_run=data_run)
    model, _ = get_model(model_name=model_name, model_path=model_path, data_train=data_train_model,
                         data_test=data_test_model, class_label_map=class_label_map, model_trainer=StandardTrainer(),
                         n_epochs=10, batch_size=128, statistics=Statistics())
    print_data_information(data_train_monitor, data_test_monitor, data_run)

    # create monitor
    layer2abstraction = {layer: BoxAbstraction(euclidean_mean_distance)}
    # monitor = Monitor(layer2abstraction=layer2abstraction)
    monitor = Monitor(layer2abstraction=layer2abstraction)
    # monitor = DistanceMonitor(monitor, box_distance_parameter)
    monitor_manager = MonitorManager([monitor], clustering_threshold=0.3,
                                     layer2n_components=layer2n_components,
                                     fit_distribution_method=None)
    monitor_wrapper = MonitorWrapper(monitor_manager=monitor_manager)
    monitor_wrapper.score_thresholds = {0: 1.0, 1: 1.0, 2: 1.0}  # , 3: 1.0, 4: 1.0, 5: 1.0}

    # run instance
    monitor_manager.normalize_and_initialize(model, class_label_map=class_label_map, n_classes_total=total_classes)
    monitor_manager.train(model=model, data_train=data_train_monitor, data_test=data_test_monitor,
                          statistics=Statistics())  # , ignore_misclassifications=False)

    dimensions = [0, 1]

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
    plot_2d_projection(history=history, monitor=monitor, layer=layer, all_classes=all_classes,
                       class_label_map=class_label_map,
                       category_title=model_name + plot_title,
                       dimensions=dimensions, distance_thresholds=monitor_wrapper.score_thresholds)

    # running data
    '''
    history_run = History()
    history_run.set_ground_truths(data_run.ground_truths())
    layer2values_run, _, _ = obtain_predictions(model=model, data=data_run, class_label_map=class_label_map,
                                                layers=[layer])

    if layer2n_components is not None:
        layer2values_run, layer2components = \
            reduce_dimension(layer2data=layer2values_run, layers=monitor_manager.layers(),
                             layer2n_components=layer2n_components,
                             layer2components=monitor_manager.layer2components)
        plot_title = " run with reduced data"
    else:
        plot_title = " run with original data"
    history_run.set_layer2values(layer2values_run)
    '''

    history_run_all = monitor_manager.run(model=model, data=data_run, statistics=Statistics())

    history_run_all.update_statistics(monitor.id())
    print("_fn =", history_run_all._fn)
    print("_fp =", history_run_all._fp)
    print("_tn =", history_run_all._tn)
    print("_tp =", history_run_all._tp)

    outside_the_box = []
    for i, (prediction_i, ground_truth_i) in enumerate(zip(history_run_all.predictions, history_run_all.ground_truths)):
        is_outlier = monitor_wrapper.update_history_result(i=i, monitor=monitor, prediction_i=prediction_i,
                                                           ground_truth_i=ground_truth_i, history=history_run_all)
        if is_outlier:
            outside_the_box.append(i)

            #labels = ['label' + str(i) for i in range(max(data_run.classes) + 1)]
            #warnings_full = [history_run_all.warnings(monitor=monitor, data=data_run)[len(outside_the_box)-1]]
            #monitor_results = history_run_all.monitor2results[monitor.id()][i]
            # print(monitor_results.distance(), monitor_results.prediction(), monitor_results.suggestion())
            # plot_images(images=warnings_full, labels=labels, classes=data_run.classes,
            #            iswarning=True,
            #            monitor_id=monitor.id(),
            #            c_suggested=[monitor_results.suggestion()])
    plot_2d_projection(history=history_run_all, monitor=monitor, layer=layer, all_classes=all_classes,
                       class_label_map=class_label_map,
                       category_title=model_name + plot_title,
                       dimensions=dimensions, distance_thresholds=monitor_wrapper.score_thresholds,
                       additional_point_lists=[history_run_all.layer2values[layer][outside_the_box]],#[[history_run_all.layer2values[layer][i]]],
                       distances=[])#[monitor_results.distance()])

    plt.show()
    #plt.savefig("detection_example.png", dpi=300, bbox_inches='tight', transparent=True)
    # save_all_figures()



if __name__ == "__main__":
    run_script()
