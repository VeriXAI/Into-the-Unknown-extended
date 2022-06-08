from run.experiment_helper import *
from utils import plot_and_store_rejections


def run_experiment_novelty_variation():
    # global options
    seed = 0
    alphas = [0.1, 0.01]
    logger = Logger.start("log_novelty_variation.txt")
    do_reduce_dimension = False#True

    # instance options
    instances = [
        # entries: instance constructor, box_abstraction, flag for using Boolean abstraction, epsilon,
        # clustering threshold, class count
        (instance_MNIST, box_abstraction_MNIST, False, .0, 0.07, 9),
        #(instance_F_MNIST, box_abstraction_F_MNIST, False, .0, 0.07, -1),
        #(instance_CIFAR10, box_abstraction_CIFAR10, False, .0, 0.3, -1),
        #(instance_GTSRB, box_abstraction_GTSRB, False, .0, 0.3, -1),
        #(instance_Doom, box_abstraction_Doom, False, .0, 0.07, 3),
        #(instance_MELMAN, box_abstraction_MELMAN, False, .0, 0.07, 2)
    ]

    for instance_function, monitor_constructor, use_boolean_abstraction, epsilon, clustering_threshold,\
            n_classes_max in instances:
        model_name, data_name, stored_network_name, total_classes, _, _ = instance_function()
        if use_boolean_abstraction:
            storage_monitors = [[], [], []]
        else:
            storage_monitors = [[], []]
        storage_at = [[] for _ in alphas]
        if n_classes_max == -1:
            n_classes_max = total_classes  # total_classes are the number of classes the network sees during runtime
        else:
            n_classes_max += 1
        for n_classes in range(2, n_classes_max): #n_classes_max-1
            print("\n--- new instance ---\n")
            # load instance
            data_train_model, data_test_model, data_train_monitor, data_test_monitor, data_run, model_path,\
                transfer_model_path, _ = \
                load_instance(n_classes, total_classes, stored_network_name)

            # create (fresh) monitors
            monitor1 = monitor_constructor(epsilon)
            # monitor2 = box_abstraction(epsilon, learn_from_test_data=False)
            monitors = [monitor1]  # , monitor2]
            if use_boolean_abstraction:
                layer2abstraction = {-2: BooleanAbstraction()}
                monitor2 = Monitor(layer2abstraction=layer2abstraction)
                monitors.append(monitor2)
            layer2n_components = dict() if do_reduce_dimension else None
            monitor_manager = MonitorManager(monitors, clustering_threshold=clustering_threshold,
                                             layer2n_components=layer2n_components,
                                             fit_distribution_method=None)

            # run instance
            history_run, histories_alpha_thresholding, novelty_wrapper_run, novelty_wrappers_alpha_thresholding, \
                statistics = evaluate_all(seed=seed, data_name=data_name, data_train_model=data_train_model,
                                          data_test_model=data_test_model, data_train_monitor=data_train_monitor,
                                          data_test_monitor=data_test_monitor, data_run=data_run, model_name=model_name,
                                          model_path=model_path, monitor_manager=monitor_manager, alphas=alphas)

            # plot rejected inputs
            if REPORT_REJECTIONS:
                layer = 1
                class_label_map = class_label_map_from_labels([0, 1])
                all_classes = [0, 1, 2]
                dimensions = [0, 1]
                confidence_threshold = GMM_CONFIDENCE_TEST
                group_consecutive = 20
                # column names if inputs shall be plotted
                row_header = ["road_curvature", "velocity", "tlc", "lat_error", "lat_accel", "steering_wheel_angle",
                              "steering_wheel_angle_rate", "driver_wheel_torque", "throttle_deflection",
                              "throttle_deflection_rate"]
                plot_and_store_rejections(model_name=model_name, model_path=model_path, data_name=data_name,
                                          data_run=data_run, history_run=history_run, monitor_manager=monitor_manager,
                                          layer=layer, class_label_map=class_label_map, all_classes=all_classes,
                                          dimensions=dimensions, confidence_threshold=confidence_threshold,
                                          group_consecutive=group_consecutive)

            # print/store statistics
            print_general_statistics(statistics, data_train_monitor, data_run)
            print_and_store_monitor_statistics(storage_monitors, monitors, statistics, history_run,
                                               novelty_wrapper_run, data_train_monitor, data_run)

            # store statistics for alpha thresholding
            for i, (history_alpha, novelty_wrapper_alpha, alpha) in enumerate(zip(
                    histories_alpha_thresholding, novelty_wrappers_alpha_thresholding, alphas)):
                history_alpha.update_statistics(0, confidence_threshold=alpha)
                fn = history_alpha.false_negatives()
                fp = history_alpha.false_positives()
                tp = history_alpha.true_positives()
                tn = history_alpha.true_negatives()
                novelty_results = novelty_wrapper_alpha.evaluate_detection(0, confidence_threshold=alpha)
                storage = CoreStatistics(fn=fn, fp=fp, tp=tp, tn=tn,
                                         novelties_detected=len(novelty_results["detected"]),
                                         novelties_undetected=len(novelty_results["undetected"]))
                storage_at[i].append(storage)

        # store results
        filename_prefix = "novelty_" + data_name
        store_core_statistics(storage_monitors[0], "monitor1", filename_prefix=filename_prefix)
        # store_core_statistics(storage_monitors[1], "monitor2", filename_prefix=filename_prefix)
        if use_boolean_abstraction:
            store_core_statistics(storage_monitors[1], "monitor2", filename_prefix=filename_prefix)
        store_core_statistics(storage_at, alphas, filename_prefix=filename_prefix)

    # close log
    logger.stop()


def plot_experiment_novelty_variation():
    # global options
    alphas = [0.1, 0.01]

    # instance options
    instances = [
        ("MNIST", False, 8, 8),
        #("F_MNIST", False, 8, None),
        #("CIFAR10", False, 8, None),
        #("GTSRB", False, 41, None),
        #("Doom", False, 3, None),
        #("MELMAN", False, 3, None)
    ]

    for data_name, use_boolean_abstraction, n_ticks, n_bars in instances:
        filename_prefix = "novelty_" + data_name
        storage_all = load_core_statistics(alphas, filename_prefix=filename_prefix)
        if use_boolean_abstraction:
            storage_all.append(load_core_statistics("monitor2", filename_prefix=filename_prefix))
        storage_all.append(load_core_statistics("monitor1", filename_prefix=filename_prefix))
        # storage_all.append(load_core_statistics("monitor2", filename_prefix=filename_prefix))

        plot_false_decisions_given_all_lists(storage_all, n_ticks=n_ticks, name=filename_prefix, n_bars=n_bars)

    plt.show()
    save_all_figures(close=True)


def run_experiment_novelty_variation_all():
    run_experiment_novelty_variation()
    plot_experiment_novelty_variation()


if __name__ == "__main__":
    run_experiment_novelty_variation_all()
