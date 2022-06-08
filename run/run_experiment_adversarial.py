from run.experiment_helper import *


def run_experiment_adversarial():
    # global options
    seed = 0
    alphas = [0.1, 0.01]
    do_reduce_dimension = True
    logger = Logger.start("log_adversarial.txt")

    # instance options
    instances = [
        # entries: instance constructor, box_abstraction, flag for using Boolean abstraction, epsilon,
        # clustering threshold, class count
        #(instance_MNIST, box_abstraction_MNIST, False, 0.0, 0.07, 9),
        #(instance_F_MNIST, box_abstraction_F_MNIST, False, 0.0, 0.07, 9),
        #(instance_CIFAR10, box_abstraction_CIFAR10, False, 0.0, 0.07, 9),
        #(instance_GTSRB, box_abstraction_GTSRB, False, 0.0, 0.3, 20),
        (instance_Doom, box_abstraction_Doom, False, 0.0, 0.07, 6)
    ]

    for instance_function, box_abstraction, use_boolean_abstraction, epsilon, clustering_threshold, n_classes in \
            instances:
        model_name, data_name, stored_network_name, total_classes = instance_function()
        if use_boolean_abstraction:
            storage_monitors = [[], [], []]
        else:
            storage_monitors = [[], []]
        storage_at = [[] for _ in alphas]

        print("\n--- new instance ---\n")

        adversarial_data_suffix = "_0-{:d}".format(n_classes - 1)

        # load instance
        data_train_model, data_test_model, data_train_monitor, data_test_monitor, data_run, model_path, _ = \
            load_instance(n_classes, total_classes, stored_network_name)

        # create (fresh) monitors
        monitor1 = box_abstraction(epsilon)
        monitors = [monitor1]
        if use_boolean_abstraction:
            layer2abstraction = {-2: BooleanAbstraction()}
            monitor2 = Monitor(layer2abstraction=layer2abstraction)
            monitors.append(monitor2)
        layer2n_components = dict() if do_reduce_dimension else None
        monitor_manager = MonitorManager(monitors, clustering_threshold=clustering_threshold,
                                         layer2n_components=layer2n_components)

        # run instance
        history_run, histories_alpha_thresholding, novelty_wrapper_run, novelty_wrappers_alpha_thresholding, \
            statistics = evaluate_all(seed=seed, data_name=data_name, data_train_model=data_train_model,
                                      data_test_model=data_test_model, data_train_monitor=data_train_monitor,
                                      data_test_monitor=data_test_monitor, data_run=data_run, model_name=model_name,
                                      model_path=model_path, monitor_manager=monitor_manager, alphas=alphas,
                                      adversarial_data_suffix=adversarial_data_suffix)

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
        filename_prefix = "adversarial_" + data_name
        store_core_statistics(storage_monitors[0], "monitor1", filename_prefix=filename_prefix)
        if use_boolean_abstraction:
            store_core_statistics(storage_monitors[1], "monitor2", filename_prefix=filename_prefix)
        store_core_statistics(storage_at, alphas, filename_prefix=filename_prefix)

    # close log
    logger.stop()


def plot_experiment_adversarial():
    # global options
    alphas = [0.1, 0.01]

    # instance options
    instances = [
        #("MNIST", False),
        #("F_MNIST", False),
        #("CIFAR10", False),
        #("GTSRB", False),
        ("Doom", False)
    ]

    for data_name, use_boolean_abstraction in instances:
        filename_prefix = "adversarial_" + data_name
        storage_all = load_core_statistics(alphas, filename_prefix=filename_prefix)
        if use_boolean_abstraction:
            storage_all.append(load_core_statistics("monitor2", filename_prefix=filename_prefix))
        storage_all.append(load_core_statistics("monitor1", filename_prefix=filename_prefix))

        plot_false_decisions_given_all_lists(storage_all, n_ticks=1, name=filename_prefix)

    save_all_figures(close=True)


def run_experiment_adversarial_all():
    run_experiment_adversarial()
    plot_experiment_adversarial()


if __name__ == "__main__":
    run_experiment_adversarial_all()
