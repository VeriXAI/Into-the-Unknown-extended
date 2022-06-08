from data import *
from run.experiment_helper import *
from trainers import *


def run_script():
    model_name, data_name, stored_network_name, total_classes = instance_MNIST(transfer=False)#instance_Doom()
    known_classes = [0, 1, 2]
    all_classes = [0, 1, 2, 3]
    interesting_classes = [1]
    classes_string = classes2string(known_classes)
    all_classes_string = classes2string(all_classes)
    trans_model_path = "{}_trans_{}.h5".format(stored_network_name, all_classes_string)
    model_path = "{}_{}.h5".format(stored_network_name, classes_string)
    data_train_model = DataSpec(randomize=False, classes=known_classes) #known_classes
    data_test_model = DataSpec(randomize=False, classes=known_classes)
    data_train_monitor = DataSpec(randomize=False, classes=known_classes)
    data_test_monitor = DataSpec(randomize=False, classes=known_classes)
    data_run = DataSpec(randomize=False, classes=all_classes)#all_classes)
    layer = -2
    layer2n_components = {layer: 4}  # dimension reduction; use 'None' to deactivate

    class_label_map, all_labels = get_data_loader(data_name)(
        data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
        data_test_monitor=data_test_monitor, data_run=data_run)
    model, _ = get_model(model_name=model_name, model_path=model_path, data_train=data_train_model,
                         data_test=data_test_model, class_label_map=class_label_map, model_trainer=StandardTrainer(),
                         n_epochs=10, batch_size=10, statistics=Statistics())
    trans_model, _ = get_model(model_name=model_name, model_path=trans_model_path, data_train=data_train_model,
                               data_test=data_test_model, class_label_map=class_label_map,
                               model_trainer=StandardTrainer(), n_epochs=10, batch_size=10, statistics=Statistics())
    print_data_information(data_train_monitor, data_test_monitor, data_run)

    # create monitor
    layer2abstraction = {layer: BoxAbstraction(euclidean_mean_distance)}
    monitor = Monitor(layer2abstraction=layer2abstraction)
    monitor_manager = MonitorManager([monitor], clustering_threshold=0.07, layer2n_components=layer2n_components)

    # run instance
    monitor_manager.normalize_and_initialize(model, class_label_map=class_label_map, n_classes_total=total_classes)
    monitor_manager.train(model=model, data_train=data_train_monitor, data_test=data_test_monitor,
                          statistics=Statistics())#, ignore_misclassifications=False)

    trans_monitor = Monitor(layer2abstraction=layer2abstraction)
    trans_monitor_manager = MonitorManager([trans_monitor], clustering_threshold=0.07,
                                           layer2n_components=layer2n_components)

    # run instance
    trans_monitor_manager.normalize_and_initialize(trans_model,
                                                   class_label_map=class_label_map,
                                                   n_classes_total=total_classes)
    trans_monitor_manager.train(model=trans_model, data_train=data_train_monitor, data_test=data_test_monitor,
                                statistics=Statistics())  # , ignore_misclassifications=False)

    dimensions = [0, 1]#[3, 13]

    # training data
    history = History()
    history.set_ground_truths(data_train_monitor.ground_truths())
    layer2values, _, _ = obtain_predictions(model=model, data=data_train_monitor, class_label_map=class_label_map,
                                            layers=[layer])
    trans_layer2values, _, _ = obtain_predictions(model=trans_model, data=data_train_monitor,
                                                  class_label_map=class_label_map, layers=[layer])
    if layer2n_components is not None:
        layer2values, layer2components = \
            reduce_dimension(layer2data=layer2values, layers=monitor_manager.layers(),
                             layer2n_components=monitor_manager.layer2n_components,
                             layer2components=monitor_manager.layer2components)
        trans_layer2values, trans_layer2components = \
            reduce_dimension(layer2data=trans_layer2values, layers=trans_monitor_manager.layers(),
                             layer2n_components=trans_monitor_manager.layer2n_components,
                             layer2components=trans_monitor_manager.layer2components)
        plot_title = " training with reduced data"
    else:
        plot_title = " training with original data"
    history.set_layer2values(layer2values)
    plot_2d_projection(history=history, monitor=monitor, layer=layer, all_classes=interesting_classes,
                       class_label_map=class_label_map,
                       category_title=model_name + plot_title,
                       dimensions=dimensions, additional_point_lists=[trans_layer2values[layer]])

    # running data
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
    plot_2d_projection(history=history_run, monitor=monitor, layer=layer, all_classes=interesting_classes,
                       class_label_map=class_label_map,
                       category_title=model_name + plot_title,
                       dimensions=dimensions)

    plt.show()
    save_all_figures()


if __name__ == "__main__":
    run_script()
