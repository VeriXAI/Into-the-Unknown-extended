import numpy as np

from utils import DataSpec, load_data


def load_ToyData(data_train_model: DataSpec, data_test_model: DataSpec, data_train_monitor: DataSpec,
                 data_test_monitor: DataSpec, data_run: DataSpec):
    # add data
    x = np.array([[0.7, 0.2], [0.6, 0.2], [0.7, 0.1], [0.8, 0.1],  # class 1, first cluster
                  [0.9, 0.2],  # class 1, second cluster
                  [0.5, 0.5], [0.5, 0.6], [0.4, 0.6],  # class 2, first cluster
                  [0.2, 0.7]  # class 2, second cluster
                  ])
    y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])
    all_labels = [0, 1]
    data_train_model.set_data(inputs=x, labels=y, assertion=False)
    data_test_model.set_data(inputs=x, labels=y, assertion=False)
    data_train_monitor.set_data(inputs=x, labels=y, assertion=False)
    data_test_monitor.set_data(inputs=x, labels=y, assertion=False)
    data_run.set_data(inputs=x, labels=y, assertion=False)

    pixel_depth = None

    class_label_map = load_data(
        data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
        data_test_monitor=data_test_monitor, data_run=data_run, pixel_depth=pixel_depth)

    return class_label_map, all_labels
