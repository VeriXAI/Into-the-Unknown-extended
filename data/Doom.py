import h5py
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

from utils import DataSpec, load_data, DATA_PATH

FILE_PATH = DATA_PATH + "Doom"


def load_Doom(data_train_model: DataSpec, data_test_model: DataSpec, data_train_monitor: DataSpec,
               data_test_monitor: DataSpec, data_run: DataSpec, adversarial_data_suffix=None):
    # load data from basic Doom
    with h5py.File(FILE_PATH + "/data_basic.h5", "r") as f:
        data = {}
        # List all groups
        print("Keys: %s" % f.keys())
        for a_group_key in f.keys():
            # Get the data
            data[a_group_key] = list(f[a_group_key])
    # combine with the data from corridor Doom
    with h5py.File(FILE_PATH + "/data_corridor.h5", "r") as f:
        data_novel = {}
        # List all groups
        print("Keys: %s" % f.keys())
        for a_group_key in f.keys():
            # Get the data
            data_novel[a_group_key] = list(f[a_group_key])
    # filter out the actions 0,1,2 from corridor Doom
    # as basic Doom is trained for 0,1,2 but they are different from corridor Doom
    for ind in range(0, len(data_novel['actions'])):
        if data_novel['actions'][ind] not in [0, 1, 2]:
            for k in data.keys():
                data[k] += [data_novel[k][ind]]

    x_train, x_test, y_train, y_test = train_test_split(np.array(data['states']), np.array(data['actions']),
                                                        test_size=0.3, random_state=35)

    if adversarial_data_suffix is not None:
        with open(FILE_PATH + "/adversarial{}".format(adversarial_data_suffix), mode='rb') as file:
            adversarial = pickle.load(file, encoding='latin1')
        x_run = np.array(adversarial['data'])
        y_run = np.array(adversarial['labels'])
    else:
        x_run = x_test
        y_run = y_test

    data_train_model.set_data(inputs=x_train, labels=y_train, assertion=False)
    data_train_monitor.set_data(inputs=x_train, labels=y_train, assertion=False)
    data_test_model.set_data(inputs=x_test, labels=y_test, assertion=False)
    data_test_monitor.set_data(inputs=x_test, labels=y_test, assertion=False)
    data_run.set_data(inputs=x_run, labels=y_run, assertion=False)
    pixel_depth = 255.0
    class_label_map, all_labels = load_data(
        data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
        data_test_monitor=data_test_monitor, data_run=data_run, pixel_depth=pixel_depth,
        is_adversarial_data=(adversarial_data_suffix is not None))

    return class_label_map, all_labels
