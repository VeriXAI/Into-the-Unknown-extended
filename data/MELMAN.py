import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.mlab as mlb

from utils import DataSpec, load_data, DATA_PATH, PLOT_ADDITIONAL_FEEDBACK

FILE_PATH = DATA_PATH + "MELMAN"
DATASET = 2
SAFE = 10
ROLLING = False  # True


def load_MELMAN(data_train_model: DataSpec, data_test_model: DataSpec, data_train_monitor: DataSpec,
                data_test_monitor: DataSpec, data_run: DataSpec, adversarial_data_suffix=None):
    # Select the dataset (used 2 by default)
    # Choose between (1) "speedup" or (2) "lowertlc"

    if DATASET == 1:
        file_manual = FILE_PATH + '/data_driving_melman2017_speedup_manual.csv'
        file_assistance = FILE_PATH + '/data_driving_melman2017_speedup_assistance.csv'
    elif DATASET == 2:
        file_manual = FILE_PATH + '/data_driving_melman2017_lowertlc_manual.csv'
        file_assistance = FILE_PATH + '/data_driving_melman2017_lowertlc_assistance.csv'

    # Read CSV data from the driving experiment
    data_manual = pd.read_csv(file_manual)
    data_assistance = pd.read_csv(file_assistance)

    # Test on implementing a rolling window on the dataset
    if ROLLING:
        rolling_frame = 100
        data_manual = data_manual.rolling(rolling_frame).mean()  # Window length
        data_assistance = data_assistance.rolling(rolling_frame).mean()  # Window length
    # Define the threshold of SAFE states and labeling the data
    # data_manual
    tlc_greater_zero_manual = data_manual.tlc[data_manual.tlc > 0]
    tlc_threshold_manual = np.percentile(tlc_greater_zero_manual, SAFE)

    tlc_greater_zero_assistance = data_assistance.tlc[data_assistance.tlc > 0]
    tlc_threshold_assistance = np.percentile(tlc_greater_zero_assistance, SAFE)

    if PLOT_ADDITIONAL_FEEDBACK:
        d_manual = np.sort(tlc_greater_zero_manual)
        d_assistance = np.sort(tlc_greater_zero_assistance)
        # Percentile values
        p = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
        perc_manual = np.percentile(d_manual, p)
        perc_assistance = np.percentile(d_assistance, p)
        plt.plot(d_manual)
        plt.plot(d_assistance)
        # Place red dots on the percentiles
        plt.plot((len(d_manual) - 1) * p / 100., perc_manual, 'go')
        plt.plot((len(d_assistance) - 1) * p / 100., perc_assistance, 'ro')
        # Set tick locations and labels
        plt.xticks((len(d_manual) - 1) * p / 100., map(str, p))
        plt.xticks((len(d_assistance) - 1) * p / 100., map(str, p))
        # plot thresholds
        n = max((len(d_manual) - 1) / 100., (len(d_assistance) - 1) / 100.)
        plt.hlines(y=tlc_threshold_manual, xmin=0.0, xmax=100.0 * n, colors='g', linestyles='dashed')
        plt.hlines(y=tlc_threshold_assistance, xmin=0.0, xmax=100.0 * n, colors='r', linestyles='dashed')
        plt.show()

    # data_assistance['safe_state_x_manual'] = data_assistance.tlc < tlc_threshold_manual
    # data_assistance['safe_state_x_assistance'] = data_assistance.tlc < tlc_threshold_assistance

    print('Threshold tlc for manual driving:', tlc_threshold_manual)
    print('Threshold tlc for assistance driving:', tlc_threshold_assistance)

    data_manual['safe_state'] = 1 * (data_manual.tlc < tlc_threshold_manual)
    data_assistance['safe_state'] = 1 * (data_assistance.tlc < tlc_threshold_manual) + \
                                    2 * (data_assistance.tlc < tlc_threshold_assistance) * \
                                    (data_assistance.tlc > tlc_threshold_manual)

    # Creating input features and targets
    upper_bound = -1
    if ROLLING:
        x = data_manual.iloc[rolling_frame:upper_bound,
            2:12]  # [2, 3, 5, 6, 7, 8, 9, 10, 11, 12]]# # Ignored "condition" and "roadpoint_index"
        y = data_manual.iloc[rolling_frame:upper_bound, 16]  # 13]
        x_validation = data_assistance.iloc[rolling_frame:upper_bound,
                       2:12]  # [2, 3, 5, 6, 7, 8, 9, 10, 11, 12]]#  # Ignored "condition" and "roadpoint_index"
        y_validation = data_assistance.iloc[rolling_frame:upper_bound, 16]  # 13]
    else:
        x = data_manual.iloc[:upper_bound, 2:12]  # Ignored "condition" and "roadpoint_index"
        y = data_manual.iloc[:upper_bound, 16]
        x_validation = data_assistance.iloc[:upper_bound, 2:12]  # Ignored "condition" and "roadpoint_index"
        y_validation = data_assistance.iloc[:upper_bound, 16]

    # Standardizing the input features
    sc = Normalizer()  # Standardize features by removing the mean and scaling to unit variance
    x = sc.fit_transform(x)  # Fit to data, then transform it.
    x_validation = sc.fit_transform(x_validation)  # Fit to data, then transform it.

    # Divide dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
    #x_train = x_test = x
    #y_train = y_test = y

    if adversarial_data_suffix is not None:
        with open(FILE_PATH + "/adversarial{}".format(adversarial_data_suffix), mode='rb') as file:
            adversarial = pickle.load(file, encoding='latin1')
        x_run = np.array(adversarial['data'])
        y_run = np.array(adversarial['labels'])
    else:
        x_run = x_validation
        y_run = y_validation

    data_train_model.set_data(inputs=np.reshape(x_train,
                                                (x_train.shape[0],
                                                 1, x_train.shape[1])),
                              labels=y_train, assertion=False)
    data_train_monitor.set_data(inputs=np.reshape(x_train,
                                                  (x_train.shape[0],
                                                   1, x_train.shape[1])),
                                labels=y_train, assertion=False)
    data_test_model.set_data(inputs=np.reshape(x_test,
                                               (x_test.shape[0],
                                                1, x_test.shape[1])),
                             labels=y_test, assertion=False)
    data_test_monitor.set_data(inputs=np.reshape(x_test,
                                                 (x_test.shape[0],
                                                  1, x_test.shape[1])),
                               labels=y_test, assertion=False)
    data_run.set_data(inputs=np.reshape(x_run,
                                        (x_run.shape[0],
                                         1, x_run.shape[1])),
                      labels=y_run, assertion=False)

    pixel_depth = 255.0
    class_label_map, all_labels = load_data(
        data_train_model=data_train_model, data_test_model=data_test_model, data_train_monitor=data_train_monitor,
        data_test_monitor=data_test_monitor, data_run=data_run,  # pixel_depth=pixel_depth,
        is_adversarial_data=(adversarial_data_suffix is not None))

    return class_label_map, all_labels
