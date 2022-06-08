import math

from run.experiment_helper import *
from run import train_MNIST_distance, train_MNIST, train_F_MNIST_distance, train_F_MNIST, train_GTSRB_distance, train_GTSRB


def experiment_generator(ex_name, instances, trainer, known_classes,all_classes,interesting_classes,epochs,accuracy_threshold,base,batch_size,freeze):
    def experiment():
        name = "{}_e{}_t{}_b{}_bs{}_f{}_{}-{}".format(ex_name, epochs, accuracy_threshold, base, batch_size, freeze,''.join([str(i) for i in known_classes]),
                                      ''.join([str(i) for i in all_classes if i not in known_classes]))
        return [name] + [[trainer[i].run_script, instances[i], known_classes, all_classes, interesting_classes, epochs, accuracy_threshold,base,batch_size,freeze] for i in range(2)]
    return experiment


def GTSRB_experiment_generator(known_classes,all_classes,interesting_classes,epochs,accuracy_threshold,base,batch_size,freeze):
    return experiment_generator("GTSRB",[instance_GTSRB, instance_AT_GTSRB], [train_GTSRB_distance, train_GTSRB],
                                known_classes,all_classes,interesting_classes,epochs,accuracy_threshold,base,batch_size,freeze)


def F_MNIST_experiment_generator(known_classes,all_classes,interesting_classes,epochs,accuracy_threshold,base,batch_size,freeze):
    return experiment_generator("FMNIST", [instance_F_MNIST, instance_AT_F_MNIST],  [train_F_MNIST_distance, train_F_MNIST],
                                known_classes,all_classes,interesting_classes,epochs,accuracy_threshold,base,batch_size,freeze)


def MNIST_experiment_generator(known_classes,all_classes,interesting_classes,epochs,accuracy_threshold,base,batch_size,freeze):
    return experiment_generator("MNIST", [instance_AT_MNIST, instance_MNIST],  [train_MNIST_distance, train_MNIST],
                                known_classes,all_classes,interesting_classes,epochs,accuracy_threshold,base,batch_size,freeze)


def GTSRB_experiments():
    accuracy_threshold = [0.9]
    base = [10]
    epochs = 16
    batch_size = [128]
    freeze = [0.5]
    known_classes = [[0,1], list(range(22)),list(range(42))]
    all_classes = [[0,1,2],list(range(23)), list(range(43))]
    experiments = []
    for kc in known_classes:
        for ac in all_classes:
            if len(kc)<len(ac):
                for t in accuracy_threshold:
                    for b in base:
                        for bs in batch_size:
                            for f in freeze:
                                experiments.append(GTSRB_experiment_generator(kc, ac, ac, epochs, t, b, bs,f))
    return experiments


def F_MNIST_experiments():
    accuracy_threshold = [0.9]
    base = [10]
    epochs = 16
    batch_size = [128]
    freeze = [0.5]
    known_classes = [[0,1], [0,1,2,3,4], [0,1,2,3,4,5,6,7,8]]
    all_classes = [[0,1,2],[0,1,2,3,4,5], [0,1,2,3,4,5,6,7,8,9]]
    experiments = []
    for kc in known_classes:
        for ac in all_classes:
            if len(kc)<len(ac):
                for t in accuracy_threshold:
                    for b in base:
                        for bs in batch_size:
                            for f in freeze:
                                experiments.append(F_MNIST_experiment_generator(kc, ac, ac, epochs, t, b, bs,f))
    return experiments


def MNIST_experiments():
    accuracy_threshold = [0.9]
    base = [10]
    epochs = 16
    batch_size = [128]
    freeze = [0.5]
    known_classes = [[0,1], [0,1,2,3,4], [0,1,2,3,4,5,6,7,8]]
    all_classes = [[0,1,2],[0,1,2,3,4,5], [0,1,2,3,4,5,6,7,8,9]]
    experiments = []
    for kc in known_classes:
        for ac in all_classes:
            if len(kc)<len(ac):
                for t in accuracy_threshold:
                    for b in base:
                        for bs in batch_size:
                            for f in freeze:
                                experiments.append(MNIST_experiment_generator(kc, ac, ac, epochs, t, b, bs,f))
    return experiments

