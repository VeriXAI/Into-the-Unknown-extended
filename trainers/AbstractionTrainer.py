import json

import keras.losses
import matplotlib.pyplot as plt
from .StandardTrainer import StandardTrainer
from utils import *
import sys


def get_dataset_partitions_tf(ds, ds_size, val_split=0.1):
    val_size = int(val_split * ds_size)
    val_ds = ds.take(val_size)
    train_ds = ds.skip(val_size)
    return train_ds, val_ds


def box_distance_loss(y_true, y_pred, radii):
    dist = tf.zeros(0, tf.float32)
    for p in tf.range(y_pred.shape[0]):
        dist_p = tf.constant(0.0)
        for i in tf.range(radii.shape[1]):
            dist_i = tf.abs(y_pred[p][i] - y_true[p][i])
            if radii[p][i] > 0:
                # normalization so that result 1.0 corresponds to dist == radius (i.e., point is on the border)
                dist_i /= radii[p][i]
            elif dist_i > tf.constant(0.0):
                # distance is defined as > 1 for flat dimensions unless point lies inside
                tf.add(dist_i, tf.constant(1.0))
            dist_p = tf.maximum(dist_p, dist_i)
        dist = tf.concat([dist, [dist_p]], -1)
    return dist


def scaling_factor(accuracy, accuracy_threshold, base):
    accuracy = max(accuracy, accuracy_threshold)
    x = (accuracy - accuracy_threshold) / (1 - accuracy_threshold)
    return 1 - base ** (-x)


class AbstractionTrainer(StandardTrainer):
    def __init__(self, monitor_manager, epochs_distance=1, accuracy_threshold=0.90, base=10,
                 experiment_name="EX", freeze=0.5, plot=False):
        super().__init__()
        self.monitor_manager = monitor_manager
        self.epochs_distance = epochs_distance
        self.class_label_map = None
        self.accuracy_threshold = accuracy_threshold
        self.base = base
        self.experiment_name = experiment_name
        self.freeze = freeze
        self.plot = plot

    def __str__(self):
        return "AbstractionTrainer"

    def set_class_label_map(self, class_label_map):
        self.class_label_map = class_label_map

    def train(self, model, data_train: DataSpec, data_test: DataSpec, epochs: int, batch_size: int):
        # first train normally
        beta = None
        training_data = []
        history1 = super().train(model=model, data_train=data_train, data_test=data_test, epochs=epochs,
                                 batch_size=batch_size)

        original_fn = model.loss
        old_loss_fn = keras.losses.deserialize(model.loss)
        data_train.shuffle()
        data_size = len(data_train.inputs())
        data_train_tf = tf.data.Dataset.from_tensor_slices((data_train.inputs(), data_train.categoricals()))
        data_train_tf, data_val_tf = get_dataset_partitions_tf(data_train_tf, data_size, 0.1)
        data_val_tf = data_val_tf.batch(batch_size)
        data_train_tf = data_train_tf.batch(batch_size)
        for epoch in range(self.epochs_distance):
            print("#" * 100)
            print('AbstractionTraining Epoch:', epoch)
            print("-" * 100)
            eval_results = model.evaluate(data_val_tf, batch_size=batch_size)
            alpha = scaling_factor(eval_results[1], self.accuracy_threshold, self.base)
            print('Accuracy & Scaling', eval_results, alpha)
            if alpha > 0:
                if epoch / self.epochs_distance <= self.freeze:
                    assert self.class_label_map is not None
                    n_classes_total = data_train.n_classes()
                    self.monitor_manager.normalize_and_initialize(model, class_label_map=self.class_label_map,
                                                                  n_classes_total=n_classes_total)
                    self.monitor_manager.train(model=model, data_train=data_train, data_test=None,
                                               statistics=Statistics())

                    '''plotting boxes'''

                    if self.plot:
                        try:
                            history = History()
                            history.set_ground_truths(data_train.ground_truths())
                            layer2values, _, _ = obtain_predictions(model=model, data=data_train,
                                                                    class_label_map=self.class_label_map,
                                                                    layers=[8])
                            if self.monitor_manager.layer2n_components is not None:
                                layer2values, layer2components = \
                                    reduce_dimension(layer2data=layer2values, layers=self.monitor_manager.layers(),
                                                     layer2n_components=self.monitor_manager.layer2n_components,
                                                     layer2components=self.monitor_manager.layer2components)
                            history.set_layer2values(layer2values)
                            plot_2d_projection(history=history, monitor=self.monitor_manager._monitors[0], layer=8,
                                               all_classes=[0, 1, 2],
                                               class_label_map=self.class_label_map,
                                               category_title=[],
                                               dimensions=[0, 1])
                            plt.savefig('results/figure-projection_{}_epoch{}.png'.format(self.experiment_name, epoch), dpi=300, bbox_inches='tight', transparent=True)
                        except Exception as e:
                            text = 'EXPERIMENT: {0} plot "results/figure-projection_{0}_epoch{1}.png" could not be drawn -> {2}) \n'.format(self.experiment_name,epoch,e)
                            with open('results/error_{}.txt'.format(self.experiment_name), 'a') as f:
                                f.write(text)
                            print("ERROR:", text)
                # NOTE: this code assumes we only watch a single layer
                layer_index = self.monitor_manager.layers()[0]
                layer_output = model.layers[layer_index].output
                model_until_layer = Model(inputs=model.input, outputs=layer_output)
                abstraction = self.monitor_manager.monitor().abstraction(layer_index)
                pca = self.monitor_manager.layer2components[layer_index]
                pca_mean = pca.mean_
                pca_components = pca.components_.T

            for step, (x_batch_train, y_batch_train) in enumerate(data_train_tf):
                if step > 2:
                    break
                # -- pre-compute box centers for current batch --
                # obtain layer values
                if alpha > 0:
                    y_batch_pred = model_until_layer(x_batch_train, training=False)

                    # project into principal components
                    y_batch_pred = tf.tensordot(y_batch_pred - pca_mean, pca_components, 1)

                    # compute centers
                    y_batch_train_classes = categoricals2numbers(
                        y_batch_train.numpy())  # TODO filter out misclassifications
                    centers = []
                    radii = []
                    for (i, ci) in enumerate(y_batch_train_classes):
                        boxes = abstraction._abstractions[ci]
                        point = y_batch_pred[i]
                        centers.append(boxes.closest_box_with_box_distance(point).center())
                        radii.append(boxes.closest_box_with_box_distance(point).radius())
                    centers = tf.convert_to_tensor(centers)
                    centers = tf.cast(centers, tf.float32)
                    radii = tf.convert_to_tensor(radii)
                    radii = tf.cast(radii, tf.float32)

                # -- compute old loss and gradients for current batch --
                with tf.GradientTape() as tape:
                    # -- compute old loss --
                    y_batch_pred = model(x_batch_train, training=True)
                    loss_value1 = old_loss_fn(y_true=y_batch_train, y_pred=y_batch_pred)

                    # compute distance to centers
                    if alpha > 0:
                        # -- compute distance loss and gradients for current batch --
                        # obtain watched layer's values
                        values = model_until_layer(x_batch_train, training=True)

                        # project into principal components
                        values = tf.tensordot(values - pca_mean, pca_components, 1)
                        distances = box_distance_loss(y_true=centers, y_pred=values, radii=radii)
                        loss_value2 = distances
                        alpha = scaling_factor(eval_results[1], self.accuracy_threshold, self.base)
                        if beta is None:
                            beta = np.mean(loss_value1) / np.mean(loss_value2)
                        loss_value = (1 - alpha) * loss_value1 + alpha * loss_value2 * beta
                    else:
                        loss_value2 = 0
                        loss_value = loss_value1
                # compute gradients for current batch
                grads = tape.gradient(loss_value, model.trainable_variables)
                # train weights for current batch
                model.optimizer.apply_gradients(zip(grads, model.trainable_variables))

                if step % 20 == 0:
                    print("Training loss (for one batch) at step %d: %.4f" % (step, float(sum(loss_value))))
                    print("Seen so far: %s samples" % ((step + 1) * batch_size))

                data_dict = {'loss_value': (1. * np.array(loss_value)).tolist(),
                             'loss_value1': (1. * np.array(loss_value1)).tolist(),
                             'loss_value2': (1. * np.array(loss_value2)).tolist(),
                             'alpha': 1. * alpha,
                             'beta': None if beta is None else 1. * beta,
                             'loss': float(eval_results[0]),
                             'accuracy': float(eval_results[1]),
                             'epoch': epoch,
                             'step': step
                             }
                training_data.append(data_dict)
                with open('results/training-data_{}.json'.format(self.experiment_name), 'w') as f:
                    json.dump(training_data, f)

        data_test_tf = tf.data.Dataset.from_tensor_slices((data_test.inputs(), data_test.categoricals()))
        data_test_tf = data_test_tf.batch(batch_size)
        print('Model Evaluation:', model.evaluate(data_test_tf, batch_size=batch_size))
        print("#" * 100)
        with open('results/training-data_{}.json'.format(self.experiment_name), 'w') as f:
            json.dump(training_data, f)
        return history1
