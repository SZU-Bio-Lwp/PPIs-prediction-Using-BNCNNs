import tensorflow._api.v2.compat.v1 as tf

tf.disable_v2_behavior()

import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from model import CNNs_Model

import utils
import data_loader


def specificity_score(y_true, pre_label):
    confusion = confusion_matrix(y_true, pre_label)
    return confusion[0][0] / (confusion[0][0] + confusion[0][1])


def train_and_eval(model):
    kf = KFold(n_splits=10)
    train_data, test_data = data_loader.load_data_wrapper()
    train_data = shuffle(train_data)

    for train_index, test_index in kf.split(train_data):
        metrics = []
        train_x, train_y = utils.td_to_2d(train_index, train_data)
        test_x, test_y = utils.td_to_2d(test_index, train_data)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            for i in range(2000):
                x_batch, y_batch = utils.random_batch(train_x, train_y, model.batch_size)
                x_batch = x_batch.astype(np.float32)
                y_batch = y_batch.astype(np.float32)
                _, cross = sess.run([model.train_step, model.cross_entropy],
                                    feed_dict={model.x: x_batch, model.y: y_batch, model.drop_prob: 0.5})

            test_prediction_value = sess.run(model.pred, feed_dict={model.x: test_x, model.drop_prob: 0})
            pre_label = np.argmax(test_prediction_value, axis=1)

            y_true = utils.one_hot_to_label(test_y)

        metrics.append(round(accuracy_score(y_true, pre_label), 3))
        metrics.append(round(recall_score(y_true, pre_label), 3))
        metrics.append(round(specificity_score(y_true, pre_label), 3))
        metrics.append(round(precision_score(y_true, pre_label, zero_division=0), 3))
        metrics.append(round(f1_score(y_true, pre_label), 3))
        metrics.append(round(matthews_corrcoef(y_true, pre_label), 3))

        print("Acc:", metrics[0], "  Se:", metrics[1], "  Sp:", metrics[2], "  Pr:", metrics[3], "  F-measure:",
              metrics[4], "  MCC:", metrics[5])


if __name__ == "__main__":
    model = CNNs_Model()
    train_and_eval(model)
