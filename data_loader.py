import pandas as pd
import numpy as np


def load_data():
    tr_d = pd.read_csv('', header=None)
    te_d = pd.read_csv('', header=None)

    return np.array(tr_d), np.array(te_d)


def load_data_wrapper():
    tr_d, te_d = load_data()

    training_inputs = [np.reshape(x[:-1], (300, 1)) for x in tr_d]
    training_inputs = np.array(training_inputs)
    training_results = [vectorized_result(y[-1]) for y in tr_d]
    training_results = np.array(training_results)
    training_data = list(zip(training_inputs, training_results))

    test_inputs = [np.reshape(x[:-1], (300, 1)) for x in te_d]
    test_inputs = np.array(test_inputs)
    test_results = [vectorized_result(y[-1]) for y in te_d]
    test_results = np.array(test_results)
    test_data = list(zip(test_inputs, test_results))

    return training_data, test_data


def vectorized_result(j):
    e = np.zeros((2, 1))
    if j == 1:
        e[1] = 1
    else:
        e[0] = 1
    return e
