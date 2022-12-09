import pandas as pd
import numpy as np
import pickle


def load_data():
    tr_d = pickle.load(open('training_data.pkl', 'rb'))
    te_d = pickle.load(open('testing_data.pkl', 'rb'))

    return np.array(tr_d), np.array(te_d)


def load_data_wrapper():
    tr_d, te_d = load_data()

    training_inputs = [np.reshape(x[:-1], (300, 1)) for x in tr_d]
    training_results = [vectorized_result(y[-1]) for y in tr_d]
    training_data = list(zip(training_inputs, training_results))

    test_inputs = [np.reshape(x[:-1], (300, 1)) for x in te_d]
    test_results = [vectorized_result(y[-1]) for y in te_d]
    test_data = list(zip(test_inputs, test_results))

    return training_data, test_data


def vectorized_result(j):
    e = np.zeros((2, 1))
    if j == 1:
        e[1] = 1
    else:
        e[0] = 1
    return e
