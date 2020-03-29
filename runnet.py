from book2_neural_network import Network

import numpy as np

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

tr_data_all = np.loadtxt(open("mnist_train.csv", 'r'), delimiter=",", skiprows=0)
tr_inputs = [np.reshape(x, (784, 1)) for x in tr_data_all[:, 1:785]]
tr_results = [vectorized_result(int(y)) for y in tr_data_all[:, 0]]

tr_data = list(zip(tr_inputs, tr_results))
tr_d_data = list(zip(tr_inputs, tr_data_all[:, 0]))

training_data = tr_data[0:5000]

validation_data = tr_d_data[5001:6001]

test_data_all = np.loadtxt(open("mnist_test.csv", 'r'), delimiter=",", skiprows=0)
test_inputs = [np.reshape(x, (784, 1)) for x in test_data_all[:, 1:785]]
test_data = list(zip(test_inputs, test_data_all[:, 0]))

net = Network([784, 30, 10])
# net.SGD(training_data, 30, 10, 0.015, test_data=test_data)
net.SGD(training_data, 50, 10, 0.001, lmbda=15.0, evaluation_data=validation_data, monitor_evalution_accuracy=True)