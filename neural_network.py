import numpy as np

in_layer = 5
hidden1 = 25
hidden2 = 10
out_layer = 1

weight1_shape = (25, 5)
weight2_shape = (10, 25)
weight3_shape = (1, 10)


# Auxiliary


def softmax(fz):
    return np.exp(fz.T) / np.sum(np.exp(fz.T), axis=1).reshape(-1, 1)


def weights(i):
    weight1 = i[0:weight1_shape[0] * weight1_shape[1]]
    weight2 = i[weight1_shape[0] * weight1_shape[1]:weight2_shape[0] * weight2_shape[1] + weight1_shape[0] * weight1_shape[1]]
    weight3 = i[weight2_shape[0] * weight2_shape[1] + weight1_shape[0] * weight1_shape[1]:]

    return \
        weight1.reshape(weight1_shape[0], weight1_shape[1]), \
        weight2.reshape(weight2_shape[0], weight2_shape[1]), \
        weight3.reshape(weight3_shape[0], weight3_shape[1])


# The Network - tanh is probably better as the negatives are mapped to strong negatives!


def forward_propagation(X, i):
    w1, w2, w3 = weights(i)

    fz1 = np.matmul(w1, X.T)
    activation1 = np.tanh(fz1)
    fz2 = np.matmul(w2, activation1)
    activation2 = np.tanh(fz2)
    fz3 = np.matmul(w3, activation2)
    activation3 = softmax(fz3)

    return activation3
