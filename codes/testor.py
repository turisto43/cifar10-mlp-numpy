import numpy as np
from codes.activation_functions import relu, sigmoid, softmax


# 从x计算到y
def predict(x_data, nn, activation_function):
    h = x_data
    for i in range(len(nn)):
        h = h.dot(nn[i])
        if activation_function[i] == "relu":
            h = relu(h)
        elif activation_function[i] == "sigmoid":
            h = sigmoid(h)
        elif activation_function[i] == "softmax":
            h = softmax(h)
    return h


def accuracy(y_pred, y_true):
    # 将输出概率转化为类别预测
    y_pred_class = np.argmax(y_pred, axis=1)
    return np.mean(y_pred_class == y_true)


def test_my_model(nn, activation_function_settings, x_test_batches, y_test_batches):
    y_pred_all = []
    y_true_all = []
    for i in range(len(x_test_batches)):
        x_test_batch = x_test_batches[i]
        y_test_batch = y_test_batches[i]
        y_pred = predict(x_test_batch, nn, activation_function_settings)

        y_pred_all.append(y_pred)
        y_true_all.append(y_test_batch)
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    # print(y_pred_all)
    y_true_all = np.concatenate(y_true_all, axis=0)
    accuracy_test = accuracy(y_pred_all, y_true_all)
    return accuracy_test