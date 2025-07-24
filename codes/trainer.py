import numpy as np
from tqdm import tqdm
from codes.activation_functions import relu, sigmoid, softmax, relu_derivative, sigmoid_derivative


# 交叉熵损失
def sparse_cross_entropy_loss(y_pred, y_true):
    return -np.sum(np.log(y_pred[np.arange(len(y_pred)), y_true] + 1e-10)) / len(y_pred)


# SGD优化器
def sgd_optimizer(weights, grads, learning_rate):
    for weight, grad in zip(weights, grads):
        weight -= learning_rate * grad


# 学习率衰减
def learning_rate_decay(initial_rate, step, decay_rate=0.1, decay_steps=50):
    return initial_rate * (1 / (1 + decay_rate * (step // decay_steps)))


# L2正则化
def l2_regularization(weights, l2_lambda):
    l2_loss = 0.0
    for weight in weights:
        l2_loss += np.sum(np.square(weight))
    l2_loss = l2_loss * 0.5 * l2_lambda
    return l2_loss


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


def backpropagation(x, y, nn, activation_functions):
    # 前向传播，保存必要的中间结果
    activations = [x]
    zs = []  # z是激活函数的输入
    for i, weight in enumerate(nn):
        z = activations[-1].dot(weight)
        zs.append(z)
        if activation_functions[i] == 'relu':
            activation = relu(z)
        elif activation_functions[i] == 'sigmoid':
            activation = sigmoid(z)
        elif activation_functions[i] == 'softmax':
            activation = softmax(z)
        activations.append(activation)

    # 反向传播
    grads = [None] * len(nn)
    # softmax与稀疏交叉熵的组合导数
    delta = activations[-1]
    delta[np.arange(len(activations[-1])), y] -= 1
    grads[-1] = activations[-2].T.dot(delta)

    for l in range(2, len(nn) + 1):
        z = zs[-l]
        if activation_functions[-l] == 'relu':
            sp = relu_derivative(z)
        elif activation_functions[-l] == 'sigmoid':
            sp = sigmoid_derivative(z)
        delta = delta.dot(nn[-l + 1].T) * sp
        grads[-l] = activations[-l - 1].T.dot(delta)

    return grads


def accuracy(y_pred, y_true):
    # 将输出概率转化为类别预测
    y_pred_class = np.argmax(y_pred, axis=1)
    return np.mean(y_pred_class == y_true)

def evaluate(nn, activation_function_settings, x_vali_batches, y_vali_batches, l2_lambda):
    y_pred_all = []
    y_true_all = []
    val_loss = 0.0
    for i in range(len(x_vali_batches)):
        x_vali_batch = x_vali_batches[i]
        y_vali_batch = y_vali_batches[i]
        # print(nn)
        y_pred = predict(x_vali_batch, nn, activation_function_settings)

        y_pred_all.append(y_pred)
        y_true_all.append(y_vali_batch)
        val_loss += sparse_cross_entropy_loss(y_pred, y_vali_batch) + l2_regularization(nn, l2_lambda)
    y_pred_all = np.concatenate(y_pred_all, axis=0)
    # print(y_pred_all)
    y_true_all = np.concatenate(y_true_all, axis=0)
    accuracy_vali = accuracy(y_pred_all, y_true_all)
    return accuracy_vali, val_loss / len(x_vali_batches)


def train_my_network(nn, activation_function_settings, x_train_batches, y_train_batches, x_vali_batches, y_vali_batches, l2_lambda,
                     epochs=10, learning_rate=1e-4,
                     learning_decay_rate=0.1, learning_rate_decay_steps=50, save_best_model=True):
    best_val_loss = float("inf")
    best_weights = None
    loss_all = []
    accuracy_vali_all = []
    num_batches = len(x_train_batches)
    steps = 0
    for epoch in tqdm(range(epochs)):
        batch_indices = np.random.permutation(num_batches)
        loss_list = []
        for batch_index in batch_indices:
            steps += 1
            x_train_batch = x_train_batches[batch_index]
            y_train_batch = y_train_batches[batch_index]

            y_pred = predict(x_train_batch, nn, activation_function_settings)
            loss = sparse_cross_entropy_loss(y_pred, y_train_batch) + l2_regularization(nn, l2_lambda)
            loss_list.append(loss)

            grads = backpropagation(x_train_batch, y_train_batch, nn, activation_function_settings)
            for i, grad in enumerate(grads):
                grads[i] += l2_lambda * nn[i]

            lr = learning_rate_decay(learning_rate, steps, learning_decay_rate, learning_rate_decay_steps)
            sgd_optimizer(nn, grads, lr)

        loss_all.append(np.sum(loss_list))
        # 计算验证集上的准确率和损失
        accuracy_vali, val_loss = evaluate(nn, activation_function_settings, x_vali_batches, y_vali_batches, l2_lambda)
        print(accuracy_vali)
        accuracy_vali_all.append(accuracy_vali)

        # 保存最优模型
        if save_best_model and val_loss < best_val_loss:
            best_epoch = epoch
            best_val_loss = val_loss
            best_weights = [w.copy() for w in nn]  # 注意这里需要复制权重，否则后续的权重更新会影响已保存的最优模型

    if save_best_model:
        print(best_epoch)
        print(accuracy_vali_all)
        return [loss_all, accuracy_vali_all, best_val_loss, best_weights]
    else:
        return [loss_all, accuracy_vali_all, 'Current validation loss', nn]