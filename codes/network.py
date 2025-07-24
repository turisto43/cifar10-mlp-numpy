import numpy as np


def initialize_weights(input_dim, output_dim):
    return np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)


def neural_network(settings):
    # setting 输入字典列表，每个字典包含层神经元数和指定激活函数
    # {"input_dim": 768, "out_put_dim": 100, "activation": "relu"}
    nn = []
    activation = []

    for layer in settings:
        input_dim = layer["input_dim"]
        output_dim = layer["out_put_dim"]
        act_func = layer["activation"]
        weights = initialize_weights(input_dim, output_dim)
        nn.append(weights)
        activation.append(act_func)
    return nn, activation