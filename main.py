import numpy as np
from codes.network import neural_network
from codes.trainer import train_my_network
from codes.dataloader import get_data_cifar10
from codes.draw_my_plots import draw_pic_accuracy, draw_pic_loss
from codes.testor import test_my_model


# 定义网络结构
settings = [
    {"input_dim": 3072, "out_put_dim": 768, "activation": "relu"},
    {"input_dim": 768, "out_put_dim": 384,  "activation": "relu"},
    {"input_dim": 384,  "out_put_dim": 10,   "activation": "softmax"}
]
nn, activation_functions = neural_network(settings)

# 读取batch后的数据集 形式为[[x_train_batch, y_train_batch], [x_vali_batch, y_vali_batch], [x_test_batch, y_test_batch]]
path = "./fashion_mnist"
sample_batches = get_data_cifar10("./cifar-10-batches-py", val_split_ratio=0.1, batch_size=32)
print("数据读取完毕")

# 训练
train_result = train_my_network(nn, activation_function_settings=activation_functions, x_train_batches=sample_batches[0][0],
                 y_train_batches=sample_batches[0][1], x_vali_batches=sample_batches[1][0],
                 y_vali_batches=sample_batches[1][1], l2_lambda=0.1, epochs=100, learning_rate=0.005,
                 learning_decay_rate=0.001, learning_rate_decay_steps=50, save_best_model=True)
print("训练完毕")
loss_all, accuracy_vali_all, best_val_loss, best_weights = train_result[0], train_result[1], train_result[2], train_result[3]

# 绘制loss图像和accuracy
draw_pic_loss(loss_all, "loss_plot.png")
draw_pic_accuracy(accuracy_vali_all, "vali_accuracy.png")

# 保存权重到本地文件
np.savez('best_weights.npz', *best_weights)
print("best_val_loss: ", best_val_loss)

# 测试
# 加载本地文件中的权重
data = np.load('best_weights.npz')
loaded_weights = [data['arr_0'], data['arr_1'], data['arr_2']]  # 第一层的权重
test_accuracy = test_my_model(loaded_weights, activation_functions, x_test_batches=sample_batches[2][0],
                                              y_test_batches=sample_batches[2][1])
print("在测试集上的准确度为：", test_accuracy)