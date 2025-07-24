import os
import gzip
import numpy as np


def load_data(path, kind):
    # 加载数据
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


# 调整y
def integer_to_one_hot(labels, num_classes):
    return np.eye(num_classes)[labels]


def get_vali(x_train, y_train, split_ratio):
    # 随机化数据集
    random_indices = np.random.permutation(len(x_train))
    x_train_shuffled = x_train[random_indices]
    y_train_shuffled = y_train[random_indices]

    # 分割数据为训练集和验证集
    split_index = int(len(x_train_shuffled) * split_ratio)

    x_train_split, x_vali_split = x_train_shuffled[:split_index], x_train_shuffled[split_index:]
    y_train_split, y_vali_split = y_train_shuffled[:split_index], y_train_shuffled[split_index:]
    return x_train_split, y_train_split, x_vali_split, y_vali_split


def gen_batch(x_data, y_data, batch_size=32):
    # 生成batch
    n_samples = x_data.shape[0]
    indices = np.random.permutation(n_samples)
    x_data_batched = []
    y_data_batched = []
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i + batch_size]
        x_data_batched.append(x_data[batch_indices])
        y_data_batched.append(y_data[batch_indices])
    return [x_data_batched, y_data_batched]


def get_data_fashion_mnist(path, vali_split_ratio=0.9, batch_size=32):
    # 加载数据
    x_train, y_train = load_data(path, "train")
    x_train = x_train.astype('float32') / 255
    y_train = y_train.astype('int32')
    x_test, y_test = load_data(path, "t10k")
    x_test = x_test.astype('float32') / 255
    y_test = y_test.astype('int32')
    # 切出验证集
    x_train, y_train, x_vali, y_vali = get_vali(x_train, y_train, vali_split_ratio)
    # 生成batch
    batches = []
    batches.append(gen_batch(x_train, y_train, batch_size))
    batches.append(gen_batch(x_vali, y_vali, batch_size))
    batches.append(gen_batch(x_test, y_test, batch_size))
    return batches

# 在 dataloader.py 末尾追加
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

def get_data_cifar10(data_dir, val_split_ratio=0.1, batch_size=32):
    # 读取5个训练batch
    x_train, y_train = [], []
    for i in range(1, 6):
        d = unpickle(f"{data_dir}/data_batch_{i}")
        x_train.append(d[b'data'])
        y_train.append(np.array(d[b'labels']))
    x_train = np.concatenate(x_train).astype('float32') / 255.0
    y_train = np.concatenate(y_train)

    # 读取测试batch
    d = unpickle(f"{data_dir}/test_batch")
    x_test = d[b'data'].astype('float32') / 255.0
    y_test = np.array(d[b'labels'])

    # 划分训练/验证
    split = int((1 - val_split_ratio) * len(x_train))
    idx = np.random.permutation(len(x_train))
    x_tr, y_tr = x_train[idx[:split]], y_train[idx[:split]]
    x_val, y_val = x_train[idx[split:]], y_train[idx[split:]]

    # 打平 & 打包成 batch
    x_tr = x_tr.reshape(-1, 3072)
    x_val = x_val.reshape(-1, 3072)
    x_test = x_test.reshape(-1, 3072)

    def make_batches(x, y):
        idx = np.arange(len(x))
        np.random.shuffle(idx)
        x, y = x[idx], y[idx]
        return [x[i:i+batch_size] for i in range(0,len(x),batch_size)], \
               [y[i:i+batch_size] for i in range(0,len(y),batch_size)]

    x_tr_b, y_tr_b = make_batches(x_tr, y_tr)
    x_val_b, y_val_b = make_batches(x_val, y_val)
    x_te_b, y_te_b = make_batches(x_test, y_test)
    return [[x_tr_b, y_tr_b], [x_val_b, y_val_b], [x_te_b, y_te_b]]