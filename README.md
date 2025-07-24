# 神经网络和深度学习作业1
手工搭建三层神经网络分类器，在数据集 CIFAR-10 上进行训练以实现图像分类。

## 文件结构
- `main.py`：包括数据载入、模型训练、参数权重存储以及测试的完整流程。
- `search.py`：进行参数网格搜索，评估不同参数组合的模型训练效果。
- `weights_visual.py`：训练权重参数的可视化。

## 快速开始

### 安装依赖

安装必需的 Python 库：

```bash
pip install numpy matplotlib tqdm
```
请注意最新numpy可能会报错，请回退numpy版本
### 数据准备
可直接克隆项目内的数据集
或浏览器直接下载：
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
解压后目录结构应为
project/
├─ cifar10/
│  ├─ data_batch_1~5
│  ├─ test_batch
│  └─ ...
├─ main.py
├─ search.py
└─ weights_visual.py

```
/fashion-mnist/
```

### 训练模型

1. 打开 `main.py`，根据需求调整网络设置和 `train_my_network()` 函数的参数。（参数可通过search查找，我找到的{dim_1}_{dim_2}_{learning_rate}_{l2}为512_256_0.01_0.1）
2. 执行以下命令以启动训练：

```bash
python main.py
```

训练完成后，模型的损失图像和验证集准确率图像将被保存到指定目录。

## 测试模型

在模型参数保存之后，系统会自动执行测试并返回测试集上的准确率结果。您也可以通过加载保存的参数并调用测试函数进行单独测试。

### 权重可视化

运行 `weights_visual.py` 以读取并可视化训练过的权重参数：

```bash
python weights_visual.py
```
