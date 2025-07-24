import numpy as np
import matplotlib.pyplot as plt

# 加载npz文件
data = np.load('512_256_0.01_0.1_best_weights.npz')

# 提取权重参数数组
loaded_weights = [data['arr_0'], data['arr_1'], data['arr_2']]


# 可视化第一个权重数组
plt.figure(figsize=(10, 10))
plt.imshow(loaded_weights[0], cmap='viridis')
plt.colorbar()
plt.title('Visualization of First Weight Array')
plt.show()

# 可视化第二个权重数组
plt.figure(figsize=(10, 10))
plt.imshow(loaded_weights[1], cmap='plasma')
plt.colorbar()
plt.title('Visualization of Second Weight Array')
plt.show()

# 可视化第三个权重数组
plt.figure(figsize=(10, 10))
plt.imshow(loaded_weights[2], cmap='inferno')
plt.colorbar()
plt.title('Visualization of Third Weight Array')
plt.show()