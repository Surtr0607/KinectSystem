
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, Activation
from PIL import Image, ImageDraw

# 创建模型
model = Sequential()
model.add(Conv2D(1, (3, 3), padding='same', input_shape=(None, None, 1),
                 kernel_initializer='zeros',
                 bias_initializer='ones'))
model.add(Activation('relu'))

# 设置Sobel边缘检测器的权重
weights = [np.array([[[[-1.]],[[0.]],[[1.]]],
                     [[[-2.]],[[0.]],[[2.]]],
                     [[[-1.]],[[0.]],[[1.]]]], dtype=np.float32),
           np.array([0], dtype=np.float32)]
model.set_weights(weights)

# 创建一个矩形图像
image = Image.new('L', (100, 100))  # 'L'表示灰度图像
draw = ImageDraw.Draw(image)
draw.rectangle([(20, 20), (80, 80)], fill=255)  # 使用填充色255绘制矩形

# 显示原始图像
plt.figure(figsize=(6, 6))
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.show()

# 将图像转换为NumPy数组，并扩展维度以适应模型输入
image = np.expand_dims(np.array(image), (-1, 0))

# 使用模型进行边缘检测
edges = model.predict(image)

# 显示边缘检测结果
plt.figure(figsize=(6, 6))
plt.imshow(edges[0, :, :, 0], cmap='gray')
plt.title('Edge Detection')
plt.show()