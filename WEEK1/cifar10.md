# CIFAR10数据集的卷积神经网络分类器的训练
## 代码实现
```python
########cifar10数据集##########
###########保存模型############
########卷积神经网络##########
#train_x:(50000, 32, 32, 3), train_y:(50000, 1), test_x:(10000, 32, 32, 3), test_y:(10000, 1)
#60000条训练数据和10000条测试数据，32x32像素的RGB图像
#第一层两个卷积层16个3*3卷积核，一个池化层：最大池化法2*2卷积核，激活函数：ReLU
#第二层两个卷积层32个3*3卷积核，一个池化层：最大池化法2*2卷积核，激活函数：ReLU
#隐含层激活函数：ReLU函数
#输出层激活函数：softmax函数（实现多分类）
#损失函数：稀疏交叉熵损失函数
#隐含层有128个神经元，输出层有10个节点
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import time
print('--------------')
nowtime = time.strftime('%Y-%m-%d %H:%M:%S')
print(nowtime)


#初始化
plt.rcParams['font.sans-serif'] = ['SimHei']

#加载数据
cifar10 = tf.keras.datasets.cifar10
(train_x,train_y),(test_x,test_y) = cifar10.load_data()
print('\n train_x:%s, train_y:%s, test_x:%s, test_y:%s'%(train_x.shape,train_y.shape,test_x.shape,test_y.shape)) 

#数据预处理
X_train,X_test = tf.cast(train_x/255.0,tf.float32),tf.cast(test_x/255.0,tf.float32)     #归一化
y_train,y_test = tf.cast(train_y,tf.int16),tf.cast(test_y,tf.int16)


#建立模型
model = tf.keras.Sequential()
##特征提取阶段
#第一层
model.add(tf.keras.layers.Conv2D(16,kernel_size=(3,3),padding='same',activation=tf.nn.relu,data_format='channels_last',input_shape=X_train.shape[1:]))  #卷积层，16个卷积核，大小（3，3），保持原图像大小，relu激活函数，输入形状（28，28，1）
model.add(tf.keras.layers.Conv2D(16,kernel_size=(3,3),padding='same',activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))   #池化层，最大值池化，卷积核（2，2）
#第二层
model.add(tf.keras.layers.Conv2D(32,kernel_size=(3,3),padding='same',activation=tf.nn.relu))
model.add(tf.keras.layers.Conv2D(32,kernel_size=(3,3),padding='same',activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
##分类识别阶段
#第三层
model.add(tf.keras.layers.Flatten())    #改变输入形状
#第四层
model.add(tf.keras.layers.Dense(128,activation='relu'))     #全连接网络层，128个神经元，relu激活函数
model.add(tf.keras.layers.Dense(10,activation='softmax'))   #输出层，10个节点
print(model.summary())      #查看网络结构和参数信息

#配置模型训练方法
#adam算法参数采用keras默认的公开参数，损失函数采用稀疏交叉熵损失函数，准确率采用稀疏分类准确率函数
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])   

#训练模型
#批量训练大小为64，迭代5次，测试集比例0.2（48000条训练集数据，12000条测试集数据）
print('--------------')
nowtime = time.strftime('%Y-%m-%d %H:%M:%S')
print('训练前时刻：'+str(nowtime))

history = model.fit(X_train,y_train,batch_size=64,epochs=5,validation_split=0.2)

print('--------------')
nowtime = time.strftime('%Y-%m-%d %H:%M:%S')
print('训练后时刻：'+str(nowtime))

#评估模型
model.evaluate(X_test,y_test,verbose=2)     #每次迭代输出一条记录，来评价该模型是否有比较好的泛化能力

#保存整个模型
model.save('CIFAR10_CNN_weights.h5')

#结果可视化
print(history.history)
loss = history.history['loss']          #训练集损失
val_loss = history.history['val_loss']  #测试集损失
acc = history.history['sparse_categorical_accuracy']            #训练集准确率
val_acc = history.history['val_sparse_categorical_accuracy']    #测试集准确率

plt.figure(figsize=(10,3))

plt.subplot(121)
plt.plot(loss,color='b',label='train')
plt.plot(val_loss,color='r',label='test')
plt.ylabel('loss')
plt.legend()

plt.subplot(122)
plt.plot(acc,color='b',label='train')
plt.plot(val_acc,color='r',label='test')
plt.ylabel('Accuracy')
plt.legend()

#暂停5秒关闭画布，否则画布一直打开的同时，会持续占用GPU内存
#根据需要自行选择
#plt.ion()       #打开交互式操作模式
#plt.show()
#plt.pause(5)
#plt.close()

#使用模型
plt.figure()
for i in range(10):
    num = np.random.randint(1,10000)

    plt.subplot(2,5,i+1)
    plt.axis('off')
    plt.imshow(test_x[num],cmap='gray')
    demo = tf.reshape(X_test[num],(1,32,32,3))
    y_pred = np.argmax(model.predict(demo))
    plt.title('标签值：'+str(test_y[num])+'\n预测值：'+str(y_pred))
#y_pred = np.argmax(model.predict(X_test[0:5]),axis=1)
#print('X_test[0:5]: %s'%(X_test[0:5].shape))
#print('y_pred: %s'%(y_pred))

#plt.ion()       #打开交互式操作模式
plt.show()
#plt.pause(5)
#plt.close()
```
## 代码学习

1. 首先，从TensorFlow库中导入必要的模块和函数：

```python
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
```

2. 接下来，使用`datasets`模块加载CIFAR10数据集。这个数据集包含了60000个32x32彩色图像，分为10个类别。将训练集和测试集分别存储在`train_images、train_labels`和`test_images、test_labels`这四个数组中：

```python
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
```

3. 在加载数据集后，可以使用`matplotlib.pyplot`库来可视化数据集中的一些图像。比如，可以显示训练集中的第一个图像和其对应的标签：

```python
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
```

4. 接下来，对数据进行预处理。首先，将图像像素值缩放到0到1之间，这可以通过除以255完成：

```python
train_images, test_images = train_images / 255.0, test_images / 255.0
```

这样做有助于确保图像数据的归一化和模型的更好训练。

5. 在数据准备完成后，我们可以定义CNN模型。在这个例子中，我们使用了一些常见的卷积层、池化层和全连接层。模型的结构可以通过以下代码来定义：

```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
```

这个模型使用了三个卷积层，每个卷积层后面跟着一个最大池化层，然后是一个全连接层。最后一层使用softmax激活函数，输出10个类别的概率分布。

6. 在模型定义完成后，需要对模型进行编译。这包括指定损失函数、优化器和评估指标。在这个例子中，使用了稀疏交叉熵作为损失函数，Adam优化器作为优化器，并使用准确率作为评估指标：

```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

7. 完成模型编译后，使用训练集对模型进行训练。以下代码将模型训练5个epochs（迭代周期）：

```python
history = model.fit(train_images, train_labels, epochs=5, 
                    validation_data=(test_images, test_labels))
```

训练过程将输出训练集和验证集的损失和准确率。

8. 训练完成后，使用测试集对模型进行评估。以下代码将输出测试集的损失和准确率：

```python
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)
```

9. 最后，我们可以保存整个模型供以后使用。以下代码将整个模型保存为HDF5格式的文件：

```python
model.save('cifar10_model.h5')
```
加载模型并进行预测。

## 结果
![image](https://github.com/chenzhh253/chenzhenghan/assets/145008761/0f7875fe-7d94-495d-91d8-b6f7ad2ec849)
![image](https://github.com/chenzhh253/chenzhenghan/assets/145008761/94db0d8d-f6a3-4b39-abe6-e087da6fa56e)

