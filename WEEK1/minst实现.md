# MINSTD数据集的分类器模型实现
## 代码实现
```python
########手写数字数据集##########
###########保存模型############
########1层隐含层（全连接层）##########
#60000条训练数据和10000条测试数据，28x28像素的灰度图像
#隐含层激活函数：ReLU函数
#输出层激活函数：softmax函数（实现多分类）
#损失函数：稀疏交叉熵损失函数
#输入层有784个节点，隐含层有128个神经元，输出层有10个节点
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import time
print('--------------')
nowtime = time.strftime('%Y-%m-%d %H:%M:%S')
print(nowtime)

#指定GPU
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#初始化
plt.rcParams['font.sans-serif'] = ['SimHei']

#加载数据
mnist = tf.keras.datasets.mnist
(train_x,train_y),(test_x,test_y) = mnist.load_data()
print('\n train_x:%s, train_y:%s, test_x:%s, test_y:%s'%(train_x.shape,train_y.shape,test_x.shape,test_y.shape)) 

#数据预处理
#X_train = train_x.reshape((60000,28*28))
#Y_train = train_y.reshape((60000,28*28))       #后面采用tf.keras.layers.Flatten()改变数组形状
X_train,X_test = tf.cast(train_x/255.0,tf.float32),tf.cast(test_x/255.0,tf.float32)     #归一化
y_train,y_test = tf.cast(train_y,tf.int16),tf.cast(test_y,tf.int16)

#建立模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))     #添加Flatten层说明输入数据的形状
model.add(tf.keras.layers.Dense(128,activation='relu'))     #添加隐含层，为全连接层，128个节点，relu激活函数
model.add(tf.keras.layers.Dense(10,activation='softmax'))   #添加输出层，为全连接层，10个节点，softmax激活函数
print('\n',model.summary())     #查看网络结构和参数信息

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

#保存模型参数
#model.save_weights('C:\\Users\\xuyansong\\Desktop\\深度学习\\python\\MNIST\\模型参数\\mnist_weights.h5')
#保存整个模型
model.save('mnist_weights.h5')


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
    demo = tf.reshape(X_test[num],(1,28,28))
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
该代码片段是使用TensorFlow库来构建一个手写数字识别模型的示例。
大概分为几个步骤：

1. 导入所需的库：首先，我们导入了需要使用的TensorFlow库以及其他一些必要的库。

```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
```

2. 加载MNIST数据集：接下来，我们使用`input_data.read_data_sets`函数加载MNIST数据集。

```python
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

3. 定义模型：定义了一个包含1层隐含层和1层输出层的全连接神经网络模型。

```python
input_size = 784
hidden_size = 128
output_size = 10

x = tf.placeholder(tf.float32, [None, input_size])
y = tf.placeholder(tf.float32, [None, output_size])

W1 = tf.Variable(tf.random_normal([input_size, hidden_size]))
b1 = tf.Variable(tf.random_normal([hidden_size]))

W2 = tf.Variable(tf.random_normal([hidden_size, output_size]))
b2 = tf.Variable(tf.random_normal([output_size]))

hidden_layer = tf.nn.relu(tf.matmul(x, W1) + b1)
output_layer = tf.matmul(hidden_layer, W2) + b2
```

在这个模型中，输入层有784个神经元，隐含层有128个神经元，输出层有10个神经元。

4. 定义损失函数和优化器：使用稀疏交叉熵作为损失函数，Adam优化器作为优化器。

```python
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output_layer))
optimizer = tf.train.AdamOptimizer().minimize(loss)
```

5. 定义评估函数：为了评估模型的性能，定义了准确率的评估函数。

```python
correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
```

6. 训练模型：使用训练数据进行模型的训练。

```python
num_epochs = 10
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for epoch in range(num_epochs):
        num_batches = mnist.train.num_examples // batch_size
        
        for batch in range(num_batches):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        
        # 每个epoch结束时打印损失函数和准确率
        curr_loss, curr_acc = sess.run([loss, accuracy], feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Epoch: {}, Loss: {:.4f}, Accuracy: {:.4f}".format(epoch+1, curr_loss, curr_acc))
```

在每个epoch结束时，我们通过将测试数据提供给模型来计算损失函数和准确率，并将它们打印出来。

7. 保存模型：我们使用`tf.train.Saver`保存模型的参数。

```python
saver = tf.train.Saver()
saver.save(sess, "model.ckpt")
```

8. 可视化结果：最后，我们使用Matplotlib库对损失函数和准确率进行可视化，并显示一些模型的预测结果。

```python
import matplotlib.pyplot as plt

# ...
# 在训练循环之后添加以下代码来可视化结果
# ...

plt.plot(losses, label='Loss')
plt.plot(accuracies, label='Accuracy')
plt.legend()
plt.show()
```

改代码展示了使用TensorFlow构建、训练和保存手写数字识别模型的一般流程。

## 运行结果
![image](https://github.com/chenzhh253/chenzhenghan/assets/145008761/bace217c-fe19-43da-9ceb-66dc83431032)
![image](https://github.com/chenzhh253/chenzhenghan/assets/145008761/54e7d649-83f8-4edd-b8f4-9ecc749f5ca6)


