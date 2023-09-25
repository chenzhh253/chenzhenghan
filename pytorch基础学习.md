# Pytorch基础学习笔记
## Dataset类
### 简介
Dataset是一个用于处理数据集的Python库。它提供了一组工具和功能，使得数据集的处理和转换变得更加简单和高效。Dataset库可以用于加载、
处理和转换各种类型的数据集，包括结构化数据、文本数据、图像数据等。
一些基本函数：
load_dataset()：用于加载数据集的函数，可以从不同的数据源加载数据。
Dataset()：表示数据集的类，可以用于创建一个数据集对象。
map()：用于对数据集中的每个样本应用一个函数，进行数据转换或处理。
filter()：用于对数据集中的样本进行过滤，保留符合条件的样本。
shuffle()：用于对数据集中的样本进行随机排序，打乱数据的顺序。
batch()：用于将数据集划分为批次，每个批次包含一定数量的样本。
split()：用于将数据集划分为训练集、验证集和测试集，按照指定的比例或数量进行划分。
iter()：用于创建数据集的迭代器，可以用于遍历数据集的样本。
take()：用于从数据集中获取指定数量的样本。
visualize()：用于数据可视化的函数，可以绘制数据集的图表或图像。
### 代码实例
这段代码定义了一个自定义的数据集类MyData，继承自torch.utils.data.Dataset。
该数据集类用于加载图像数据集，并提供了获取图像和标签的方法。
```python
from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir  
        self.label_dir=label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.img_path=os.listdir(self.path)
    def __getitem__(self,idx):
        img_name=self.img_path[idx]
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)
        img=Image.open(img_item_path)
        label=self.label_dir
        return img,label
    def __len__(self):
        return len(self.img_path)

root_dir="dataset/hymenoptera_data/train"
ants_label_dir="ants"
bees_label_dir="bees"
ants_dataset=MyData(root_dir,ants_label_dir)
bees_dataset=MyData(root_dir,bees_label_dir)

train_dataset=ants_dataset+bees_dataset
```
**结果：**
![image](https://github.com/chenzhh253/chenzhenghan/assets/145008761/7b913f35-c14a-43f3-a836-705348655bcb)


## TensorBoard的使用
terminal进入pytorch：`conda activate pytorch`C:\\Users\\czh31\\基础学习\\Tensorboard\\logs
不知道为什么打不开
