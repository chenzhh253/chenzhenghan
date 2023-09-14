# 代码精读笔记
## `model.py`
**定义了一个GCN模型，包括两个图卷积层和相应的前向传播和L2正则化损失计算方法。**
1. 导入所需的库和模块：

```python
import torch
from torch import nn
from torch.nn import functional as F
from layer import GraphConvolution
from config import args
```

2. 定义GCN类，继承自nn.Module类：

```python
class GCN(nn.Module):
```

3. 在GCN类的构造函数`__init__`中初始化输入维度、输出维度和非零特征数量：

```python
def __init__(self, input_dim, output_dim, num_features_nonzero):
    super(GCN, self).__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
```

4. 创建一个包含两个图卷积层的Sequential模型：

```python
self.layers = nn.Sequential(
    GraphConvolution(self.input_dim, args.hidden, num_features_nonzero,
                     activation=F.relu,
                     dropout=args.dropout,
                     is_sparse_inputs=True),
    GraphConvolution(args.hidden, output_dim, num_features_nonzero,
                     activation=F.relu,
                     dropout=args.dropout,
                     is_sparse_inputs=False)
)
```

5. 定义前向传播函数`forward`，接受输入inputs，包含节点特征x和邻接矩阵support：

```python
def forward(self, inputs):
    x, support = inputs
    x = x.to(torch.float32)
    x = self.layers((x, support))
    return x
```

6. 定义L2正则化损失函数`l2_loss`：

```python
def l2_loss(self):
    layer = self.layers.children()
    layer = next(iter(layer))
    loss = None
    for p in layer.parameters():
        if loss is None:
            loss = p.pow(2).sum()
        else:
            loss += p.pow(2).sum()
    return loss
```

## `train.py`
这段代码用于评估训练好的GCN模型在测试集上的准确率。首先，将模型设置为评估模式。然后，通过调用GCN模型的前向传播方法，得到模型在测试集上的输出。最后，计算模型在测试集上的准确率，并打印出来。
这段代码实现了一个使用PyTorch的图卷积网络（GCN）模型，并在给定的数据集上进行训练和评估。
1. 导入必要的库和模块：

```python
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
import numpy as np
from data import load_data, preprocess_features, preprocess_adj
from model import GCN
from config import args
from utils import masked_loss, masked_acc
```

在这段代码中，我们导入了一些必要的库和模块。这些库和模块包括了PyTorch的相关模块（如`torch`、`nn`、`optim`、`functional`），以及一些自定义的模块（如`data`、`model`、`config`和`utils`）。

2. 设置随机种子：

```python
seed = 134
np.random.seed(seed)
torch.random.manual_seed(seed)
```

这段代码设置了随机种子，以确保实验的可重复性。通过设置相同的随机种子，每次运行代码时得到的随机结果都是相同的。

3. 加载数据：

```python
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(args.dataset)
```

这行代码调用了`load_data`函数来加载数据集。`args.dataset`是一个参数，用于指定要加载的数据集。`load_data`函数会返回加载的数据集的相关信息，包括邻接矩阵`adj`、特征矩阵`features`，以及训练集、验证集和测试集的标签`y_train`、`y_val`、`y_test`，以及训练集、验证集和测试集的掩码`train_mask`、`val_mask`、`test_mask`。

4. 预处理特征和邻接矩阵：

```python
features = preprocess_features(features)
supports = preprocess_adj(adj)
```

这两行代码分别对特征矩阵和邻接矩阵进行预处理。
`preprocess_features`函数将特征矩阵进行归一化处理，返回归一化后的特征矩阵。`preprocess_adj`函数将邻接矩阵进行预处理，返回一个稀疏矩阵的表示形式。

5. 设置设备和数据类型：

```python
device = torch.device('cuda')
train_label = torch.from_numpy(y_train).long().to(device)
num_classes = train_label.shape[1]
train_label = train_label.argmax(dim=1)
train_mask = torch.from_numpy(train_mask.astype(np.int64)).to(device)
val_label = torch.from_numpy(y_val).long().to(device)
val_label = val_label.argmax(dim=1)
val_mask = torch.from_numpy(val_mask.astype(np.int64)).to(device)
test_label = torch.from_numpy(y_test).long().to(device)
test_label = test_label.argmax(dim=1)
test_mask = torch.from_numpy(test_mask.astype(np.int64)).to(device)
```

这段代码设置了设备（`device`）为CUDA，并将数据加载到CUDA设备上。
首先，将训练集、验证集和测试集的标签转换为PyTorch的张量，并将其移动到CUDA设备上。
然后，将标签转换为one-hot编码形式。
最后，将训练集、验证集和测试集的掩码转换为PyTorch的张量，并将其移动到CUDA设备上。

6. 创建稀疏张量：

```python
i = torch.from_numpy(features[0]).long().to(device)
v = torch.from_numpy(features[1]).to(device)
feature = torch.sparse.FloatTensor(i.t(), v, features[2]).to(device)

i = torch.from_numpy(supports[0]).long().to(device)
v = torch.from_numpy(supports[1]).to(device)
support = torch.sparse.FloatTensor(i.t(), v, supports[2]).float().to(device)
```

这段代码创建了稀疏张量（sparse tensor）来表示特征矩阵和邻接矩阵。首先，将特征矩阵的非零元素的行索引、列索引和值分别存储在`i`、`v`和`features[2]`中，并将它们转换为PyTorch的张量。然后，使用`torch.sparse.FloatTensor`函数创建稀疏张量。同样，对邻接矩阵也进行了相同的操作。

7. 创建GCN模型和优化器：

```python
net = GCN(feat_dim, num_classes, num_features_nonzero)
net.to(device)
optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)
```

这段代码创建了一个GCN模型，并将其移动到CUDA设备上。`feat_dim`表示输入特征的维度，`num_classes`表示类别的数量，`num_features_nonzero`表示特征矩阵中非零元素的数量。然后，创建了一个Adam优化器，用于优化GCN模型的参数。

8. 模型训练：

```python
net.train()
for epoch in range(args.epochs):
    out = net((feature, support))
    out = out[0]
    loss = masked_loss(out, train_label, train_mask)
    loss += args.weight_decay * net.l2_loss()
    acc = masked_acc(out, train_label, train_mask)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(epoch, loss.item(), acc.item())
```

这段代码用于训练GCN模型。首先，将模型设置为训练模式。然后，通过调用GCN模型的前向传播方法，得到模型的输出。接下来，计算损失函数，并加上L2正则化项。然后，计算模型的准确率。接着，将梯度清零，进行反向传播和参数更新。最后，每隔10个epoch打印当前的损失和准确率。

9. 模型评估：

```python
net.eval()
out = net((feature, support))
out = out[0]
acc = masked_acc(out, test_label, test_mask)
print('test:', acc.item())
```
## `utils.py`
这段代码实现了一个图卷积网络（GCN）模型，并提供了一些辅助函数。

- `masked_loss(out, label, mask)`函数计算带有掩码的交叉熵损失。它接受模型的输出`out`、标签`label`和掩码`mask`作为输入，并返回损失值。掩码用于过滤掉不需要计算损失的样本。

- `masked_acc(out, label, mask)`函数计算带有掩码的准确率。它接受模型的输出`out`、标签`label`和掩码`mask`作为输入，并返回准确率。

- `sparse_dropout(x, rate, noise_shape)`函数实现了稀疏输入的随机失活。它接受输入张量`x`、失活率`rate`和噪声形状`noise_shape`作为输入，并返回应用了随机失活的稀疏张量。

- `dot(x, y, sparse=False)`函数计算两个张量的点积。它接受输入张量`x`和`y`，并返回它们的点积。如果`sparse`参数设置为`True`，则使用稀疏矩阵乘法计算点积。

这些函数可以用于定义GCN模型的前向传播和损失计算。


