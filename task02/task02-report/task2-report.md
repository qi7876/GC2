# 任务二 LeNet-5 的 MindSpore 框架训练实验报告

## 1. 任务完成摘要



## 2. 任务目标



## 3. 主要内容



## 4. 主要思路及关键步骤

### 对 LeNet-5 的改进

使用 Fashion_MNIST 替代 MNIST 后，模型的准确率衰减到了0.85左右，于是我对模型做出了一些改进来提高其在 Fashion_MNIST 上的准确率。

#### 提高模型宽度

通过增加卷积核数量，提升了模型的宽度。

原网络：

```python
# 1 -> 6 -> 16 -> 120 -> 84 -> 10
self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode="valid")
self.conv2 = nn.Conv2d(6, 16, 5, pad_mode="valid")
self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))
```

修改后：

```python
# 1 -> 32 -> 64 -> 128 -> 256 -> 10
self.conv1 = nn.Conv2d(num_channel, 32, 5, pad_mode="valid")
self.conv2 = nn.Conv2d(32, 64, 5, pad_mode="valid")
self.fc1 = nn.Dense(64 * 5 * 5, 128, weight_init=Normal(0.02))
self.fc2 = nn.Dense(128, 256, weight_init=Normal(0.02))
self.fc3 = nn.Dense(256, num_class, weight_init=Normal(0.02))
```

**结果**

<img src="/Users/qi7876/Desktop/Screenshot 2024-10-06 at 20.01.47.png" style="zoom: 50%;" />

<img src="/Users/qi7876/Desktop/Screenshot 2024-10-06 at 19.18.45.png" style="zoom:50%;" />

<img src="/Users/qi7876/Desktop/Screenshot 2024-10-06 at 20.05.09.png" style="zoom:50%;" />

模型准确度提高了约0.015，但是不太稳定。

#### 添加批量规范化层（Batch Normalization）

原代码：

```python
self.conv1 = nn.Conv2d(num_channel, 32, 5, pad_mode="valid")
self.conv2 = nn.Conv2d(32, 64, 5, pad_mode="valid")

x = self.max_pool2d(self.relu(self.conv1(x)))
x = self.max_pool2d(self.relu(self.conv2(x)))
```

修改后：

```python
self.conv1 = nn.Conv2d(num_channel, 32, 5, pad_mode="valid")
self.bn1 = nn.BatchNorm2d(32)
self.conv2 = nn.Conv2d(32, 64, 5, pad_mode="valid")
self.bn2 = nn.BatchNorm2d(64)

x = self.max_pool2d(self.relu(self.bn1(self.conv1(x))))
x = self.max_pool2d(self.relu(self.bn2(self.conv2(x))))
```

**结果**

<img src="/Users/qi7876/Desktop/bn1.png" style="zoom:50%;" />

<img src="/Users/qi7876/Desktop/bn2.png" style="zoom:50%;" />

<img src="/Users/qi7876/Desktop/bn3.png" style="zoom:50%;" />

<img src="/Users/qi7876/Desktop/bn4.png" style="zoom:50%;" />

模型准确度提高到约0.87，并且较为稳定。

#### 优化器切换为 Adam

原代码：

```python
net_opt = nn.Momentum(net.trainable_params(), lr, momentum)
```

修改后：

```python
net_opt = nn.Adam(
    net.trainable_params(),
    learning_rate=lr,
    weight_decay=0.0,
    use_lazy=False,
    use_offload=False,
)
```

**结果**

<img src="/Users/qi7876/Desktop/adam1.png" style="zoom:50%;" />

<img src="/Users/qi7876/Desktop/adam2.png" style="zoom:50%;" />

<img src="/Users/qi7876/Desktop/adam3.png" style="zoom:50%;" />

<img src="/Users/qi7876/Desktop/adam4.png" style="zoom:50%;" />

有一些提高，但是不太稳定。



## 5. 完成情况与结果分析



## 6. 总结

