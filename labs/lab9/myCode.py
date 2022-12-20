# %% [markdown]
# # Import 库

# %%
import time

import pandas as pd
import platform
import io

# 可视化相关libraries
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm 

# pytorch相关libraries 以及再命名
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn as nn
import torch.nn.functional as F

# pytorch中torchvision集成了计算机视觉相关的数据集
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torchvision

import matplotlib.pyplot as plt
import numpy as np

# %%
# 定义一些常量
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
COLOR_CHANNELS = 3
N_CLASSES = 10

# %%
# 检查 cuda 的可用性
cuda = torch.cuda.is_available()
print(cuda)

# %% [markdown]
# # 加载CIFAR-10的数据集和标签

# %%
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# %%
train_dataset = CIFAR10(root='.',
                        train=True,
                        transform=transform,
                        download=True)

test_dataset  = CIFAR10(root='.',
                        train=False,
                        transform=transform,)


# %%
# 检查部分如下
data_batch_1 = pd.read_pickle(r'./cifar-10-batches-py/data_batch_1')
meta_data    = pd.read_pickle(r'./cifar-10-batches-py/batches.meta')

# %%
print(data_batch_1.keys())
print(meta_data.keys())

# %%
one_hot_labels = set(data_batch_1['labels']) # One-hot encoding labels
one_hot_labels

# %%
labels_name = meta_data['label_names']
labels_name

# %%
label_dict = {k:v for k,v in zip(one_hot_labels, labels_name)} # dict(zip(labels, one_hot))
label_dict

# %%
label_dict.get(8)

# %%
# 检查数据集维度
train_data = list(train_dataset)
test_data  = list(test_dataset)
print(f'Train data shape: {len(train_data)}')
print(f'Test data shape:  {len(test_data)}')

# %%
# 检查单个数据的维度
image, label = train_data[300]
print(image.shape)
print(label)

# %%
# 检查输出的图片
def imshow(img, label):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0))) # tensor.permute = np.transpose
    plt.title(''.join(f'{label_dict.get(label)}' )) # show label
    

# plt.imshow(image.permute(1,2,0))
# plt.show()

image, label = train_data[20]

imshow(image, label)

plt.show()

# %% [markdown]
# ## 创建 DataLoader

# %%
batch_size = 64         
train_loader = torch.utils.data.DataLoader(
                            dataset = train_dataset, 
                            batch_size = batch_size, 
                            shuffle = True)

test_loader = torch.utils.data.DataLoader(
                            dataset = test_dataset, 
                            batch_size = batch_size, 
                            shuffle = True)

# %%
# 检查维度
dd = list(train_loader)


# %%
images, labels = dd[0]
print(images.shape)
print(labels.shape)


# %% [markdown]
# # 网络搭建
# 
# 将搭建如下三种网络：
# * Multi-Layer Perceptron (MLP)
# * Convolutional Neural Network (CNN)
# * ResNet9

# %% [markdown]
# ## MLP

# %%
class MLP(torch.nn.Module):
  def __init__(self, input_size, n_hidden_units, n_classes):
    super(MLP, self).__init__()

    h1, h2, h3 = n_hidden_units

    # Add Linear layers
    self.fc1 = nn.Linear(input_size,h1)
    self.fc2 = nn.Linear(h1, h2)
    self.fc3 = nn.Linear(h2, h3)
    self.fc4 = nn.Linear(h3, n_classes) # 10 classes
    self.dropout = nn.Dropout(0.25)

  def forward(self, x):
    x = x.view(-1, input_size) # Flatten out 
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.dropout(x)
    x = self.fc4(x)
    return x



# %%
# 初始化
input_size = IMAGE_HEIGHT*IMAGE_WIDTH*COLOR_CHANNELS
n_hidden_units = [512, 256, 128]

mlp_model = MLP(input_size, n_hidden_units, N_CLASSES)

# %%
# 检查output的shape
output = mlp_model(images)
output.shape

# %% [markdown]
# ## CNN

# %%
def conv3x3(in_channels, out_channels, stride=1):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


# %%
class CNN(torch.nn.Module):
  def __init__(self, in_channels, n_classes):
    super(CNN, self).__init__()
    # Conv layers
    self.conv =conv3x3(in_channels=in_channels, out_channels=64)
    self.bn = nn.BatchNorm2d(64)
    self.relu = nn.ReLU(inplace=True)

    #self.conv0 =conv3x3(in_channels=in_channels, out_channels=64)
    #self.bn0 = nn.BatchNorm2d(64)

    # Residual block
    self.conv1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
    self.bn1 = nn.BatchNorm2d(64)
    self.relu1 = nn.ReLU(inplace=True)

    self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
    self.bn2 = nn.BatchNorm2d(64)
    self.relu2 = nn.ReLU(inplace=True)

    # Pooling
    self.avg_pool = nn.AvgPool2d(2)

    # Linear layers
    self.fc1 = nn.Linear(16384, 64 )
    self.fc2 = nn.Linear(64, n_classes)

  def forward(self, x):
    
    #print(identity.shape)
    out = self.conv(x)
    out = self.bn(out)
    out = self.relu(out)

    #out = self.conv0(x)
    #out = self.bn0(out)
    #out = self.relu(out)

    out = self.conv1(out)
    out = self.bn1(out)
    out = self.relu1(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu2(out)  # skip connection + out
    #print(out.shape)

    out = self.avg_pool(out)

    out = out.view(out.size(0), -1) # Flatten all dimension except batch
    #print(out.shape)

    out = self.fc1(out)
    out = F.relu(out)

    out = self.fc2(out)
    return out

# %%
cnn_model = CNN(in_channels=3, n_classes=10)

# %%
images, labels = dd[0]
print(images.shape)
print(labels.shape)

# %%
# 检查output的shape
output = cnn_model(images)
output.shape

# %% [markdown]
# ## ResNet9

# %%
class BasicResidualBlock(nn.Module):
  def __init__(self):
    super(BasicResidualBlock, self).__init__()
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
    self.relu1 = nn.ReLU()
    self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
    self.relu2 = nn.ReLU()

  def forward(self, x):
    identity = x
    out = self.conv1(x)
    out = self.relu1(out)
    out = self.conv2(out)
    out = self.relu2(out)
    return self.relu2(out) + identity

# %%
# 检查output的shape
simple_resblock = BasicResidualBlock()
images, labels = dd[0]
print(images.shape)
print(labels.shape)

# %%
out = simple_resblock(images)
out.shape

# %%
def conv_block(in_channels, out_channels, pool=False):
  layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)]
  if pool: 
    layers.append(nn.MaxPool2d(2))
  return nn.Sequential(*layers)

# %%
class ResNet9(nn.Module):
  def __init__(self, in_channels, n_classes):
    super(ResNet9, self).__init__()

    self.conv1 = conv_block(in_channels, 64)
    self.conv2 = conv_block(64, 128, pool=True)
    self.res1  = nn.Sequential(conv_block(128,128), conv_block(128,128))

    self.conv3 = conv_block(128, 256, pool=True)
    self.conv4 = conv_block(256, 512, pool=True)
    self.res2  = nn.Sequential(conv_block(512,512), conv_block(512,512))

    self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                    nn.Flatten(),
                                    nn.Linear(512, 128),
                                    nn.Linear(128, n_classes))
    
  def forward(self, x):
    out = self.conv1(x)
    out = self.conv2(out)
    out = self.res1(out) + out # skip connection
    out = self.conv3(out)
    out = self.conv4(out)
    out = self.res2(out) + out # skip connection
    out = self.classifier(out)
    return out

# %%
resnet_model = ResNet9(3,10)
resnet_model

# %%
images, labels = dd[0]
print(images.shape)
print(labels.shape)

# %%
out = resnet_model(images)
out.shape

# %% [markdown]
# ### 定义辅助函数

# %%
def prediction(data_loader, model, criterion, cuda=None):
  correct = 0
  total = 0
  losses = 0

  for i, (images, labels) in enumerate(data_loader):
    if cuda is not None:
      # 转到 GPU
      images = images.cuda()
      labels = labels.cuda()
    
    outputs = model(images)
    
    loss = criterion(outputs, labels)
  
    _, predictions = torch.max(outputs, dim=1)
  
    correct += torch.sum(labels == predictions).item()
    total += labels.shape[0]
    
    losses += loss.data.item()
    
  return losses/len(list(data_loader)), 1 - correct/total 

# %% [markdown]
# # 构造损失函数

# %% [markdown]
# 1. 对于学习率，采用“一周期内学习率改变法则”：
#  从较低的学习率开始，在大约30%的epoch中，逐批增加到较高的学习率，然后在剩余的     epoch中逐渐降低到非常低的值。 
# 2. 正则化，通过在损失函数中增加一项来防止权重过大 
# 3. 梯度剪切:可以将梯度值限制在一个较小的范围内，防止因梯度值过大导致参数发生不必要的变化。
# 

# %% [markdown]
# ## Fit one cycle一周期内学习率改变法则

# %%
def get_lr(optimizer):
  for param_group in optimizer.param_groups:
    return param_group['lr']


# %%
def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, criterion,
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD, cuda=None):
  torch.cuda.empty_cache() # 清空缓存
  history = []

  optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
  sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                              steps_per_epoch=len(train_loader))
  
  if cuda is not None:
    model.cuda() 

  train_losses = []
  test_losses = []

  train_error_rates = []
  test_error_rates = []

  for epoch in range(epochs):
    train_loss = 0
    n_iter = 0
    total = 0
    correct = 0
    lrs = []

    for i, (images, labels) in enumerate(train_loader):
      optimizer.zero_grad() # 初始化梯度为0

      if cuda is not None:
        images = images.cuda()
        labels = labels.cuda()

      outputs = model(images) 

  
      _, predictions = torch.max(outputs, 1) 
      correct += torch.sum(labels == predictions).item()
      total += labels.shape[0]

      # 计算 loss
      loss = criterion(outputs, labels)
      # 反向传播
      loss.backward()

      # 梯度剪切
      if grad_clip is not None:
        nn.utils.clip_grad_value_(model.parameters(), grad_clip)

      # 更新梯度
      optimizer.step()

      # 记录并更新学习率
      lrs.append(get_lr(optimizer))
      sched.step()

      train_loss += loss.detach().item()

      n_iter += 1
    
    train_error_rate = 1 - correct/total

    with torch.no_grad():
      test_loss, test_error_rate = prediction(val_loader, model, criterion, cuda)

    train_error_rates.append(train_error_rate)
    test_error_rates.append(test_error_rate)
    train_losses.append(train_loss/n_iter)
    test_losses.append(test_loss)
    results ={'train_loss': train_loss/n_iter,'val_loss': test_loss, 'val_acc': (1-train_error_rate)*100}
    results['lrs'] = lrs
    history.append(results)

    if epoch%1 == 0:
      print('Epoch: {}/{}, last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_accuracy: {:.1f}%'.format(epoch+1, epochs, results['lrs'][-1], train_loss/n_iter, test_loss, (1-train_error_rate)*100))

  return history
    
    

# %% [markdown]
# # 网络训练

# %%
epochs = 1
max_lr = 0.01
grad_clip = 0.1
weight_decay = 1e-4
opt_func = torch.optim.Adam
criterion = nn.CrossEntropyLoss()

# %% [markdown]
# ### MLP

# %%
history_mlp = fit_one_cycle(epochs, max_lr, mlp_model, train_loader, test_loader, criterion, weight_decay, opt_func=opt_func )

# %%
# 保存模型训练结果
PATH_MLP = './mlp_cifar_net.pth'
torch.save(mlp_model.state_dict(), PATH_MLP)

# %% [markdown]
# ### CNN

# %%
history_cnn = fit_one_cycle(epochs, max_lr, cnn_model, train_loader, test_loader, criterion, weight_decay, opt_func=opt_func)

# %%
# 保存模型训练结果
PATH_CNN = './cnn_cifar_net.pth'
torch.save(cnn_model.state_dict(), PATH_CNN)

# %% [markdown]
# ### ResNet9

# %%
history_resnet = fit_one_cycle(epochs, max_lr, resnet_model, train_loader, test_loader, criterion, weight_decay, opt_func=opt_func )

# %%
PATH_RESNET = './rescnn_cifar_lr_net.pth'
torch.save(resnet_model.state_dict(), PATH_RESNET)

# %% [markdown]
# # 模型评估

# %%
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');

# %%
def plot_losses(history):
  train_losses = [x.get('train_loss') for x in history]
  test_losses = [x['val_loss'] for x in history]
  plt.plot(train_losses, label='train', marker='o', alpha=0.7)
  plt.plot(test_losses, label='test', marker='o', alpha=0.7)
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.grid(True)
  plt.legend()
  plt.title('Model Loss')

# %%
plt.figure(figsize=(10, 8))
plt.subplot(1,3,1)
plot_losses(history_mlp)
plt.title('MLP')
plt.subplot(1,3,2)
plot_losses(history_cnn)
plt.title('CNN')
plt.subplot(1,3,3)
plot_losses(history_resnet)
plt.title('ResNet9')

# %%
plt.figure(figsize=(10, 8))
plt.subplot(1,3,1)
plot_accuracies(history_mlp)
plt.title('MLP')
plt.subplot(1,3,2)
plot_accuracies(history_cnn)
plt.title('CNN')
plt.subplot(1,3,3)
plot_accuracies(history_resnet)
plt.title('ResNet9')

# %% [markdown]
# ## 将预测可视化

# %%
dataiter = iter(test_loader)
images, labels = dataiter.next()

# %%
images.shape

# %%
labels.shape

# %%
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# %%
dataiter = iter(test_loader)
images, labels = dataiter.next()
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
# print images
imshow(torchvision.utils.make_grid(images[:5]))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(5)))

# %%
resnet = ResNet9(in_channels=3, n_classes=10)
resnet.load_state_dict(torch.load(PATH_RESNET))

# %%
outputs = resnet(images)

# %%
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(5)))

# %%



