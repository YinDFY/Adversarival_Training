import torch as t
import torchvision
from torchvision import transforms
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from Alexnet import AlexNet

#图像转换
'''
 将输入的图片转换为 112*112大小。
'''
data_transforms = transforms.Compose([transforms.Resize([227, 227]),
                                      transforms.ToTensor()]
                                      )
#配置参数
BATCH_SIZE = 256     # 训练批量：一批256个
learning_rate = 0.001  # 学习率
EPOCHS = 7            # 学习7轮
numClasses = 43       # 分类数，一共43类
#训练集加载（43个文件夹，文件夹名就是分类序号）
train_data_path = './GTSRB'      #加载训练集
# 从路径加载图片，并将其转换为227*227大小
train_data = torchvision.datasets.ImageFolder(root = train_data_path, transform = data_transforms)


#训练集划分 为训练集和验证集（8-2分）
ratio = 0.8
n_train_examples = int(len(train_data) * ratio)
n_val_examples = len(train_data) - n_train_examples
train_data, val_data = data.random_split(train_data, [n_train_examples, n_val_examples])
print(f"Number of training samples = {len(train_data)}")
print(f"Number of validation samples = {len(val_data)}")


#创建训练集和验证集的Dataloader
trainloader = data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
valloader = data.DataLoader(val_data, shuffle=True, batch_size=BATCH_SIZE)

# 计算模型参数的个数（没什么用，看看就行）
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 加载神经网络
model=AlexNet(43)
print(f'The model has {count_parameters(model):,} trainable parameters.')

# 定义训练过程中的 优化器 和 损失函数（代价函数）
optimiser = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()


# 如果设备有显卡，则使用显卡进行计算
if t.cuda.is_available():
    model = model
    criterion = criterion.cuda()


# 准确度：衡量最终训练结果
def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc

# 训练函数
def train(model, loader, opt, criterion):
    epoch_loss = 0
    epoch_acc = 0

    # Train the model
    model.train()

    for (images, labels) in loader:
        images = images
        labels = labels

        output = model(images)

        loss = criterion(output, labels)


        opt.zero_grad()


        loss.backward()

        # Calculate accuracy
        acc = calculate_accuracy(output, labels)

        # Optimizing weights
        opt.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)

# 评估函数
def evaluate(model, loader, opt, criterion):
    epoch_loss = 0
    epoch_acc = 0

    # evaluate the model
    model.eval()

    with t.no_grad():
        for (images, labels) in loader:
            images = images
            labels = labels

            output = model(images)
            # output, _ = model(images)
            loss = criterion(output, labels)

            acc = calculate_accuracy(output, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(loader), epoch_acc / len(loader)


# 开始训练-----------------------------------------
train_loss_list = [0]*EPOCHS    #每轮训练的损失值
train_acc_list = [0]*EPOCHS     # 每轮训练的准确率
val_loss_list = [0]*EPOCHS      #每轮使用验证集验证的损失值
val_acc_list = [0]*EPOCHS       #每轮使用验证集验证的准确率

for epoch in range(EPOCHS):    #训练EPOCHS轮
    print("Epoch {}: ".format(epoch))
    train_start_time=time.monotonic()
    train_loss, train_acc= train(model, trainloader, optimiser, criterion)
    train_end_time = time.monotonic()

    val_start_time = time.monotonic()
    val_loss, val_acc = evaluate(model, valloader, optimiser, criterion)
    val_end_time = time.monotonic()

    train_loss_list[epoch] = train_loss
    train_acc_list[epoch] = train_acc
    val_loss_list[epoch] = val_loss
    val_acc_list[epoch] = val_acc

    print("Training: Loss = %.4f, Accuracy = %.4f, Time = %.2f seconds" %(train_loss, train_acc, train_end_time-train_start_time))
    print("Validation: Loss = {}, Accuracy = {}, Time = {} seconds".format(val_loss, val_acc, val_end_time - val_start_time))
    print("")

# 训练结束后，保存模型（这里只保存了模型参数）
MODEL_FOLDER = "./Model"
if not os.path.isdir(MODEL_FOLDER):
    os.mkdir(MODEL_FOLDER)

PATH_TO_MODEL = MODEL_FOLDER + "/pytorch_classification_AlexNetTS.pth"
if os.path.exists(PATH_TO_MODEL):
    os.remove(PATH_TO_MODEL)
t.save(model.state_dict(), PATH_TO_MODEL)   #保存模型参数

print("Model saved at %s" % (PATH_TO_MODEL))