import copy
import time
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import GoogLeNet, Inception
import torch.nn as nn
import pandas as pd

def train_val_data_process():
    train_data = ImageFolder(
        root='./data/train',  # 直接指向 train 文件夹（里面要有 dog 和 cat 子文件夹）
        transform=transforms.Compose([
            transforms.Resize((224, 224)),  # 统一缩放到 224x224
            transforms.RandomHorizontalFlip(),  # 新增：随机左右翻转（增强数据，防止过拟合）
            transforms.ToTensor(),  # 转成张量
            transforms.Normalize(  # 新增：标准化（让模型训练更稳定）
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    )

    train_data, val_data = Data.random_split(train_data, [round(len(train_data) * 0.8), round(len(train_data) * 0.2)])

    train_dataloader = Data.DataLoader(dataset = train_data,
                                       batch_size = 32,
                                       shuffle = True,
                                       num_workers =4)

    val_dataloader = Data.DataLoader(dataset = val_data,
                                       batch_size=32,
                                       shuffle=False,
                                       num_workers=4)

    return train_dataloader, val_dataloader

def train_model_process(model, train_dataloader, val_dataloader, num_epochs):

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    #   使用Adam优化器，学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    #   损失函数为交叉熵函数
    criterion = nn.CrossEntropyLoss()
    #   将模型放入训练设备中
    model = model.to(device)
    #   复制当前模型参数
    best_model_wts = copy.deepcopy(model.state_dict())

    #   初始化参数
    #   最高准确度
    best_acc = 0.0
    #   训练集损失列表
    train_loss_all = []
    #   验证集损失列表
    val_loss_all = []
    #   训练集准确度列表
    train_acc_all = []
    #   验证集准确度列表
    val_acc_all = []
    #   当前时间
    since = time.time()

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        #   初始化参数
        #   训练集损失值
        train_loss = 0.0
        #   训练集准确度
        train_correct = 0
        #   验证集损失值
        val_loss = 0.0
        #   验证集准确度
        val_correct = 0
        #   训练集样本数量
        train_num = 0
        #   验证集样本数量
        val_num = 0

        for step, (b_x, b_y) in enumerate(train_dataloader):
            #   将特征放入到训练设备中
            b_x = b_x.to(device)
            #   将标签放入到训练设备中
            b_y = b_y.to(device)
            #   设置模型为训练模式
            model.train()

            #   前向传播过程，输入为一个batch，输出为一个batch对应的预测
            output = model(b_x)
            #   当前batch每一行中概率值最大的行标
            pre_lab = torch.argmax(output, dim=1)
            #   计算每一个batch的损失值
            loss = criterion(output, b_y)

            #   将梯度初始化为0
            optimizer.zero_grad()
            #   反向传播
            loss.backward()
            #   根据网络反向传播的梯度信息来更新网络的参数， 起到降低loss值的作用
            optimizer.step()
            #   对损失值进行累加
            train_loss += loss.item() * b_x.size(0)
            #   预测准确， 准确度加1
            train_correct += torch.sum(pre_lab == b_y)
            #   当前训练过的样本数
            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            model.eval()

            output = model(b_x)
            pre_lab = torch.argmax(output, dim=1)
            loss = criterion(output, b_y)

            val_loss += loss.item() * b_x.size(0)
            val_correct += torch.sum(pre_lab == b_y)
            val_num += b_x.size(0)

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_correct.float().item() / train_num)

        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_correct.float().item() / val_num)

        print("{} trains, train_loss:{:.4f}, train_acc:{:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} trains, val_loss:{:.4f}, val_acc:{:.4f}".format(epoch,val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
            best_model_wts = copy.deepcopy(model.state_dict())

        time_used = time.time() - since
        print("训练、验证耗费时间：{:.0f}min{:.0f}s".format(time_used//60, time_used%60))

        model.load_state_dict(best_model_wts)
        torch.save(model.state_dict(), './model.pth')

    train_process = pd.DataFrame(data = {"epoch":range(num_epochs),
                                "train_loss_all":train_loss_all,
                                "train_acc_all":train_acc_all,
                                "val_loss_all":val_loss_all,
                                "val_acc_all":val_acc_all})

    return train_process

def matplot_acc_loss(train_process):
    plt.figure(figsize = (12,4))
    plt.subplot(1,2,1)
    plt.plot(train_process["epoch"],train_process.train_loss_all,'ro-',label="train_loss")
    plt.plot(train_process["epoch"],train_process.val_loss_all,'bo-',label="val_loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process["epoch"], train_process.train_acc_all, 'ro-', label="train_acc")
    plt.plot(train_process["epoch"], train_process.val_acc_all, 'bo-', label="val_acc")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.show()


if __name__ == '__main__':
    GoogLeNet = GoogLeNet(Inception)
    train_dataloader, val_dataloader = train_val_data_process()
    train_process = train_model_process(GoogLeNet, train_dataloader, val_dataloader, 16)
    matplot_acc_loss(train_process)