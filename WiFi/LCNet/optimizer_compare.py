from LCNet import base_LCNet, prune_LCNet
from data_load import TrainDataset, TestDataset
from optimizer_APGD_NAG import APGD
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch import nn
from Train import train_and_val, set_seed
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class HP_train:
    classifier_lr = 0.01
    lambda_lr = 0.001
    alpha = 15
    epoch = 100
    train_batch_size = 256
    val_batch_size = 64
    classes = 16


if __name__ == '__main__':
    set_seed(300)

    # 2. 加载数据
    X_train_label, X_val, Y_train_label, Y_val = TrainDataset(30)
    train_dataset = TensorDataset(torch.Tensor(X_train_label), torch.Tensor(Y_train_label))
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))

    # 3. 设置损失函数和优化器
    loss_nll = nn.CrossEntropyLoss(reduction='sum').to(device)

    # 4. 对3个优化器依次进行训练，并写进excel里
    train_loss_list = np.zeros((100, 3))
    train_acc_list = np.zeros((100, 3))

    # 初始化三个基模型，使用相同的参数
    model1 = base_LCNet(num_classes=HP_train.classes, n_neuron=32, n_mobileunit=7).to(device)
    model2 = base_LCNet(num_classes=HP_train.classes, n_neuron=32, n_mobileunit=7).to(device)
    model3 = base_LCNet(num_classes=HP_train.classes, n_neuron=32, n_mobileunit=7).to(device)

    # 为每个模型的lambda参数设置优化器，采用不同的动量和权重衰减策略
    optimizer_lambda_1 = APGD([{'params': model1.lamda}], alpha=HP_train.alpha, device=device, lr=HP_train.lambda_lr,
                              momentum=0, weight_decay=0)
    optimizer_lambda_2 = APGD([{'params': model2.lamda}], alpha=HP_train.alpha, device=device, lr=HP_train.lambda_lr,
                              momentum=0.9, weight_decay=0.0001, nesterov=False)
    optimizer_lambda_3 = APGD([{'params': model3.lamda}], alpha=HP_train.alpha, device=device, lr=HP_train.lambda_lr,
                              momentum=0.9, weight_decay=0.0001, nesterov=True)

    # 为每个模型设置分类器的Adam优化器
    optimizer_classifier_1 = torch.optim.Adam(model1.parameters(), lr=HP_train.classifier_lr)
    optimizer_classifier_2 = torch.optim.Adam(model2.parameters(), lr=HP_train.classifier_lr)
    optimizer_classifier_3 = torch.optim.Adam(model3.parameters(), lr=HP_train.classifier_lr)

    # 将优化器聚合在一起，便于管理和调用
    optimizer_lambda = {'PGD': optimizer_lambda_1, 'APGD': optimizer_lambda_2, 'APGD_NAG': optimizer_lambda_3}
    optimizer_classifier = [optimizer_classifier_1,optimizer_classifier_2,optimizer_classifier_3]
    model = [model1, model2, model3]



    i = 0
    for key, optimizer_value in optimizer_lambda.items():
        print(f"Starting training with {key} optimizer")
        train_loss_list[:, i], train_acc_list[:, i] = train_and_val(model[i],
                                                                    loss_nll,
                                                                    HP_train,
                                                                    train_dataset,
                                                                    val_dataset,
                                                                    optimizer_classifier[i],
                                                                    optimizer_value)
        i += 1


    df1 = pd.DataFrame(train_loss_list)
    df2 = pd.DataFrame(train_acc_list)
    loss_name = 'Optimizer_Loss_Acc/LCNet_epoch%d_lr%.3f_alpha%.2f_loss.xlsx' % (
        HP_train.epoch, HP_train.lambda_lr, HP_train.alpha)
    acc_name = 'Optimizer_Loss_Acc/LCNet_epoch%d_lr%.3f_alpha%.2f_acc .xlsx' % (
        HP_train.epoch, HP_train.lambda_lr, HP_train.alpha)
    df1.to_excel(loss_name, index=False)
    df2.to_excel(acc_name, index=False)

    # 6. 绘制三种优化器的损失和精度图
    iterations = range(len(train_loss_list))
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(iterations, train_loss_list[:, 0], 'r-', label='Optimizer 1 Loss')
    ax1.plot(iterations, train_loss_list[:, 1], 'r--', label='Optimizer 2 Loss')
    ax1.plot(iterations, train_loss_list[:, 2], 'r-.', label='Optimizer 3 Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    # 创建第二个纵坐标轴共享同一个横坐标轴
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(iterations, train_acc_list[:, 0], 'b-', label='Optimizer 1 Accuracy')
    ax2.plot(iterations, train_acc_list[:, 1], 'b--', label='Optimizer 2 Accuracy')
    ax2.plot(iterations, train_acc_list[:, 2], 'b-.', label='Optimizer 3 Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    # 添加图例
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    # 设置标题
    plt.title('LCNet')
    path = 'Optimizer_Loss_Acc/classifier_lr%.3f_lambda_lr%.3f_alpha%.2f_LCNet.png' % (
    HP_train.classifier_lr, HP_train.lambda_lr, HP_train.alpha)
    # 保存图像
    plt.savefig(path, dpi=600)
    # 显示图形
    plt.show()