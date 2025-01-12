from LCNet import base_LCNet, prune_LCNet
from data_load import TrainDataset, TestDataset
from optimizer_APGD_NAG import APGD
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)  # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现


def train(model, loss_nll, hp_train, train_dataloader, optimizer_lambda, epoch):
    model.train()

    correct = 0
    result_loss = 0
    classifier_loss = 0
    r1_loss = 0
    optimizer_classifier = torch.optim.Adam(model.parameters(), lr=HP_train.classifier_lr)
    # 加载数据
    for data_label in train_dataloader:
        data, target = data_label
        target = target.long()
        data = data.to(device)
        target = target.to(device)

        # 梯度清零
        optimizer_classifier.zero_grad()
        optimizer_lambda.zero_grad()

        # 计算交叉熵损失
        _, output = model(data)
        classifier_loss_batch = loss_nll(output, target)

        # 计算lambda损失
        zero_data = torch.zeros(model.lamda.size())
        zero_data = zero_data.to(device)
        r1_loss_batch = nn.SmoothL1Loss()(model.lamda, zero_data)

        # 计算联合损失

        result_loss_batch = classifier_loss_batch + hp_train.alpha * r1_loss_batch
        # result_loss_batch = classifier_loss_batch

        # 反向传播
        result_loss_batch.backward()

        # 更新参数
        optimizer_classifier.step()
        optimizer_lambda.step()

        # 输出损失/精度
        classifier_loss += classifier_loss_batch.item()
        r1_loss += r1_loss_batch.item()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    classifier_loss /= len(train_dataloader.dataset)
    r1_loss /= len(train_dataloader)
    result_loss = classifier_loss + hp_train.alpha * r1_loss

    print(
        'Train Epoch: {} \tclassifier_Loss: {:.6f}, lmbda_Loss, {: 6f}, result_Loss: {: 6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
            epoch,
            classifier_loss,
            r1_loss,
            result_loss,
            correct,
            len(train_dataloader.dataset),
            100.0 * correct / len(train_dataloader.dataset))
    )
    return result_loss, correct


def val(model, loss_nll, val_dataloader):
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_dataloader:
            target = target.long()
            data = data.to(device)
            target = target.to(device)

            _, output = model(data)
            val_loss += loss_nll(output, target)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_dataloader.dataset)
    fmt = '\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            val_loss,
            correct,
            len(val_dataloader.dataset),
            100.0 * correct / len(val_dataloader.dataset),
        ))

    return  val_loss


def train_and_val(model,
                  loss_nll,
                  hp_train,
                  train_dataset,
                  val_dataset,
                  optimizer_lambda):
    train_loss_list = []
    train_acc_list = []
    current_min_test_loss = 100
    train_dataloader = DataLoader(train_dataset, batch_size=hp_train.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=hp_train.val_batch_size, shuffle=True)
    for epoch in range(1, hp_train.epoch + 1):

        result_loss, train_correct = train(model, loss_nll, hp_train, train_dataloader,optimizer_lambda, epoch)
        train_loss_list.append(result_loss)
        train_acc_list.append(100.0 * train_correct / len(train_dataloader.dataset))
        val_loss = val(model, loss_nll, val_dataloader)


        if val_loss < current_min_test_loss:
            print("The validation loss is improved from {} to {}, new encoder and classifier weight is saved.".format(
                current_min_test_loss, val_loss))
            current_min_test_loss = val_loss
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")
    return train_loss_list, train_acc_list


class HP_train:
    classifier_lr = 0.01
    lambda_lr = 0.001
    alpha = 2
    epoch = 100
    train_batch_size = 256
    val_batch_size = 64
    classes = 10


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

    model1 = base_LCNet(num_classes=HP_train.classes, n_neuron=32, n_mobileunit=7).to(device)
    model2 = base_LCNet(num_classes=HP_train.classes, n_neuron=32, n_mobileunit=7).to(device)
    model3 = base_LCNet(num_classes=HP_train.classes, n_neuron=32, n_mobileunit=7).to(device)
    optimizer_lambda_1 = APGD([{'params': model1.lamda}], alpha=HP_train.alpha, device=device, lr=HP_train.lambda_lr,
                              momentum=0, weight_decay=0)
    optimizer_lambda_2 = APGD([{'params': model2.lamda}], alpha=HP_train.alpha, device=device, lr=HP_train.lambda_lr,
                              momentum=0.9, weight_decay=0.0001, nesterov=False)
    optimizer_lambda_3 = APGD([{'params': model3.lamda}], alpha=HP_train.alpha, device=device, lr=HP_train.lambda_lr,
                              momentum=0.9, weight_decay=0.0001, nesterov=True)

    optimizer = {'PGD': optimizer_lambda_1, 'APGD': optimizer_lambda_2, 'APGD_NAG': optimizer_lambda_3}
    model = [model1, model2, model3]



    i = 0
    for key, optimizer_value in optimizer.items():
        print(f"Starting training with {key} optimizer")
        train_loss_list[:, i], train_acc_list[:, i] = train_and_val(model[i],
                                                                    loss_nll,
                                                                    HP_train,
                                                                    train_dataset,
                                                                    val_dataset,
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
    path = 'Optimizer_Loss_Acc/classifier_lr%.3f_lambda_lr%.3f_alpha%.2f_LCNet.png' % (HP_train.classifier_lr, HP_train.lambda_lr, HP_train.alpha)
    # 保存图像
    plt.savefig(path, dpi=600)
    # 显示图形
    plt.show()