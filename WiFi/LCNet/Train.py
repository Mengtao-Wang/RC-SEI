import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import random
import os

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


def train(model, loss_nll, hp_train, train_dataloader, optimizer_classifier, optimizer_lambda, epoch):

    model.train()

    correct = 0
    result_loss = 0
    classifier_loss = 0
    r1_loss = 0

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
        result_loss += result_loss_batch.item()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    classifier_loss /= len(train_dataloader.dataset)
    r1_loss /= len(train_dataloader)
    result_loss /= len(train_dataloader)

    print('Train Epoch: {} \tclassifier_Loss: {:.6f}, lambda_Loss, {: 6f}, result_Loss: {: 6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
            epoch,
            classifier_loss,
            r1_loss,
            result_loss,
            correct,
            len(train_dataloader.dataset),
            100.0 * correct / len(train_dataloader.dataset))
    )
    return r1_loss, correct

def val(model, loss_nll, val_dataloader):
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_dataloader:
            target = target.long()
            data = data.to(device)
            target = target.to(device)

            s_f,output = model(data)
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

    return val_loss,s_f


def train_and_val(model,
                   loss_nll,
                   hp_train,
                   train_dataset,
                   val_dataset,
                   optimizer_classifier,
                   optimizer_lambda):
    train_loss_list = []
    train_acc_list = []
    s_f_list = []
    current_min_test_loss = 100
    train_dataloader = DataLoader(train_dataset, batch_size=hp_train.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=hp_train.val_batch_size, shuffle=True)
    for epoch in range(1, hp_train.epoch + 1):

        train_loss, train_correct = train(model, loss_nll, hp_train, train_dataloader, optimizer_classifier,optimizer_lambda, epoch)
        train_loss_list.append(train_loss)
        train_acc_list.append(100.0 * train_correct / len(train_dataloader.dataset))
        val_loss,s_f = val(model, loss_nll, val_dataloader)
        s_f_list.append(s_f)

        if val_loss< current_min_test_loss:
            print("The validation loss is improved from {} to {}, new encoder and classifier weight is saved.".format(
                current_min_test_loss, val_loss))
            current_min_test_loss = val_loss
            torch.save(model, hp_train.save_path)
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")
    return train_loss_list, train_acc_list, s_f_list

