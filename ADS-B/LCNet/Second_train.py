from Model_prune_test import test
from data_load import TrainDataset, TestDataset
from torch.utils.data import TensorDataset, DataLoader
from main import HP_train
from Train import set_seed
import torch
import torch.nn as nn
import pandas as pd
from LCNet import base_LCNet, prune_LCNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HP2_train:
    epoch = 100
    train_batch_size = 256
    val_batch_size = 64
    classes = 10
    save_path = 'Second_train_Model_weights/LCNet_epoch%d_classifier_lr%.3f_lambda_lr%.3f_alpha%.2f.pth' % (
        epoch, HP_train.classifier_lr, HP_train.lambda_lr, HP_train.alpha)


def second_train(model, loss_nll, train_dataloader, optimizer_classifier, epoch):
    model.train()
    correct = 0
    classifier_loss = 0

    for data_label in train_dataloader:
        data, target = data_label
        target = target.long()
        data = data.to(device)
        target = target.to(device)

        # 梯度清零
        optimizer_classifier.zero_grad()

        # 计算交叉熵损失
        output = model(data)
        classifier_loss_batch = loss_nll(output, target)

        # 反向传播
        # classifier_loss_batch.backward(retain_graph=True)
        classifier_loss_batch.backward()

        # 更新参数
        optimizer_classifier.step()

        # 输出损失/精度
        classifier_loss += classifier_loss_batch.item()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    classifier_loss /= len(train_dataloader)
    print(
        'Train Epoch: {} \tclassifier_loss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
            epoch,
            classifier_loss,
            correct,
            len(train_dataloader.dataset),
            100.0 * correct / len(train_dataloader.dataset))
    )

    return classifier_loss, correct


def second_val(model, loss_nll, val_dataloader):
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_dataloader:
            target = target.long()
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            val_loss += loss_nll(output, target)

            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_dataloader)
    fmt = '\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:0f}%)\n'
    print(
        fmt.format(
            val_loss,
            correct,
            len(val_dataloader.dataset),
            100.0 * correct / len(val_dataloader.dataset),
        ))

    return val_loss


def second_train_and_val(model,
                         loss_nll,
                         hp2_train,
                         train_dataset,
                         val_dataset,
                         optimizer_classifier):
    current_loss = 100
    train_loss_list = []
    train_acc_list = []

    train_dataloader = DataLoader(train_dataset, batch_size=hp2_train.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=hp2_train.val_batch_size, shuffle=True)

    for epoch in range(1, hp2_train.epoch + 1):
        # train

        train_loss, train_correct = second_train(model, loss_nll, train_dataloader, optimizer_classifier, epoch)
        val_loss = second_val(model, loss_nll, val_dataloader)

        train_loss_list.append(train_loss)
        train_acc_list.append(100.0 * train_correct / len(train_dataloader.dataset))

        if (epoch) % 20 == 0:
            for param_group in optimizer_classifier.param_groups:
                param_group['lr'] *= 0.5
            print(f"在第{epoch}轮，学习率减半为：{optimizer_classifier.param_groups[0]['lr']}")

        # save best model
        if val_loss < current_loss:
            print("The validation loss is improved from {} to {}, new encoder and classifier weight is saved.".format(
                current_loss, val_loss))
            current_loss = val_loss
            torch.save(model, hp2_train.save_path)
        else:
            print("The validation loss is not improved.")
        print("------------------------------------------------")

    return train_loss_list, train_acc_list


def main():
    set_seed(300)
    # 1. 加载数据
    X_train_label, X_val, Y_train_label, Y_val = TrainDataset(30)
    train_dataset = TensorDataset(torch.Tensor(X_train_label), torch.Tensor(Y_train_label))
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))

    # 2. 加载模型(使用之前的权重继续训练，只不过少了lamda优化)
    model = torch.load(HP_train.prune_save_path).to(device)
    # model = base_LCNet(num_classes=10, n_neuron=32, n_mobileunit=10).to(device)  # alpha =0
    # model = Std_LCNet(num_classes=10, n_neuron=32, n_mobileunit=7).to(device)  # alpha =0

    # 3. 设置损失函数和优化器，超参数在HP2_train中修改
    loss_nll = nn.CrossEntropyLoss(reduction='mean').to(device)
    optimizer_classifier = torch.optim.Adam(model.parameters(), lr=HP_train.classifier_lr)

    # 4. train and val
    train_loss_list, train_acc_list = second_train_and_val(model,
                                                           loss_nll,
                                                           HP2_train,
                                                           train_dataset,
                                                           val_dataset,
                                                           optimizer_classifier)

    # 5. 将train_loss和train_acc写进excel
    df1 = pd.DataFrame(train_loss_list)
    loss_name = 'Train_Loss_Acc/Loss/LCNet_epoch%d_lr%.3f_alpha%.4f_loss_2train.xlsx' % (
        HP2_train.epoch, HP_train.classifier_lr, HP_train.alpha)
    df1.to_excel(loss_name, sheet_name="Sheet1", index=False, engine='openpyxl')

    df2 = pd.DataFrame(train_acc_list)
    acc_name = 'Train_Loss_Acc/Acc/LCNet_epoch%d_lr%.3f_alpha%.4f_acc_2train.xlsx' % (
        HP2_train.epoch, HP_train.classifier_lr, HP_train.alpha)
    df2.to_excel(acc_name, sheet_name="Sheet1", index=False, engine='openpyxl')


if __name__ == '__main__':
    # 0. 超参数链接
    HP_train()
    print("  weight  /  alpha  =  ", HP_train.alpha)
    # 1. train and val
    # main()

    # 2. Load the test data
    X_test, Y_test = TestDataset()
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # 3. test
    model = torch.load(HP2_train.save_path)
    test(model, test_dataloader)
