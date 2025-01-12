from LCNet import base_LCNet, prune_LCNet
from Train import train_and_val, set_seed
from Model_prune_test import test, prune_model
from data_load import TrainDataset, TestDataset
from optimizer_APGD_NAG import APGD
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class HP_train:
    classifier_lr = 0.01
    lambda_lr = 0.001
    alpha = 2
    epoch = 100
    train_batch_size = 256
    val_batch_size = 309 # 64
    classes = 10
    save_path = './Model_weights/LCNet_epoch%d_classifier_lr%.3f_lambda_lr%.3f_alpha%.2f.pth' % (
        epoch, classifier_lr, lambda_lr, alpha)
    prune_save_path = './Prune_Model_weights/PruneLCNet_epoch%d_classifier_lr%.3f_lambda_lr%.3f_alpha%.2f.pth' % (
    epoch, classifier_lr, lambda_lr, alpha)



class HP_test:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 128
    load_path = HP_train.save_path
    save_path = HP_train.prune_save_path


def main():
    set_seed(300)
    # 1. 加载数据
    X_train_label, X_val, Y_train_label, Y_val = TrainDataset(30)
    train_dataset = TensorDataset(torch.Tensor(X_train_label), torch.Tensor(Y_train_label))
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))

    # 2. 加载模型
    model = base_LCNet(num_classes=HP_train.classes, n_neuron=32, n_mobileunit=7).to(device)

    # 3. 设置损失函数和优化器，超参数在HP_train中修改
    loss_nll = nn.CrossEntropyLoss(reduction='sum').to(
        device)  # 无lambda优化时为reduction='mean', 有lambda优化时为reduction='sum'
    optimizer_classifier = torch.optim.Adam(model.parameters(), lr=HP_train.classifier_lr)
    #
    optimizer_lambda = APGD([{'params': model.lamda}], alpha=HP_train.alpha, device=device, lr=HP_train.lambda_lr,
                            momentum=0, weight_decay=0, nesterov=False)  # PGD
    # optimizer_lambda = APGD([{'params': model.lamda}], alpha=HP_train.alpha, device=device, lr=HP_train.lambda_lr,
    #                         momentum=0.9, weight_decay=0.0001, nesterov=False)  # APGD
    # optimizer_lambda = APGD([{'params': model.lamda}], alpha=HP_train.alpha, device=device, lr=HP_train.lambda_lr,
    #                         momentum=0.9, weight_decay=0.0001, nesterov=True)  # APGD-NAG
    # 4. train and val
    train_loss_list, train_acc_list, s_f_list = train_and_val(model,
                                                    loss_nll,
                                                    HP_train,
                                                    train_dataset,
                                                    val_dataset,
                                                    optimizer_classifier,
                                                    optimizer_lambda)

    # 5. 将train_loss和train_acc写进excel
    df1 = pd.DataFrame(train_loss_list)
    loss_name = 'Train_Loss_Acc/Loss/LCNet_epoch%d_lr%.3f_alpha%.4f_PGD_loss.xlsx' % (
        HP_train.epoch, HP_train.lambda_lr, HP_train.alpha)
    df1.to_excel(loss_name, sheet_name="Sheet1", index=False, engine='openpyxl')

    df2 = pd.DataFrame(train_acc_list)
    acc_name = 'Train_Loss_Acc/Acc/LCNet_epoch%d_lr%.3f_alpha%.4f_PGD_acc.xlsx' % (
        HP_train.epoch, HP_train.lambda_lr, HP_train.alpha)
    df2.to_excel(acc_name, sheet_name="Sheet1", index=False, engine='openpyxl')

    df3 = pd.DataFrame(s_f_list)
    acc_name = 'Train_Loss_Acc/LCNet_epoch%d_lr%.3f_alpha%.4f_sf.xlsx' % (
        HP_train.epoch, HP_train.lambda_lr, HP_train.alpha)
    df3.to_excel(acc_name, sheet_name="Sheet1", index=False, engine='openpyxl')


if __name__ == '__main__':
    # 1. train and val
    main()

    # 2. Load the test data
    X_test, Y_test = TestDataset()
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # 3. purne model
    prune_model(HP_test, prune_LCNet)

    # 4. test
    purne_model = torch.load(HP_train.prune_save_path)  # prune 训练时修改
    test(purne_model, test_dataloader)
