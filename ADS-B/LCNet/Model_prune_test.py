# -*-coding:utf-8-*-
# @Time : 2024/10/19 11:25
import torch
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
'''
fuction: 修剪对应行
'''


def prune_new(input1, input2):
    # 将 input1 和 input2 从计算图中分离出来，并转换为 numpy 数组
    input1 = input1.cpu().detach_().numpy()
    input2 = input2.cpu().detach_().numpy()

    # 找到 input1 中所有非零元素的索引
    i = np.nonzero(input1)
    input2 = np.array(input2)
    # 将 input2 转换为 numpy 数组（如果它还不是的话）
    # 根据非零元素的索引，选择 input2 中的相应元素
    input2_new = input2[i, :]
    # 将 numpy 数组转换回 PyTorch 张量
    input2_new_ = torch.tensor(input2_new, dtype=torch.float32)
    # 将张量移动到设备上
    input2_new_ = input2_new_.to(device)
    # 去除张量中多余的维度
    input2_new_ = input2_new_.squeeze()
    return input2_new_


'''
fuction: 修剪对应列
'''


def prune_new2(input1, input2):
    input1 = input1.cpu().detach_().numpy()
    input2 = input2.cpu().detach_().numpy()
    i = np.nonzero(input1)
    input2 = np.array(input2)
    input2_new = input2[:, i]
    input2_new_ = torch.tensor(input2_new, dtype=torch.float32)
    input2_new_ = input2_new_.to(device)
    input2_new_ = input2_new_.squeeze()
    return input2_new_, i


'''
fuction: 修剪对应元素
'''


def prune_new3(input1, input2):
    input1 = input1.cpu().detach_().numpy()
    input2 = input2.cpu().detach_().numpy()
    i = np.nonzero(input1)
    input2 = np.array(input2)
    input2_new = input2[i]
    input2_new_ = torch.tensor(input2_new, dtype=torch.float32)
    input2_new_ = input2_new_.to(device)
    input2_new_ = input2_new_.squeeze()
    return input2_new_


'''
fuction: 模型剪枝
'''

def get_prune_paras(hp_test):
    load_path = hp_test.load_path  # original_weights
    save_path = hp_test.save_path  # prune_weights

    model = torch.load(load_path)
    torch.save(model.state_dict(), save_path)
    dict_ = torch.load(save_path)

    input_ = dict_["lamda"]
    input_ = input_.cpu().detach_().numpy()
    f_list = np.nonzero(input_)
    f_list = np.array(f_list).squeeze()
    m = len(f_list)
    return f_list, m


def prune_model(hp_test, model_prune):
    # savepath已保存的模型
    # loadpath剪枝后模型

    load_path = hp_test.load_path  # original_weights
    save_path = hp_test.save_path  # prune_weights

    model = torch.load(load_path)
    torch.save(model.state_dict(), save_path)
    dict = torch.load(save_path)

    #  --------进行剪枝 --------
    # tensor_new = prune_new(dict["lamda"], dict["linear1.weight"])
    # dict["linear1.weight"] = tensor_new
    tensor_new2, f_list = prune_new2(dict["lamda"], dict["linear.weight"])
    dict["linear.weight"] = tensor_new2
    # tensor_new3 = prune_new3(dict["lamda"], dict["linear1.bias"])
    # dict["linear1.bias"] = tensor_new3
    tensor_lamda = prune_new3(dict["lamda"], dict["lamda"])
    dict["lamda"] = tensor_lamda
    # print(f_list)


    #  --------计算特征稀疏度 --------
    m = 0
    for i in tensor_lamda.cpu().numpy():
        if i != 0:
            m = m + 1
    print('特征维度:', str(m))
    print('特征稀疏度:', str(m / 1216))

    f_list = np.array(f_list).squeeze()

    model_new = model_prune(num_classes=10, n_neuron=32, n_mobileunit=7, f_list=f_list, m=m)
    model_new.load_state_dict(dict)
    model_new = model_new.to(device)
    torch.save(model_new, hp_test.save_path)
    return hp_test.save_path

def test(model, test_dataloader):

    model.eval()
    correct = 0

    target_pred = []
    target_real = []
    with torch.no_grad():
        for data, target in test_dataloader:
            target = target.long()
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            target_pred[len(target_pred):len(target) - 1] = pred.tolist()
            target_real[len(target_real):len(target) - 1] = target.tolist()
        #
        # target_pred = np.array(target_pred)
        # target_real = np.array(target_real)

    # 将预测标签存下来
    # data_Y_pred = pd.DataFrame(target_pred)
    # writer = pd.ExcelWriter("DRCN_15label/Y_pred.xlsx")
    # data_Y_pred.to_excel(writer, 'page_1', float_format='%.5f')
    # writer.save()
    # writer.close()

    # 将原始标签存下来
    # data_Y_real = pd.DataFrame(target_real)
    # writer = pd.ExcelWriter("DRCN_15label/Y_real.xlsx")
    # data_Y_real.to_excel(writer, 'page_1', float_format='%.5f')
    # writer.save()
    # writer.close()

    fmt = '\nTest set: Accuracy: {}/{} ({:.6f}%)\n'
    print(
        fmt.format(
            correct,
            len(test_dataloader.dataset),
            100.0 * correct / len(test_dataloader.dataset),
        )
    )
