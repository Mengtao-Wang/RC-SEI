import torch
import numpy as np
from torchsummary import summary
from thop import profile
from thop import clever_format
import time
import tqdm

from LCNet import LCNet, base_LCNet, prune_LCNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")


def calcGPUTime(model, input, alpha):
    """
    计算推理时间
    """
    model.eval()
    model.to(device)

    num_iterations = 1000  # 迭代次数

    total_forward_time = 0.0  # 使用time来测试
    # 记录开始时间
    start_event = time.time() * 1000
    with torch.no_grad():
        for _ in tqdm.tqdm(range(num_iterations)):
            start_forward_time = time.time()
            _ = model(input)
            end_forward_time = time.time()
            forward_time = end_forward_time - start_forward_time
            total_forward_time += forward_time * 1000  # 转换为毫秒

    # 记录结束时间
    end_event = time.time() * 1000

    elapsed_time = (end_event - start_event) / 1000.0  # 转换为秒
    fps = num_iterations / elapsed_time

    elapsed_time_ms = elapsed_time / (num_iterations * input.shape[0])

    avg_forward_time = total_forward_time / (num_iterations * input.shape[0])

    # print(f"FPS: {fps}")
    # print("elapsed_time_ms:", elapsed_time_ms * 1000)
    print('alpha = ', alpha)
    print(f"Avg Forward Time per sample: {avg_forward_time:.3f} ms")
    print('----------------------------------------------------------------')


def get_prune_paras(alpha=None):
    """
    获取稀释特征m, 和稀疏特征索引f_List
    """
    epoch = 100
    classifier_lr = 0.01
    lambda_lr = 0.001
    alpha = alpha  # 修改prune_model 对应的 alpha , 输出相应模型的参数量和计算量
    load_path = './Model_weights/LCNet_epoch%d_classifier_lr%.3f_lambda_lr%.3f_alpha%.2f.pth' % (
        epoch, classifier_lr, lambda_lr, alpha)
    save_path = './Prune_Model_weights/PruneLCNet_epoch%d_classifier_lr%.3f_lambda_lr%.3f_alpha%.2f.pth' % (
        epoch, classifier_lr, lambda_lr, alpha)
    model = torch.load(load_path)
    torch.save(model.state_dict(), save_path)
    dict_ = torch.load(save_path)

    input_ = dict_["lamda"]
    input_ = input_.cpu().detach_().numpy()
    f_list = np.nonzero(input_)
    f_list = np.array(f_list).squeeze()
    m = len(f_list)
    return m, f_list


if __name__ == '__main__':

    alpha = 0.01
    input_ = torch.rand(100, 2, 4800)
    m, f_list = get_prune_paras(alpha=alpha)

    model_0 = LCNet(num_classes=10, n_neuron=32, n_mobileunit=7)
    model_1 = base_LCNet(num_classes=10, n_neuron=32, n_mobileunit=7)  # model_1对应alpha = 0
    model_2 = prune_LCNet(num_classes=10, n_neuron=32, n_mobileunit=7, m=m, f_list=f_list)

    #  1. 获取Params
    Total_params, Params_size = summary(model_2, input_size=(2, 4800), batch_size=1, device="cpu")
    print('alpha =', alpha, '---', 'm =', m)
    print('Total_params =', int(Total_params))
    print('Params_size(MB) = {:.3f}'.format(Params_size))

    #  2. 计算flops
    flops, meters = profile(model_2, inputs=(input_,))
    flops, meters = clever_format([flops, meters], '%.6f')
    print('flops(M)={:}'.format(flops))

    #  3. 计算推理时间（单个样本）
    #  预热, GPU 平时可能为了节能而处于休眠状态, 因此需要预热
    print('warm up ...\n')
    model_0.to(device)
    input_test = input_.to(device)
    with torch.no_grad():
        for _ in range(100):
            # print("Model device:", next(model.parameters()).device)
            # print("Input device:", input_test.device)
            _ = model_0(input_test)
    print('testing ...\n')
    # alpha_ = [15, 12, 10, 5, 2, 1, 0.1, 0.01]
    alpha_ = [0.01, 0.1, 1, 2, 5, 10, 12, 15]
    calcGPUTime(model_0, input_test, alpha=0)  # 预热组
    calcGPUTime(model_0, input_test, alpha=0)
    for i in range(8):
        m, f_list = get_prune_paras(alpha=alpha_[i])
        model_ = prune_LCNet(num_classes=10, n_neuron=32, n_mobileunit=7, m=m, f_list=f_list)
        calcGPUTime(model_, input_test, alpha=alpha_[i])

