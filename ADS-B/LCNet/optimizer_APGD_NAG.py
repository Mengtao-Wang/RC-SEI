import torch
from torch.optim import Optimizer

class SGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:  # 遍历每个参数组
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']: #遍历每个参数
                if p.grad is None:
                    continue  # 如果参数没有梯度，则跳过该参数
                d_p = p.grad.data   # 获取参数的梯度
                if weight_decay != 0:  # 如果权重衰减系数不为0，则在梯度上添加权重衰减项
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p] # 获取参数的状态信息
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()  # 如果状态信息中没有动量缓冲区，则创建并初始化动量缓冲区。
                    else:
                        buf = param_state['momentum_buffer'] #否则，获取现有的动量缓冲区并更新它。
                        buf.mul_(momentum).add_(1 - dampening, d_p) # buf = buf * momentum + d_P
                    if nesterov:
                        d_p = d_p.add(momentum, buf)  # d_p = d_p + momentum * buf
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)  # p.data = p.data - lr * d_p

        return loss

class APGD(SGD):
    def __init__(self, params, alpha, device, lr=1e-3, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self.alpha = alpha
        self.device = device
        super(SGD, self).__init__(params, defaults)
        #super(APGD, self).__init__(params, lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        #super(APGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data   #参数的一阶梯度
                # update weight_decay
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                # update momentum
                if momentum != 0:
                    param_state = self.state[p]   # 之前的累计的数据，v(t-1)
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()  # detach()函数可以返回一个完全相同的tensor, 与旧tensor共享内存，脱离计算图，不牵扯梯度计算
                    else:
                        buf = param_state['momentum_buffer']

                    z = p.data.add(-group['lr'], d_p)  # * z = p.data - lr * d_p
                    z = self.soft_thresholding(z, group['lr'] * self.alpha, self.device)  # * z = S
                    if nesterov:
                        v = z - p.data + buf.mul_(momentum)   # * v = S - p.data + momentum * buf     buf: v_{t-1}  momentum: \mu （向前多走了一步p.data)
                        p.data = z + v.mul_(momentum)  # p.data = S + v * buf
                    else:
                        p.data = z + buf.mul_(momentum) # p.data = S + momentum * buf

                else:
                    z = p.data.add(-group['lr'], d_p) #  * z = p.data - lr * d_p
                    p.data = self.soft_thresholding(z, group['lr'] * self.alpha, self.device)  # P.data = S(z)

                if torch.cuda.is_available():
                    p.data = torch.max(p.data, torch.zeros(len(p.data)).to(self.device))
                else:
                    p.data = torch.max(p.data, torch.zeros(len(p.data)))
                scale = 1 / (torch.max(torch.abs(p.data)))
                p.data = scale * p.data  #对lamda除以max值修正

                    # average = (torch.sum(torch.abs(p.data))) / 1024
                    # p.data = self.hard_thresholding(p.data, average)  # 添加threshold
                    #print("lambda", p.data)
        return loss

    @staticmethod
    def soft_thresholding(input, alpha, device):
        #device = torch.device("cuda:" + str(device_num))
        if torch.cuda.is_available():
            return torch.sign(input) * torch.max(torch.zeros(len(input)).to(device), torch.abs(input) - alpha)

    @staticmethod
    def hard_thresholding(input, average):
        return (torch.sign(input - average) + torch.sign(input + average)) / 2