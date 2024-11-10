import torch
from torch import nn

class FishNet(nn.Sequential):
    """
    钓鱼判断网络：用于判断是否应该点击鼠标
    输入：游戏状态（鱼的位置信息）
    输出：动作（是否点击）
    """
    def __init__(self, in_ch, out_ch):
        # 定义神经网络层
        layers=[
            nn.Linear(in_ch, 16),     # 输入层到隐藏层，16个神经元
            nn.LeakyReLU(),           # 激活函数：带泄漏的ReLU
            nn.Linear(16, out_ch)      # 隐藏层到输出层
        ]
        # 调用父类初始化
        super(FishNet, self).__init__(*layers)
        # 应用权重初始化
        self.apply(weight_init)

class MoveFishNet(nn.Sequential):
    """
    鱼竿移动网络：用于预测鱼竿应该移动的方向
    输入：游戏状态（鱼的位置信息）
    输出：移动方向
    """
    def __init__(self, in_ch, out_ch):
        # 定义一个更深的神经网络
        layers=[
            nn.Linear(in_ch, 32),     # 输入层到第一隐藏层，32个神经元
            nn.LeakyReLU(),           # 激活函数
            nn.Linear(32, 32),        # 第一隐藏层到第二隐藏层
            nn.LeakyReLU(),           # 激活函数
            nn.Linear(32, out_ch)      # 第二隐藏层到输出层
        ]
        # 调用父类初始化
        super(MoveFishNet, self).__init__(*layers)
        # 应用权重初始化
        self.apply(weight_init)

def weight_init(m):
    """
    权重初始化函数
    对线性层进行初始化：
    - 权重使用正态分布初始化（均值0，标准差0.1）
    - 偏置初始化为0
    """
    if isinstance(m, nn.Linear):
        # 权重初始化为正态分布
        nn.init.normal_(m.weight, 0, 0.1)
        # 如果有偏置项，初始化为0
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)