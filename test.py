import keyboard
import winsound
from fisher.models import FishNet
from fisher.environment import *
import torch
import argparse

# 创建命令行参数解析器，用于原神钓鱼DQN测试
parser = argparse.ArgumentParser(description='原神自动钓鱼DQN测试程序')
# 设置状态空间维度
parser.add_argument('--n_states', default=3, type=int)
# 设置动作空间维度
parser.add_argument('--n_actions', default=2, type=int)
# 设置步进时间间隔
parser.add_argument('--step_tick', default=12, type=int)
# 设置模型权重文件路径
parser.add_argument('--model_dir', default='./weights/fish_genshin_net.pth', type=str)
参数 = parser.parse_args()

if __name__ == '__main__':
    # 初始化神经网络模型
    钓鱼网络 = FishNet(in_ch=参数.n_states, out_ch=参数.n_actions)
    # 初始化钓鱼环境
    钓鱼环境 = Fishing(delay=0.1, max_step=10000, show_det=True)

    # 加载预训练模型权重
    钓鱼网络.load_state_dict(torch.load(参数.model_dir))
    钓鱼网络.eval()

    while True:
        # 发出开始提示音
        winsound.Beep(500, 500)
        # 等待按下'r'键开始钓鱼
        keyboard.wait('r')
        # 等待鱼上钩
        while True:
            if 钓鱼环境.is_bite():
                break
            time.sleep(0.5)
        # 发出鱼上钩提示音
        winsound.Beep(700, 500)
        # 开始拉钩
        钓鱼环境.drag()
        time.sleep(1)

        # 重置环境状态
        当前状态 = 钓鱼环境.reset()
        # 开始钓鱼循环
        for i in range(10000):
            # 将状态转换为张量格式
            当前状态 = torch.FloatTensor(当前状态).unsqueeze(0)
            # 通过网络预测动作
            动作 = 钓鱼网络(当前状态)
            动作 = torch.argmax(动作, dim=1).numpy()
            # 执行动作并获取新状态
            当前状态, 奖励, 完成 = 钓鱼环境.step(动作)
            if 完成:
                break