# 导入必要的库
from utils.render import *
from fisher.models import FishNet
from fisher.environment import *
import torch
import argparse
from matplotlib.animation import FFMpegWriter

# 设置命令行参数解析器
parser = argparse.ArgumentParser(description='原神自动钓鱼DQN测试程序')
parser.add_argument('--n_states', default=3, type=int, help='状态空间维度')
parser.add_argument('--n_actions', default=2, type=int, help='动作空间维度')
parser.add_argument('--step_tick', default=12, type=int, help='每步时间间隔')
parser.add_argument('--model_dir', default='./output/fish_sim_net_399.pth', type=str, help='模型保存路径')
args = parser.parse_args()

if __name__ == '__main__':
    # 初始化视频写入器
    视频写入器 = FFMpegWriter(fps=60)  # 设置帧率为60fps
    渲染器 = PltRender(call_back=视频写入器.grab_frame)

    # 初始化神经网络和环境
    钓鱼网络 = FishNet(in_ch=args.n_states, out_ch=args.n_actions)
    钓鱼环境 = Fishing_sim(step_tick=args.step_tick, drawer=渲染器, stop_tick=10000)

    # 加载预训练模型
    钓鱼网络.load_state_dict(torch.load(args.model_dir))

    # 设置为评估模式
    钓鱼网络.eval()
    当前状态 = 钓鱼环境.reset()

    # 开始测试并录制视频
    with 视频写入器.saving(渲染器.fig, 'out.mp4', 100):
        for i in range(2000):  # 最多进行2000步
            钓鱼环境.render()  # 渲染当前状态

            # 将状态转换为张量并预测动作
            状态张量 = torch.FloatTensor(当前状态).unsqueeze(0)
            动作预测 = 钓鱼网络(状态张量)
            选择动作 = torch.argmax(动作预测, dim=1).numpy()
            
            # 执行动作并获取新状态
            当前状态, 奖励, 结束标志 = 钓鱼环境.step(选择动作)
            if 结束标志:
                break