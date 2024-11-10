from fisher.agent import DQN
from fisher.models import FishNet
from fisher.environment import *
import torch
import argparse
import os
from utils.render import *

parser = argparse.ArgumentParser(description='原神钓鱼模拟训练程序（基于DQN）')
parser.add_argument('--batch_size', default=32, type=int)  # 批次大小
parser.add_argument('--n_states', default=3, type=int)     # 状态空间维度
parser.add_argument('--n_actions', default=2, type=int)    # 动作空间维度
parser.add_argument('--step_tick', default=12, type=int)   # 时间步长
parser.add_argument('--n_episode', default=400, type=int)  # 训练回合数
parser.add_argument('--save_dir', default='./output', type=str)  # 模型保存路径
parser.add_argument('--resume', default=None, type=str)    # 继续训练的模型路径
参数 = parser.parse_args()

if not os.path.exists(参数.save_dir):
    os.makedirs(参数.save_dir)

网络 = FishNet(in_ch=参数.n_states, out_ch=参数.n_actions)

if 参数.resume:  # 如果提供了resume参数（即模型路径）
    网络.load_state_dict(torch.load(参数.resume))  # 加载已有的模型权重

智能体 = DQN(网络, 参数.batch_size, 参数.n_states, 参数.n_actions, memory_capacity=2000)
环境 = Fishing_sim(step_tick=参数.step_tick, drawer=PltRender())

def 训练():
    '''
    状态 → 选择动作 → 执行动作 → 获得奖励 → 存储经验 → 训练网络 → 更新状态 → 循环
    状态：左右指针和当前指针的百分比位置
    动作：0-1 是否点击
    奖励机制：如果在进度条范围内 得分=上次得分+1 否则得分=上次得分-1
    模型输入：状态（3维）
    模型输出：动作（2维）
    '''
    print("\n开始收集经验...")
    打印到训练过程txt('开始收集经验...')
    网络.train()  # 设置为训练模式
    for 回合 in range(参数.n_episode):
        打印到训练过程txt(f'环境重置，回合: {回合}')
        状态 = 环境.reset() # 状态是左右进度条百分比位置和指针位置
        回合奖励 = 0
        while True:
            # 在训练后期每20回合渲染一次
            if 回合 > 200 and 回合 % 20 == 0:
                pass
                # 环境.render()
            # 根据当前状态选择动作
            动作 = 智能体.choose_action(状态)
            打印到训练过程txt('状态：'+str(状态))
            打印到训练过程txt('动作：'+str(动作))
            # 执行动作并获取下一状态、奖励和是否结束
            下一状态, 奖励, 结束 = 环境.step(动作)
            打印到训练过程txt('下一状态：'+str(下一状态))
            打印到训练过程txt('奖励：'+str(奖励))
            打印到训练过程txt('结束：'+str(结束))
            # 存储经验到回放缓冲区
            智能体.store_transition(状态, 动作, 奖励, 下一状态, int(结束))
            回合奖励 += 奖励
            # 当经验回放缓冲区填满后，开始训练网络
            if 智能体.memory_counter > 智能体.memory_capacity:
                智能体.train_step()
                if 结束:
                    print('回合: ', 回合, ' |', '回合奖励: ', round(回合奖励, 2))
            if 结束:
                break
            状态 = 下一状态
    # 保存训练好的模型
    torch.save(网络.state_dict(), os.path.join(参数.save_dir, f'钓鱼模型_{回合}.pth'))

def 打印到训练过程txt(内容):
    '''打印内容到txt文件'''
    with open('./经验收集过程打印.txt', 'a', encoding='utf-8') as f:
        f.write(str(内容) + '\n')
    

    


if __name__ == '__main__':
    训练()