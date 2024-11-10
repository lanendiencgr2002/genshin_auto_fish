import torch
from torch import nn
from copy import deepcopy
import numpy as np

def 打印到训练过程txt(内容):
    '''打印内容到txt文件'''
    with open('./训练过程打印.txt', 'a', encoding='utf-8') as f:
        f.write(str(内容) + '\n')

class DQN:
    def __init__(self, base_net, batch_size, n_states, n_actions, memory_capacity=2000, epsilon=0.9, gamma=0.9, rep_frep=100, lr=0.01, reg=False):
        """
        DQN智能体的初始化
        参数说明:
        base_net: 基础神经网络模型
        batch_size: 批次大小
        n_states: 状态空间维度 3个状态
        n_actions: 动作空间维度 2个动作
        memory_capacity: 经验回放池容量
        epsilon: ε-贪婪策略中的探索率
        gamma: 折扣因子
        rep_frep: 目标网络更新频率
        lr: 学习率
        reg: 是否为回归问题（False表示分类问题）
        """
        self.eval_net = base_net # 评估网络
        self.target_net = deepcopy(base_net) # 目标网络

        self.batch_size=batch_size # 批次大小
        self.epsilon=epsilon # ε-贪婪策略中的探索率
        self.gamma=gamma # 折扣因子
        self.n_states=n_states # 状态空间维度
        self.n_actions=n_actions # 动作空间维度
        self.memory_capacity=memory_capacity # 经验回放池容量
        self.rep_frep=rep_frep # 目标网络更新频率
        self.reg=reg # 是否为回归问题

        self.learn_step_counter = 0  # 学习步数计数器
        self.memory_counter = 0  # 经验回放池计数器

        # 经验池的列数取决于4个元素，s, a, r, s_, 总数是N_STATES*2 + 2 + 1 = 3*2 + 2 + 1 = 9
        self.memory = np.zeros((memory_capacity, n_states * 2 + 2 + 1)) # 经验池

        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr) # 优化器
        self.loss_func = nn.MSELoss() # 损失函数

    def choose_action(self, x):
        """
        选择动作的函数
        使用ε-贪婪策略：
        - ε概率选择最优动作（利用）
        - 1-ε概率随机选择动作（探索）
        """
        # 将输入状态转换为PyTorch张量，并增加一个维度作为批次维度
        x = torch.FloatTensor(x).unsqueeze(0)  

        # 使用ε-贪婪策略选择动作
        if np.random.uniform() < self.epsilon:  # epsilon的概率选择最优动作
            # 通过评估网络计算当前状态下所有动作的价值
            actions_value = self.eval_net.forward(x)
            
            # 根据问题类型选择动作：
            # 如果是回归问题(reg=True)，直接返回动作值
            # 如果是分类问题(reg=False)，返回最大值对应的索引作为动作
            action = actions_value if self.reg else torch.argmax(actions_value, dim=1).numpy()
        
        else:  # 1-epsilon的概率随机选择动作
            # 根据问题类型生成随机动作：
            # 回归问题：生成[-1,1]范围内的随机值
            # 分类问题：在可用动作中随机选择一个
            action = np.random.rand(self.n_actions)*2-1 if self.reg else np.random.randint(0, self.n_actions)
        
        return action

    def store_transition(self, s, a, r, s_, done):
        """
        存储转移经验到经验回放池
        s: 当前状态
        a: 执行的动作
        r: 获得的奖励
        s_: 下一状态
        done: 是否结束
        """
        # 将当前状态、动作、奖励、下一状态和是否结束堆叠成一个数组
        transition = np.hstack((s, [a, r], s_, done))  # 水平堆叠这些向量 [1, 2, 0, 1, 3, 4, 0]
        # 如果经验池已满，则使用索引替换旧的记忆  当前计数对经验池容量取余 就是多余出来的经验池索引
        index = self.memory_counter % self.memory_capacity
        # 再用当前经验池索引替换掉多余出来的经验池索引 
        self.memory[index, :] = transition # 选index行的所有数据设置为当前经验
        self.memory_counter += 1 # 经验池计数器加1

    def train_step(self):
        """
        DQN训练步骤的主要组成部分：
        1. 目标网络更新
        2. 经验回放采样
        3. Q值计算
        4. 网络更新
        """
        # 每隔固定步数更新目标网络
        if self.learn_step_counter % self.rep_frep == 0:
            # 将评估网络的参数复制到目标网络
            self.target_net.load_state_dict(self.eval_net.state_dict())
            打印到训练过程txt(f'步骤 {self.learn_step_counter}: 更新目标网络')
        
        self.learn_step_counter += 1 # 学习步数计数器加1

        # 从经验池中随机采样一个批次的数据 一个索引列表
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)
        # 获取采样的经验数据 从索引列表挑一个批次的经验数据  多个一维数组
        b_memory = self.memory[sample_index, :]
        
        # 从经验数据中提取各个部分并转换为张量
        b_s = torch.FloatTensor(b_memory[:, :self.n_states])              # 当前状态  选的是所有数组的前3个
        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_states + 1].astype(int))  # 动作  选的是所有数组中的第3个
        b_r = torch.FloatTensor(b_memory[:, self.n_states + 1:self.n_states + 2])         # 奖励  选的是所有数组中的第4个
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states-1:-1])        # 下一状态  选的是所有数组中的最后3个
        b_d = torch.FloatTensor(b_memory[:, -1])                          # 是否结束  选的是所有数组中的最后一个

        # 计算当前状态-动作对的Q值
        q_eval = self.eval_net(b_s).gather(1, b_a)

        # 计算下一状态的Q值（使用目标网络）
        q_next = self.target_net(b_s_).detach()  # detach()防止反向传播到目标网络

        # 计算目标Q值：奖励 + 折扣因子 * 下一状态的最大Q值
        q_target = b_r + self.gamma * (1-b_d) * q_next.max(dim=1)[0].view(self.batch_size, 1)

        # 计算损失
        loss = self.loss_func(q_eval, q_target)
        
        # 记录训练信息
        if self.learn_step_counter % 100 == 0:  # 每100步记录一次
            打印到训练过程txt(
                f'训练步骤: {self.learn_step_counter}\n'
                f'损失值: {loss.item():.4f}\n'
                f'Q值评估: {q_eval.mean().item():.4f}\n'
                f'Q值目标: {q_target.mean().item():.4f}\n'
                f'经验池大小: {self.memory_counter}\n'
                f'------------------------'
            )

        # 反向传播更新网络
        self.optimizer.zero_grad()  # 清空梯度
        loss.backward()            # 反向传播
        self.optimizer.step()      # 更新参数

class DDQN(DQN):
    """
    双DQN（Double DQN）实现
    主要改进：使用评估网络选择动作，使用目标网络评估动作值
    这样可以减少Q值过估计的问题
    """
    def train_step(self):
        """
        DDQN的训练步骤：
        与DQN的主要区别在于目标Q值的计算方式：
        1. 使用评估网络选择动作
        2. 使用目标网络评估该动作的值
        这种解耦可以减少过度乐观估计
        """
        # update the target network every fixed steps
        if self.learn_step_counter % self.rep_frep == 0:
            # Assign the parameters of eval_net to target_net
            self.target_net.load_state_dict(self.eval_net.state_dict())
            打印到训练过程txt(f'步骤 {self.learn_step_counter}: 更新目标网络')
        self.learn_step_counter += 1

        # Determine the index of Sampled batch from buffer
        sample_index = np.random.choice(self.memory_capacity, self.batch_size)  # randomly select some data from buffer
        # extract experiences of batch size from buffer.
        b_memory = self.memory[sample_index, :]
        # extract vectors or matrices s,a,r,s_ from batch memory and convert these to torch Variables
        # that are convenient to back propagation
        b_s = torch.FloatTensor(b_memory[:, :self.n_states])
        # convert long int type to tensor
        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_states + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.n_states + 1:self.n_states + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states-1:-1])
        b_d = torch.FloatTensor(b_memory[:, -1])

        # calculate the Q value of state-action pair
        q_eval = self.eval_net(b_s).gather(1, b_a)  # (batch_size, 1)
        # calculate the q value of next state
        #q_next = self.target_net(b_s_).detach()  # detach from computational graph, don't back propagate
        # select the maximum q value
        #q_target = b_r + self.gamma * q_next.max(dim=1)[0].view(self.batch_size, 1)  # (batch_size, 1)

        # double DQN
        q_eval_next = self.eval_net(b_s_).detach()
        b_a_ = q_eval_next if self.reg else torch.argmax(q_eval_next, dim=1, keepdim=True)  # get eval_net's argmax_a'(Q(s', a'))
        q_target_next = self.target_net(b_s_).detach()
        if self.reg:
            q_target = b_r + self.gamma * (1-b_d) * q_target_next*b_a_
        else:
            q_target = b_r + self.gamma * (1-b_d) * q_target_next.gather(1, b_a_)

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()  # reset the gradient to zero
        loss.backward()
        self.optimizer.step()  # execute back propagation for one step