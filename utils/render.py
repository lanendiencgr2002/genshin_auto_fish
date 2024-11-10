import sys
import numpy as np
from matplotlib import pyplot as plt

class PltRender:
    """图形化渲染器类"""
    def __init__(self, 回调函数=None):
        # 创建图形窗口，设置大小为6x1.5
        self.图形 = plt.figure(figsize=(6, 1.5))
        plt.ion()  # 开启交互模式 允许图形实时更新而不阻塞程序执行
        plt.tight_layout()  # 自动调整布局
        self.回调函数 = 回调函数

    def draw(self, 最小值, 最大值, 指针位置, 计数):
        宽度, 高度 = 300, 50
        # 创建空白图像
        图像 = np.zeros((高度, 宽度, 3), np.uint8)
        # 绘制最小值标记（红色）
        图像[:, int(最小值 * 宽度) - 3:int(最小值 * 宽度) + 3, :] = np.array([255, 0, 0])
        # 绘制指针位置（绿色）
        图像[:, int(指针位置 * 宽度) - 3:int(指针位置 * 宽度) + 3, :] = np.array([0, 255, 0])
        # 绘制最大值标记（蓝色）
        图像[:, int(最大值 * 宽度) - 3:int(最大值 * 宽度) + 3, :] = np.array([0, 0, 255])

        plt.imshow(图像)
        plt.title(f'计数:{计数}')
        if self.回调函数:
            self.回调函数()
        plt.pause(0.0001)
        plt.clf()

class CliRender:
    """命令行渲染器类"""
    def __init__(self):
        pass

    def draw(self, 最小值, 最大值, 指针位置, 计数):
        # 创建进度条
        进度条 = [' '] * 101
        进度条[int(最小值 * 100)] = '|'
        进度条[int(最大值 * 100)] = '|'
        进度条[int(指针位置 * 100)] = '+'
        显示文本 = f'计数:{计数}[' + ''.join(进度条) + ']'
        # 输出到控制台
        sys.stdout.write(显示文本)
        sys.stdout.flush()
        # 回退光标
        sys.stdout.write('\b' * len(显示文本))

if __name__ == "__main__":
    import time
    def 测试渲染器(渲染器):
        print(f"测试 {渲染器.__class__.__name__}")
        # 模拟钓鱼进度条移动
        for i in range(100):
            # 模拟进度条位置
            指针位置 = (i % 100) / 100.0
            最小值 = 0.3
            最大值 = 0.7
            计数 = i
            
            # 调用渲染器绘制
            渲染器.draw(最小值, 最大值, 指针位置, 计数)
            time.sleep(0.05)  # 暂停一小段时间，便于观察
    
    # 测试图形界面渲染器
    print("开始测试图形界面渲染器...")
    plt渲染器 = PltRender()
    测试渲染器(plt渲染器)
    
    # 测试命令行渲染器
    print("\n开始测试命令行渲染器...")
    命令行渲染器 = CliRender()
    测试渲染器(命令行渲染器)