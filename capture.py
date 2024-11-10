'''
窗口定位，根据退出按钮定位原神游戏窗口定位坐标，然后捕获钓鱼画面序列，保存为图片
'''
import time
from utils import *
import keyboard
import winsound
import cv2

# 定义常量和图像路径
退出按钮图片 = cv2.imread('./imgs/exit.png') # 退出按钮图片

# 程序初始化
print('程序已准备就绪，按下 T 键开始捕获')
keyboard.wait('t')

# 根据退出按钮定位原神游戏窗口
退出按钮位置 = match_img(cap_raw(), 退出按钮图片)

# 根据退出按钮位置，计算原神游戏窗口的坐标
gvars.genshin_window_rect_img = (
    退出按钮位置[0] - 32,     # x坐标：退出按钮的横坐标往左偏移32像素
    退出按钮位置[1] - 19,     # y坐标：退出按钮的纵坐标往上偏移19像素
    DEFAULT_MONITOR_WIDTH,    # 窗口宽度：使用预设的显示器宽度
    DEFAULT_MONITOR_HEIGHT    # 窗口高度：使用预设的显示器高度
)

# 捕获钓鱼画面序列
for 图片序号 in range(56, 56+20):
    截图 = cap()
    截图.save(f'fish_dataset/{图片序号}.png')
    time.sleep(0.5)

# 完成提示音
winsound.Beep(500, 500)