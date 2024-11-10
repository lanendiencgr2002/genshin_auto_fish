import numpy as np
import torch
from utils import *
import cv2
import time
from copy import deepcopy
from collections import Counter
import traceback
import os
import pygame
import pyautogui
def 播放mp3(mp3文件路径='e.wav'):
    # 转音频 https://www.aconvert.com/cn/audio/
    pygame.mixer.init()
    sound = pygame.mixer.Sound(mp3文件路径)
    channel = pygame.mixer.Channel(0)  # 获取第 0 通道
    channel.play(sound)
    # 使用 channel.get_busy() 来检查音频是否仍在播放
    while channel.get_busy():
        pass
    sound.stop()
    pygame.mixer.quit()

import pyautogui    
import pydirectinput  
import time
class pyautogui操作类:
    def __init__(self, 报错=True, 停顿时间=0):
        if 报错:pyautogui.FAILSAFE = False
        if 停顿时间 != 0:pyautogui.PAUSE = 停顿时间
    def 获取显示器的分辨率(self):return pyautogui.size()
    def 鼠标移动(self,x,y,时间=.3):pyautogui.moveTo(x, y, duration=时间)
    def 获取当前鼠标位置(self):return pyautogui.position()
    def 鼠标按下(self):pyautogui.mouseDown()
    def 鼠标释放(self):pyautogui.mouseUp()
    def 鼠标拖动(self,x,y,时间延时=.3):pyautogui.dragTo(x,y,duration=时间延时)
    def 鼠标滚动(self,鼠标滚动单位:int):pyautogui.scroll(鼠标滚动单位) # 取决单位方向操作系统
    def 鼠标单击(self,x,y,哪个键='左键'):
        if 哪个键=='左键':pyautogui.click(x,y,button='left')
        if 哪个键 == '右键': pyautogui.click(x, y, button='right')
        if 哪个键 == '中间': pyautogui.click(x, y, button='middle')
    def 鼠标双击(self,x,y,哪个键='左键'):
        if 哪个键=='左键':pyautogui.doubleClick(x,y)
        if 哪个键 == '右键': pyautogui.rightClick(x,y)
        if 哪个键 == '中间': pyautogui.middleClick(x,y)
    def 鼠标按方向拖动(self,左右,上下,时间延时=.3):pyautogui.dragRel(左右,上下,duration=时间延时)
    def 按下键(self,键:str):pyautogui.keyDown(键)
    def 释放键(self,键:str):pyautogui.keyUp(键)
    def 按下并释放键(self,键:str):pyautogui.press(键)
    def 键盘文本输入(self,键盘英文文本:str,时间延时=.01):pyautogui.typewrite(键盘英文文本, 时间延时)
    def 键盘列表输入(self,按键列表=['T','i','s','left','left','h',]):pyautogui.typewrite(按键列表)
    def 快捷键按下(self,*args:str):pyautogui.hotkey(args)
    def 获取截图pil对象(self):return pyautogui.screenshot()
    def 保存截图(self,left=999999, top=999999, right=999999, bottom=999999): # 传左上右下 传元组时可以用*打散开来
        if left==999999 or top==999999 or right==999999 or bottom==999999:
            pyautogui.screenshot().save('全屏截图.png')
        else:pyautogui.screenshot(region=(left, top, right - left, bottom - top)).save('区域截图.png')
    def 返回某点像素颜色(self,img,x,y):return img.getpixel((x,y))
    def 获取找图坐标(self,图片路劲):
        try:
            找的图片的位置= pyautogui.locateCenterOnScreen(图片路劲,confidence=0.66)
            return 找的图片的位置
        except:return None
    def 直接按下键(self,键:str):pydirectinput.keyDown(键)
    def 直接释放键(self,键:str):pydirectinput.keyUp(键)
    def 直接按下并释放键(self,键:str):
        pydirectinput.keyDown(键)
        pydirectinput.keyUp(键)

class FishMove:
    """钓鱼动作控制类，负责控制鱼竿的投掷和定位"""
    
    def __init__(self, predictor, fish_type='jia long', show_det=True):
        # 初始化目标检测器
        self.predictor = predictor
        # 设置目标鱼类型
        self.fish_type = fish_type
        # 是否显示检测结果的标志
        self.show_det = show_det
        # 加载鱼咬钩时的图像模板（灰度图）
        self.bite_image = cv2.imread('./imgs/bite.png', cv2.IMREAD_GRAYSCALE)
        # 创建临时图像存储目录
        os.makedirs('img_tmp/', exist_ok=True)
        # 记录开始时间
        self.start_time = None
        # 是否已投掷鱼竿的标志
        self.throw = False

    def reset(self):
        """重置钓鱼状态，按下鼠标开始新的钓鱼回合"""
        # 在屏幕中心按下鼠标
        mouse_down(960, 540)
        # 等待2秒
        time.sleep(2)
        # 重置开始时间
        self.start_time = time.time()
        # 重置投掷状态
        self.throw = False
        # 返回当前状态
        return self._get_state()

    def _is_bite(self):
        """检测鱼是否咬钩"""
        # 截取指定区域的图像
        img = cap(region=[1595, 955, 74, 74], fmt='RGB')
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 使用Canny算子进行边缘检测
        edge_output = cv2.Canny(gray, 50, 150)
        # 通过PSNR值判断是否咬钩
        return psnr(self.bite_image, edge_output) > 10

    def _do_action(self, action):
        """
        执行动作
        Args:
            action: [移动x, 移动y, 是否投掷]的动作向量
        """
        # 如果已经投掷则直接返回
        if self.throw:
            return
        # 判断是否执行投掷动作
        if action[2] > 0.5:
            # 松开鼠标完成投掷
            mouse_up(960, 540)
            self.throw = True
        else:
            # 移动鼠标（相对移动）
            mouse_move(50 * action[0], 50 * action[1])

    def step(self, action):
        # 执行动作
        self._do_action(action)
        # 检查是否超时（20秒）
        done = time.time() - self.start_time >= 20
        # 如果已投掷，等待鱼咬钩
        if self.throw:
            while not self._is_bite():
                if time.time() - self.start_time >= 20:
                    break
                time.sleep(0.2)
            done = True

        # 计算奖励：如果完成则基于剩余时间计算奖励，否则为0
        reward = max(20 - (time.time() - self.start_time), 0) * 5 if done else 0
        # 返回状态、奖励和是否完成
        return torch.zeros(8, dtype=float) if self.throw else self._get_state(), reward, done

    def _get_state(self):
        """获取当前状态"""
        while True:
            # 获取目标检测结果
            obj_list, outputs, img_info = self.predictor.image_det(cap(), with_info=True)
            # 获取鱼竿信息
            rod_info = self._get_rod_info(obj_list)
            # 获取鱼的信息
            fish_info = self._get_fish_info(obj_list, rod_info)
            # 当同时检测到鱼竿和鱼时退出循环
            if rod_info and fish_info:
                break

        # 返回鱼竿和鱼的位置信息组成的张量
        return torch.tensor(rod_info[2] + fish_info[2])

    def _get_rod_info(self, obj_list):
        """获取鱼竿信息"""
        # 筛选出鱼竿的检测结果
        rod_list = [x for x in obj_list if x[0] == 'rod']
        if rod_list:
            # 返回置信度最高的鱼竿检测结果
            return sorted(rod_list, key=lambda x: x[1], reverse=True)[0]
        else:
            # 如果未检测到鱼竿，随机移动鼠标
            mouse_move(np.random.randint(-50, 50), np.random.randint(-50, 50))
            return None

    def _get_fish_info(self, obj_list, rod_info):
        """获取目标鱼的信息"""
        # 筛选出目标鱼类型的检测结果
        fish_list = [x for x in obj_list if x[0] == self.fish_type]
        if fish_list and rod_info:
            # 计算鱼竿中心点坐标
            rod_cx = (rod_info[2][0] + rod_info[2][2]) / 2
            rod_cy = (rod_info[2][1] + rod_info[2][3]) / 2
            # 返回距离鱼竿最近的鱼
            return min(fish_list, key=lambda x: distance((x[2][0] + x[2][2]) / 2, (x[2][1] + x[2][3]) / 2, rod_cx, rod_cy))
        else:
            # 如果未检测到鱼，随机移动鼠标
            mouse_move(np.random.randint(-50, 50), np.random.randint(-50, 50))
            return None

class FishFind:
    """鱼类探测器，用于寻找和识别鱼类"""
    
    def __init__(self, predictor, show_det=True):
        """
        初始化鱼类探测器
        参数:
            predictor: 目标检测器对象
            show_det: 是否显示检测结果的可视化图像
        """
        self.predictor = predictor
        # 加载不同类型鱼饵的图像模板
        self.food_imgs = [
            cv2.imread('./imgs/food_gn.png'),  # 金鱼饵图像
            cv2.imread('./imgs/food_cm.png'),  # 虫子饵图像
            cv2.imread('./imgs/food_bug.png'), # 飞虫饵图像
            cv2.imread('./imgs/food_fy.png'),  # 假饵图像
        ]
        # 定义鱼类与对应饵的映射关系（0:金鱼饵, 1:虫子饵, 2:飞虫饵, 3:假饵）
        self.ff_dict={'hua jiang':0, 'ji yu':1, 'die yu':2, 'jia long':3, 'pao yu':3}
        # 定义不同鱼类所需的投掷距离（像素单位）
        self.dist_dict={'hua jiang':130, 'ji yu':80, 'die yu':80, 'jia long':80, 'pao yu':80}
        # 定义鱼饵选择界面的区域坐标 [x, y, width, height]
        self.food_rgn=[580,400,740,220] 
        # 记录上一次钓的鱼类型  默认是huajiang鱼
        self.last_fish_type='hua jiang'
        # 是否显示检测结果
        self.show_det=show_det
        # 创建临时图像存储目录
        os.makedirs('img_tmp/', exist_ok=True)

    def get_fish_types(self, n=12, rate=0.6):
        """
        扫描周围环境获取可钓鱼类型,用次数映射为余弦函数0-2pi来控制鼠标移动方向
        参数:
            n: 扫描次数
            rate: 识别阈值，某种鱼出现次数/总扫描次数大于此值才会被记录
        返回:
            fish_list: 可钓鱼类型列表
        """
        counter = Counter()  # 用于统计每种鱼出现的次数
        # 定义扫描时鼠标左右移动的方向函数 x：当前扫描次数 n：总扫描次数 
        # x / n // 2 表示当前扫描次数占一半扫描次数的比例  总共的范围为[0,2]
        # np.cos(np.pi * (x / (n // 2)) + 1e-4) 表示当前比例的余弦值 总共的范围为[0,2pi]
        # np.sign(np.cos(np.pi * (x / (n // 2)) + 1e-4)) 表示余弦值的符号，即左右移动的方向
        fx = lambda x: int(np.sign(np.cos(np.pi * (x / (n // 2)) + 1e-4)))
        
        # 先将鼠标向下移动以便扫描
        mouse_move(0, 200)
        time.sleep(0.2)
        
        # 开始扫描
        for i in range(n):
            # 获取目标检测结果
            obj_list = self.predictor.image_det(cap())
            if obj_list is None:  # 如果没有检测到物体
                mouse_move(70 * fx(i), 0)  # 移动鼠标
                time.sleep(0.2)
                continue
                
            # 获取检测到的所有鱼类类型
            cls_list = set([x[0] for x in obj_list])
            counter.update(cls_list)  # 更新计数器
            
            # 水平移动鼠标继续扫描
            mouse_move(70 * fx(i), 0)
            time.sleep(0.2)
            
        # 筛选出出现频率超过阈值的鱼类
        fish_list = [k for k, v in dict(counter).items() if v / n >= rate]
        return fish_list

    def throw_rod(self, fish_type):
        """
        投掷鱼竿到目标鱼的位置 目标检测鱼竿（没有检测到则随机移动） 找最近的目标鱼 根据鱼的位置计算需要移动的距离
        参数:
            fish_type: 目标鱼类型
        """
        # 按下鼠标开始瞄准
        mouse_down(960, 540)
        time.sleep(1)

        # 根据距离 计算鼠标移动速度的函数
        def move_func(dist):
            if dist > 100:  # 如果距离大于100像素
                # 快速移动：返回50或-50
                # np.sign(dist)返回距离的正负号：
                # - 如果dist > 0，返回+1
                # - 如果dist < 0，返回-1
                return 50 * np.sign(dist)
            else:  # 如果距离小于100像素
                # 慢速移动：速度与距离成正比
                # abs(dist)/2.5 + 10 确保最小速度是10像素
                # np.sign(dist)保持正确的移动方向
                return (abs(dist)/2.5 + 10) * np.sign(dist)

        # 尝试50次瞄准 目标检测鱼竿（没有检测到则随机移动） 找最近的目标鱼 根据鱼的位置计算需要移动的距离
        for i in range(50):
            try:
                # 获取目标检测结果
                obj_list, outputs, img_info = self.predictor.image_det(cap(), with_info=True)
                
                # 如果需要显示检测结果，保存检测图像
                if self.show_det:
                    cv2.imwrite(f'img_tmp/det{i}.png', self.predictor.visual(outputs[0],img_info))

                # 获取置信度最高的鱼竿位置信息
                rod_info = sorted(list(filter(lambda x: x[0] == 'rod', obj_list)), key=lambda x: x[1], reverse=True)
                if len(rod_info)<=0:  # 如果没检测到鱼竿，随机移动
                    mouse_move(np.random.randint(-50,50), np.random.randint(-50,50))
                    time.sleep(0.1)
                    continue
                    
                rod_info=rod_info[0]
                # 计算鱼竿中心点坐标
                rod_cx = (rod_info[2][0] + rod_info[2][2]) / 2
                rod_cy = (rod_info[2][1] + rod_info[2][3]) / 2

                # 找到距离鱼竿最近的目标鱼
                fish_info = min(list(filter(lambda x: x[0] == fish_type, obj_list)),
                                key=lambda x: distance((x[2][0]+x[2][2])/2, (x[2][1]+x[2][3])/2, rod_cx, rod_cy))

                # 根据鱼的位置计算需要移动的距离
                if (fish_info[2][0] + fish_info[2][2]) > (rod_info[2][0] + rod_info[2][2]):
                    x_dist = fish_info[2][0] - self.dist_dict[fish_type] - rod_cx
                else:
                    x_dist = fish_info[2][2] + self.dist_dict[fish_type] - rod_cx

                # 如果位置已经足够接近，结束瞄准
                if abs(x_dist)<30 and abs((fish_info[2][3] + fish_info[2][1]) / 2 - rod_info[2][3])<30:
                    break

                # 移动鼠标调整瞄准位置
                dx = int(move_func(x_dist))
                dy = int(move_func(((fish_info[2][3]) + fish_info[2][1]) / 2 - rod_info[2][3]))
                mouse_move(dx, dy)
            except Exception as e:
                traceback.print_exc()
                
        # 松开鼠标，投掷鱼竿
        mouse_up(960, 540)

    def select_food(self, fish_type):
        """
        选择对应鱼类的鱼饵
        参数:
            fish_type: 目标鱼类型
        """
        # 打开鱼饵选择菜单
        mouse_click(1650, 790, button=MOUSE_RIGHT)
        time.sleep(1.5)
        # 获取菜单区域图像
        img=cap(self.food_rgn, fmt='RGB')
        # 匹配对应的鱼饵图标位置
        bbox_food = match_img(img, self.food_imgs[self.ff_dict[fish_type]], type=cv2.TM_SQDIFF_NORMED)
        print('匹配鱼饵对应位置',bbox_food)
        # 点击选择鱼饵
        # 没用没反应已注释
        # mouse_click(bbox_food[4]+self.food_rgn[0], bbox_food[5]+self.food_rgn[1])
        鼠标操作=pyautogui操作类()
        鼠标操作.鼠标单击(bbox_food[4]+self.food_rgn[0], bbox_food[5]+self.food_rgn[1])
        播放mp3('aa.mp3')
        time.sleep(1.5)
        # 确认选择
        鼠标操作.鼠标单击(1183, 756)
        # 没用没反应已注释
        # mouse_click(1183, 756)

    def do_fish(self, fish_init=True) -> bool:
        """
        执行完整的钓鱼流程，扫描可钓鱼类型，选择鱼饵，投掷鱼竿
        参数:
            fish_init: 是否需要重新扫描鱼类 默认是True
        返回:
            bool: 是否成功开始钓鱼
        """
        # 如果需要，扫描可钓鱼类型
        if fish_init:
            self.fish_list = self.get_fish_types()
            print("可钓的鱼类：",self.fish_list)

        # 如果没有可钓的鱼，返回False
        if not self.fish_list:
            print("没有可钓的鱼")
            return False
        
        # 如果鱼类型变化，更换鱼饵
        if self.fish_list[0]!=self.last_fish_type:
            print("更换鱼饵")
            self.select_food(self.fish_list[0])
            self.last_fish_type = self.fish_list[0]
            
        # 投掷鱼竿
        print("丢鱼竿")
        self.throw_rod(self.fish_list[0])

        return True
    

class Fishing:
    '''
    钓鱼环境类，初始化会根据退出标志定位画面范围
    '''
    def __init__(self, delay=0.1, max_step=100, show_det=True, predictor=None):
        # delay：每0.1秒检查一次鱼是否上钩
        # max_step：最大尝试次数为100次
        # show_det：是否显示检测结果
        # predictor：目标检测器
        当前目录 = os.path.dirname(os.path.abspath(__file__))
        # 加载所需的图像模板
        self.t_l = cv2.imread('./imgs/target_left.png')   # 左侧目标图像
        self.t_r = cv2.imread('./imgs/target_right.png')  # 右侧目标图像
        self.t_n = cv2.imread('./imgs/target_now.png')    # 当前位置图像
        self.im_bar = cv2.imread('./imgs/bar2.png')       # 进度条图像
        self.bite = cv2.imread('./imgs/bite.png', cv2.IMREAD_GRAYSCALE)    # 咬钩提示图像
        self.fishing = cv2.imread('./imgs/fishing.png', cv2.IMREAD_GRAYSCALE)  # 钓鱼状态图像
        self.exit = cv2.imread('./imgs/exit.png')         # 退出按钮图像

        # 游戏参数设置
        self.std_color=np.array([192,255,255])  # 标准颜色值
        self.r_ring=21                          # 圆环半径
        self.delay=delay                        # 延迟时间
        self.max_step=max_step                  # 最大步数
        self.count=0                            # 计数器
        self.show_det=show_det                  # 是否显示检测结果

        # 根据退出标志定位画面范围
        exit_pos = match_img(cap_raw(), self.exit)
        gvars.genshin_window_rect_img = (exit_pos[0] - 32, exit_pos[1] - 19, DEFAULT_MONITOR_WIDTH, DEFAULT_MONITOR_HEIGHT)

         # 用于图像处理的偏移向量：[x1偏移, y1偏移, x2偏移, y2偏移, cx偏移, cy偏移]
        self.add_vec=[0,2,0,2,0,2]

    def is_fishing(self):
        """检测是否处于钓鱼状态"""
        # 截取指定区域的图像并转换为灰度图  1568, 951 原来的1595, 955
        img = cap(region=[1568, 951, 74, 74],fmt='RGB')
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # 使用Canny边缘检测
        edge_output = cv2.Canny(gray, 50, 150)
        # 通过PSNR值判断是否在钓鱼
        return psnr(self.fishing, edge_output)>10

    def reset(self):
        """重置钓鱼状态"""
        # 获取进度条的起始位置
        self.y_start = self.find_bar()[0]
        # 截取包含进度条的图像区域
        self.img=cap([712 - 10, self.y_start, 496 + 20, 103])

        # 重置各项状态
        # 是否开始钓鱼
        self.fish_start=False
        # 连续0次得分的次数
        self.zero_count=0
        # 步数
        self.step_count=0
        # 奖励
        self.reward=0
        # 上一次得分
        self.last_score=self.get_score()

        return self.get_state()

    def drag(self):
        """模拟拉竿动作"""
        # 鼠标点击拉竿位置
        mouse_click_raw(1630,995)

    def do_action(self, action):
        """执行动作 只有1个动作 1：拉竿"""
        if action==1:
            self.drag()

    def scale(self, x):
        """将像素坐标转换为标准化坐标"""
        return (x-5-10)/484

    def find_bar(self, img=None):
        """定位进度条位置"""
        # 如果没有提供图像，则截取屏幕指定区域
        img = cap(region=[700, 0, 520, 300]) if img is None else img[:300, 700:700+520, :]
        # 匹配进度条模板
        bbox_bar = match_img(img, self.im_bar)
        # 如果需要显示检测结果，保存标记后的图像
        if self.show_det:
            img=deepcopy(img)
            cv2.rectangle(img, bbox_bar[:2], bbox_bar[2:4], (0, 0, 255), 1)  # 画出矩形位置
            cv2.imwrite(f'../img_tmp/bar.jpg', img)
        return bbox_bar[1]-9, bbox_bar

    def is_bite(self):
        """检测鱼是否咬钩"""
        # 截取指定区域的图像
        img = cap(region=[1568, 951, 74, 74],fmt='BGR')
        # 转换为灰度图并进行边缘检测
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edge_output = cv2.Canny(gray, 50, 150)
        print('检查是否读取到了咬钩图像，和模板图',self.bite.shape,edge_output.shape,psnr(self.bite, edge_output))
        # 通过PSNR值判断是否咬钩
        return psnr(self.bite, edge_output)>8.5

    def get_state(self, all_box=False):
        """返回当前状态的各种坐标，左上角各种等等：左侧，右侧，当前位置"""
        # 截取进度条区域的图像
        bar_img=self.img[2:34,:,:]
        # 匹配左侧、右侧和当前位置的标记
        bbox_l = match_img(bar_img, self.t_l)
        bbox_r = match_img(bar_img, self.t_r)
        bbox_n = match_img(bar_img, self.t_n)

        # 添加偏移量  [x1偏移, y1偏移, x2偏移, y2偏移, cx偏移, cy偏移]
        bbox_l = tuple(list_add(bbox_l, self.add_vec))
        bbox_r = tuple(list_add(bbox_r, self.add_vec))
        bbox_n = tuple(list_add(bbox_n, self.add_vec))

        # 如果需要显示检测结果，保存标记后的图像
        if self.show_det:
            img=deepcopy(self.img)
            # 画出各个标记的位置 左上 右下 颜色 宽度 [:2]表示 [0,1] 表示左上坐标 [2:4]表示 [2,3]表示右下坐标
            cv2.rectangle(img, bbox_l[:2], bbox_l[2:4], (255, 0, 0), 1)
            cv2.rectangle(img, bbox_r[:2], bbox_r[2:4], (0, 255, 0), 1)
            cv2.rectangle(img, bbox_n[:2], bbox_n[2:4], (0, 0, 255), 1)
            # 添加分数和奖励值的文本
            fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL # 字体
            fontScale = 1 # 字体大小
            thickness = 1 # 字体宽度
            cv2.putText(img, str(self.last_score), (257+30, 72), fontScale=fontScale,fontFace=fontFace, thickness=thickness, color=(0,255,255))
            cv2.putText(img, str(self.reward), (257+30, 87), fontScale=fontScale,fontFace=fontFace, thickness=thickness, color=(255,255,0))
            cv2.imwrite(f'./img_tmp/{self.count}.jpg',img)
        self.count+=1

        # 返回状态信息
        if all_box: # 返回所有坐标
            return bbox_l, bbox_r, bbox_n
        else: # 返回缩放中心横坐标对应的百分比的坐标 左侧，右侧，当前位置
            return self.scale(bbox_l[4]),self.scale(bbox_r[4]),self.scale(bbox_n[4])

    def get_score(self):
        """计算当前得分"""
        '''
              90°
               |
        180° - + - 0°
               |
              270°
        - 圆环从0度开始逆时针旋转
        - 每检查到颜色变化的点就计算对应的得分
        - 得分 = 角度/2 - 2
        '''
        # 设置圆环中心点坐标
        cx,cy=247+10,72
        # 遍历圆环上的点
        for x in range(4,360,2):
            # 计算圆环上点的坐标
            px=int(cx+self.r_ring*np.sin(np.deg2rad(x)))
            py=int(cy-self.r_ring*np.cos(np.deg2rad(x)))
            # 检查颜色是否符合标准
            if np.mean(np.abs(self.img[py,px,:]-self.std_color))>5:
                return x//2-2
        return 360//2-2

    def step(self, action):
        """执行一步动作"""
        # 执行动作
        self.do_action(action)

        # 等待一段时间并获取新的状态
        time.sleep(self.delay-0.05)
        self.img=cap([712 - 10, self.y_start, 496 + 20, 103],fmt='RGB')
        # 步数+1
        self.step_count+=1

        # 计算得分和奖励 根据圆环进度条计算得分
        score=self.get_score()
        # 如果得分大于0，则开始钓鱼
        if score>0:
            self.fish_start=True
            self.zero_count=0
        else: # 否则连续0次得分的次数+1
            self.zero_count+=1
        # 奖励 = 当前得分 - 上一次得分  因为是根据圆环进度条计算得分，所以奖励是当前得分减去上一次得分
        self.reward=score-self.last_score
        # 更新上一次得分
        self.last_score=score

        # 返回新状态、奖励 和是否结束
        return self.get_state(), self.reward, (
            self.step_count > self.max_step or  # 条件1：步数超过最大步数
            (self.zero_count >= 15 and self.fish_start) or  # 条件2：已开始钓鱼且连续15次得分为0
            score > 176  # 条件3：得分超过176
        )
    def render(self):
        """渲染函数（此处未实现）"""
        pass


class Fishing_sim:
    """钓鱼模拟器，用于模拟钓鱼过程的物理环境"""
    
    def __init__(self, bar_range=(0.18, 0.4), move_range=(30,60*2), resize_freq_range=(15,60*5),
                 move_speed_range=(-0.3,0.3), tick_count=60, step_tick=15, stop_tick=60*15,
                 drag_force=0.4, down_speed=0.015, stable_speed=-0.32, drawer=None):
        """
        初始化钓鱼模拟器
        参数:
            bar_range: 进度条长度范围
            move_range: 移动范围
            resize_freq_range: 调整大小的频率范围
            move_speed_range: 移动速度范围
            tick_count: 时钟计数
            step_tick: 每步的时钟数
            stop_tick: 停止时钟数
            drag_force: 拉力大小
            down_speed: 下降速度
            stable_speed: 稳定速度
            drawer: 绘图器对象
        """
        # 初始化各种参数
        self.bar_range = bar_range # 进度条长度范围
        self.move_range = move_range # 移动范围
        self.resize_freq_range = resize_freq_range # 调整大小的频率范围
        # 将移动速度范围按时钟计数归一化
        self.move_speed_range = (move_speed_range[0]/tick_count, move_speed_range[1]/tick_count)
        self.tick_count = tick_count # 时钟计数

        self.step_tick = step_tick # 每步的时钟数
        self.stop_tick = stop_tick # 停止时钟数
        # 将力和速度按时钟计数归一化
        self.drag_force = drag_force/tick_count # 拉力大小
        self.down_speed = down_speed/tick_count # 下降速度
        self.stable_speed = stable_speed/tick_count # 稳定速度

        self.drawer = drawer # 绘图器对象

        # 重置环境状态
        self.reset()

    def reset(self):
        """重置模拟器状态"""
        # 随机初始化进度条长度和位置
        self.len = np.random.uniform(*self.bar_range)
        self.low = np.random.uniform(0, 1-self.len)
        self.pointer = np.random.uniform(0, 1)
        self.v = 0  # 初始速度为0

        # 重置计时器
        self.resize_tick = 0
        self.move_tick = 0
        self.move_speed = 0

        # 重置得分和总时钟数
        self.score = 100
        self.ticks = 0

        return (self.low, self.low+self.len, self.pointer)

    def drag(self):
        """执行拉竿动作"""
        self.v = self.drag_force

    def move_bar(self):
        """移动进度条"""
        # 当移动计时结束时，重新设置移动参数
        if self.move_tick <= 0:
            self.move_tick = np.random.uniform(*self.move_range)
            self.move_speed = np.random.uniform(*self.move_speed_range)
        # 移动进度条位置并确保在有效范围内
        self.low = np.clip(self.low+self.move_speed, a_min=0, a_max=1-self.len)
        self.move_tick -= 1

    def resize_bar(self):
        """调整进度条大小"""
        # 当调整计时结束时，重新设置进度条大小
        if self.resize_tick <= 0:
            self.resize_tick = np.random.uniform(*self.resize_freq_range)
            self.len = min(np.random.uniform(*self.bar_range), 1-self.low)
        self.resize_tick -= 1

    def tick(self):
        """在一步中更新一个时钟周期（得分动作指针位置速度等等）返回是否结束"""
        self.ticks += 1 # 时钟计数+1
        # 根据指针是否在进度条范围内更新得分
        if self.low < self.pointer < self.low+self.len:
            self.score += 1
        else:
            self.score -= 1

        # 检查是否需要结束
        if self.ticks > self.stop_tick or self.score <= -100000:
            return True

        # 更新指针位置和速度
        self.pointer += self.v
        self.pointer = np.clip(self.pointer, a_min=0, a_max=1)
        self.v = max(self.v-self.down_speed, self.stable_speed)

        # 更新进度条位置和大小
        self.move_bar()
        self.resize_bar()
        return False

    def do_action(self, action):
        """执行动作"""
        if action == 1:
            self.drag()

    def get_state(self):
        """获取当前状态"""
        return self.low, self.low+self.len, self.pointer

    def step(self, action):
        """执行一步模拟"""
        # 执行动作
        self.do_action(action)
        # 是否结束  
        done = False
        # 之前的得分
        score_before = self.score
        # 在一步中执行多个时钟周期
        for x in range(self.step_tick):
            if self.tick(): # 如果结束
                done = True # 设置结束标志
        # 返回新状态、奖励和是否结束
        return self.get_state(), (self.score-score_before)/self.step_tick, done

    def render(self):
        """渲染当前状态"""
        if self.drawer:
            self.drawer.draw(self.low, self.low+self.len, self.pointer, self.ticks)

