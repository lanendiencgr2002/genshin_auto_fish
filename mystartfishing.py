#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# 原神自动钓鱼程序 - 主程序文件

# 导入必要的库
import argparse  # 用于解析命令行参数
import os       # 用于处理文件路径
import time     # 用于时间相关操作
import cv2      # 用于图像处理
import pyautogui # 用于截图
from loguru import logger  # 用于日志记录

import torch        # 深度学习框架
import keyboard     # 用于键盘输入监控
import winsound     # 用于播放系统声音

# 导入YOLOX相关模块
from yolox.exp import get_exp                    # 获取实验配置
from yolox.utils import fuse_model, get_model_info  # 模型融合和信息获取工具

# 导入自定义钓鱼模块
from fisher.environment import *   # 钓鱼环境相关类
from fisher.predictor import *     # 预测器相关类
from fisher.models import FishNet  # 钓鱼网络模型
import pygame
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
# 创建命令行参数解析器
def make_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser("原神自动钓鱼程序")
    # 基础参数配置
    parser.add_argument("demo", default="image", help="演示类型，可选：图像、视频、摄像头")
    parser.add_argument("-expn", "--experiment-name", type=str, default=None, help="实验名称")
    parser.add_argument("-n", "--name", type=str, default=None, help="模型名称")
    parser.add_argument("--path", default="./assets/dog.jpg", help="图像或视频路径")

    # 模型相关参数
    parser.add_argument("-f", "--exp_file", default=None, type=str, help="实验配置文件路径")
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="模型检查点文件路径")
    parser.add_argument("--device", default="cpu", type=str, help="运行设备，可选：cpu 或 gpu")
    
    # 模型推理参数
    parser.add_argument("--conf", default=0.3, type=float, help="检测置信度阈值")
    parser.add_argument("--nms", default=0.3, type=float, help="非极大值抑制阈值")
    parser.add_argument("--tsize", default=None, type=int, help="测试图片大小")
    
    # 高级选项
    parser.add_argument("--fp16", dest="fp16", default=False, action="store_true", 
                       help="是否使用混合精度评估")
    parser.add_argument("--legacy", dest="legacy", default=False, action="store_true",
                       help="是否兼容旧版本")
    parser.add_argument("--fuse", dest="fuse", default=False, action="store_true",
                       help="是否融合卷积和批归一化层")
    parser.add_argument("--trt", dest="trt", default=False, action="store_true",
                       help="是否使用 TensorRT 模型")

    # DQN 智能体参数
    parser.add_argument('--n_states', default=3, type=int, help="状态空间维度")
    parser.add_argument('--n_actions', default=2, type=int, help="动作空间维度")
    parser.add_argument('--step_tick', default=12, type=int, help="动作执行间隔")
    parser.add_argument('--model_dir', default='./weights/fish_genshin_net.pth', 
                       type=str, help="智能体模型路径")

    return parser
def 自己的main(exp,args):
    """
    主程序入口函数
    参数：
        exp: 实验配置对象
        args: 命令行参数对象
    """
    # 设置实验名称
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    # TensorRT模式下强制使用GPU
    if args.trt:
        args.device = "gpu"

    logger.info("当前参数配置: {}".format(args))

    # 更新实验配置
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    # 初始化模型
    model = exp.get_model()
    logger.info("模型结构概要: {}".format(get_model_info(model, exp.test_size)))

    # 设置设备和精度
    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # 转换为半精度
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        if args.ckpt is None:
            trt_file = os.path.join(file_name, "model_trt.pth")
        else:
            trt_file = args.ckpt
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, FISH_CLASSES, trt_file, decoder, args.device, args.fp16, args.legacy)

    agent = FishNet(in_ch=args.n_states, out_ch=args.n_actions)
    agent.load_state_dict(torch.load(args.model_dir))
    agent.eval()

    print('INIT OK')
    # while True:
    #     # 等待按下"r"键开始钓鱼
    #     print('Waiting for "r" to perform fishing')
    #     # 播放提示音
    #     winsound.Beep(500, 500)
    #     # 等待按下"r"键
    #     keyboard.wait('r')
    #     # 播放提示音
    #     winsound.Beep(500, 500)
    #     # 根据演示类型执行钓鱼流程
    #     if args.demo == "image":
    #         start_fishing(predictor, agent)

    # 改为自动钓鱼一次
    # 等待按下"r"键开始钓鱼
    print('Waiting for "r" to perform fishing')
    # 等待按下"r"键
    keyboard.wait('r')
    # 播放提示音
    播放mp3("aa.mp3")
    # 根据演示类型执行钓鱼流程
    if args.demo == "image":
        自己的钓鱼(predictor, agent)
# 主程序入口
def main(exp, args):
    """
    主程序入口函数
    参数：
        exp: 实验配置对象
        args: 命令行参数对象
    """
    # 设置实验名称
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    # TensorRT模式下强制使用GPU
    if args.trt:
        args.device = "gpu"

    logger.info("当前参数配置: {}".format(args))

    # 更新实验配置
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    # 初始化模型
    model = exp.get_model()
    logger.info("模型结构概要: {}".format(get_model_info(model, exp.test_size)))

    # 设置设备和精度
    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # 转换为半精度
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        if args.ckpt is None:
            trt_file = os.path.join(file_name, "model_trt.pth")
        else:
            trt_file = args.ckpt
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, FISH_CLASSES, trt_file, decoder, args.device, args.fp16, args.legacy)

    agent = FishNet(in_ch=args.n_states, out_ch=args.n_actions)
    agent.load_state_dict(torch.load(args.model_dir))
    agent.eval()

    print('INIT OK')
    # while True:
    #     # 等待按下"r"键开始钓鱼
    #     print('Waiting for "r" to perform fishing')
    #     # 播放提示音
    #     winsound.Beep(500, 500)
    #     # 等待按下"r"键
    #     keyboard.wait('r')
    #     # 播放提示音
    #     winsound.Beep(500, 500)
    #     # 根据演示类型执行钓鱼流程
    #     if args.demo == "image":
    #         start_fishing(predictor, agent)

    # 改为自动钓鱼一次
    # 等待按下"r"键开始钓鱼
    print('Waiting for "r" to perform fishing')
    # 等待按下"r"键
    keyboard.wait('r')
    # 播放提示音
    播放mp3("aa.mp3")
    # 根据演示类型执行钓鱼流程
    if args.demo == "image":
        start_fishing(predictor, agent)

# 开始钓鱼流程
def start_fishing(预测器, 智能体, 上钩超时=45):
    """
    钓鱼主循环函数
    参数：
        预测器: 用于检测鱼类的模型
        智能体: 控制钓鱼操作的DQN网络
        上钩超时: 等待鱼上钩的最大时间（秒）
    """
    钓鱼探测器 = FishFind(预测器)                          # 初始化钓鱼探测器
    # delay=0.1 表示每0.1秒检查一次鱼是否上钩
    # max_step=10000 表示最大尝试次数为10000次
    # show_det=True 表示显示检测结果
    钓鱼环境 = Fishing(delay=0.1, max_step=10000, show_det=True)  # 初始化钓鱼环境

    尝试次数 = 0

    '''
    钓鱼主循环 
    '''
    while True:
        继续标志 = False
        # 超过最大尝试次数处理
        if 尝试次数 > 4:
            播放mp3('./搞错了再来.mp3')
            尝试次数 = 0              # 重置尝试次数
            break

        # 尝试找鱼  根据最多次数的鱼类出现频率来选择鱼饵，并投掷鱼竿
        结果 = 钓鱼探测器.do_fish()

        # 没找到鱼，增加尝试次数继续寻找
        if not 结果:
            尝试次数 += 1
            continue

        # 找到鱼，重置尝试次数
        尝试次数 = 0
        # 发出提示音表示找到鱼
        播放mp3("aa.mp3")
        times=0
        continue_flag = False
        # 如果找到鱼，等待鱼上钩
        while 结果 is True:
            # 如果鱼咬钩，退出等待
            if 钓鱼环境.is_bite():
                print("咬钩了！！！！！！")
                播放mp3('./来了老弟.mp3')
                break
            # 如果还没咬钩
            time.sleep(0.25)
            print("还没咬钩,`等待时间：",times)
            # 等待时间+0.25
            times+=0.25
            # 如果等待时间超过上钩超时（默认45秒），且鱼没有咬钩，则投掷鱼竿
            if times>上钩超时 and not(钓鱼环境.is_bite()):
                # 如果正在钓鱼，则投掷鱼竿
                if 钓鱼环境.is_fishing():
                    print("正在钓鱼，拉鱼竿")  
                    钓鱼环境.drag()
                # 等待3秒
                time.sleep(3)
                # 重置等待次数 
                times=0
                # 继续标志设置为True
                continue_flag = True
                break
        # 如果继续标志为True，则继续等待
        if continue_flag == True:
            continue
        # 发出提示音表示找到鱼
        播放mp3("aa.mp3")
        # 拉鱼竿
        print("拉鱼竿了！！")
        钓鱼环境.drag()

        # 等待1秒
        time.sleep(1)
        # 重置钓鱼环境
        state = 钓鱼环境.reset()
        for i in range(钓鱼环境.max_step):
            # 将状态转换为FloatTensor
            state = torch.FloatTensor(state).unsqueeze(0)
            # 根据状态预测动作
            action = 智能体(state)
            # 将动作转换为整数
            action = torch.argmax(action, dim=1).numpy()
            # 执行动作
            state, reward, done = 钓鱼环境.step(action)
            # 如果完成钓鱼，则退出循环
            if done:
                break
        # 等待3秒
        time.sleep(3)
def 在日志中写入(内容):
    with open('日志.txt', 'a', encoding='utf-8') as f:
        f.write(内容 + '\n')
# 开始钓鱼流程 钓鱼一次启动 -> 我的main -> 钓鱼一次
def 钓鱼一次(预测器, 智能体, 上钩超时=45):
    在日志中写入('钓鱼一次'+time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    钓鱼探测器 = FishFind(预测器)                          # 初始化钓鱼探测器
    钓鱼环境 = Fishing(delay=0.1, max_step=10000, show_det=True)  # 初始化钓鱼环境
    结果 = 钓鱼探测器.do_fish() # 尝试找鱼  根据最多次数的鱼类出现频率来选择鱼饵，并投掷鱼竿
    if not 结果: 
        在日志中写入('没找到鱼')
        raise Exception("没找到鱼")  # 没找到鱼，那就报错呵呵哒
    播放mp3("aa.mp3") # 发出提示音表示找到鱼
    times=0 # 计时时间
    while 结果 is True:  # 等待鱼上钩
        if 钓鱼环境.is_bite(): # 如果鱼咬钩，退出等待
            print("咬钩了！！！！！！")
            播放mp3('./来了老弟.mp3')
            break
        # 还没咬钩
        time.sleep(0.25)
        print("还没咬钩,`等待时间：",times)
        # 等待时间+0.25
        times+=0.25
        # 如果等待时间超过上钩超时（默认45秒），且鱼没有咬钩，那就报错呵呵哒
        if times>上钩超时 and not(钓鱼环境.is_bite()): 
            在日志中写入('上钩超时')
            raise Exception("上钩超时")
    播放mp3('./来了老弟.mp3') # 发出提示音表示找到鱼
    # 拉鱼竿
    在日志中写入('拉鱼竿')
    print("拉鱼竿了！！")
    钓鱼环境.drag()
    # 等待1秒
    time.sleep(1)
    # 重置钓鱼环境
    在日志中写入('重置钓鱼环境')
    state = 钓鱼环境.reset()
    for i in range(钓鱼环境.max_step): # 默认10000次
        # 将状态转换为FloatTensor # tensor_state的形状变为：[1, 3] # tensor_state的值为：tensor([[1., 2., 3.]])
        # state包含了左侧，右侧，当前位置 对应的百分比位置
        state = torch.FloatTensor(state).unsqueeze(0) # 增一个维度
        print('状态：',state)
        在日志中写入('状态：'+str(state))
        # 根据状态预测动作
        action = 智能体(state)
        print('动作：',action)
        在日志中写入('动作：'+str(action))
        # 将动作转换为整数
        action = torch.argmax(action, dim=1).numpy()
        print('动作转为整数：',action)
        在日志中写入('动作转为整数：'+str(action))
        # 执行动作
        state, reward, done = 钓鱼环境.step(action)
        print('状态，奖励，是否完成：',state, reward, done)
        在日志中写入('状态，奖励，是否完成：'+str(state)+','+str(reward)+','+str(done))
        # 如果完成钓鱼，则退出循环
        if done:
            播放mp3('./不错哟.mp3')
            break
def 钓鱼一次启动():
    # 创建参数解析器 
    parser = make_parser()
    # 手动设置所有参数，对应命令行参数
    参数列表 = [
        "image",                                  # demo参数
        "-f", "yolox/exp/yolox_tiny_fish.py",    # exp_file参数
        "-c", "weights/best_tiny3.pth",          # ckpt参数
        "--conf", "0.25",                        # conf参数
        "--nms", "0.45",                         # nms参数
        "--tsize", "640",                        # tsize参数
        "--device", "gpu"                        # device参数
    ]
    # 解析参数
    args = parser.parse_args(参数列表)
    # 获取实验配置
    exp = get_exp(args.exp_file, args.name)
    # 运行主程序
    我的main(exp, args)
def 我的main(exp,args):
    # 设置实验名称
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    # TensorRT模式下强制使用GPU
    if args.trt:
        args.device = "gpu"

    logger.info("当前参数配置: {}".format(args))

    # 更新实验配置
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    # 初始化模型
    model = exp.get_model()
    logger.info("模型结构概要: {}".format(get_model_info(model, exp.test_size)))

    # 设置设备和精度
    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # 转换为半精度
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        if args.ckpt is None:
            trt_file = os.path.join(file_name, "model_trt.pth")
        else:
            trt_file = args.ckpt
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None
    # yolox预测器
    predictor = Predictor(model, exp, FISH_CLASSES, trt_file, decoder, args.device, args.fp16, args.legacy)
    # 钓鱼智能体
    agent = FishNet(in_ch=args.n_states, out_ch=args.n_actions)
    # 加载智能体模型
    agent.load_state_dict(torch.load(args.model_dir))
    # 设置为评估模式
    agent.eval()

    print('INIT OK')
    # while True:
    #     # 等待按下"r"键开始钓鱼
    #     print('Waiting for "r" to perform fishing')
    #     # 播放提示音
    #     winsound.Beep(500, 500)
    #     # 等待按下"r"键
    #     keyboard.wait('r')
    #     # 播放提示音
    #     winsound.Beep(500, 500)
    #     # 根据演示类型执行钓鱼流程
    #     if args.demo == "image":
    #         start_fishing(predictor, agent)

    # 改为自动钓鱼一次
    # 等待按下"r"键开始钓鱼
    print('Waiting for "r" to perform fishing')
    # 等待按下"r"键
    keyboard.wait('r')
    # 播放提示音
    播放mp3("aa.mp3")
    # 根据演示类型执行钓鱼流程
    if args.demo == "image":
        钓鱼一次(predictor, agent)


def 之前的启动():
    # 解析命令行参数
    args = make_parser().parse_args()
    # 获取实验配置
    exp = get_exp(args.exp_file, args.name)
    # print('exp:',exp)
    # 运行主程序
    main(exp, args)
# 自己写的启动 -> main -> start_fishing
def 自己写的启动():
    # 创建参数解析器
    parser = make_parser()
    # 手动设置所有参数，对应命令行参数
    参数列表 = [
        "image",                                  # demo参数
        "-f", "yolox/exp/yolox_tiny_fish.py",    # exp_file参数
        "-c", "weights/best_tiny3.pth",          # ckpt参数
        "--conf", "0.25",                        # conf参数
        "--nms", "0.45",                         # nms参数
        "--tsize", "640",                        # tsize参数
        "--device", "gpu"                        # device参数
    ]
    # 解析参数
    args = parser.parse_args(参数列表)
    # 获取实验配置
    exp = get_exp(args.exp_file, args.name)
    # 运行主程序
    main(exp, args)


#python fishing.py image -f yolox/exp/yolox_tiny_fish.py -c weights/best_tiny3.pth --conf 0.25 --nms 0.45 --tsize 640 --device gpu
if __name__ == "__main__":
    钓鱼一次启动()
    '''
    显卡加速 
    python fishing.py image -f yolox/exp/yolox_tiny_fish.py -c weights/best_tiny3.pth --conf 0.25 --nms 0.45 --tsize 640 --device gpu
    cpu运行
    python fishing.py image -f yolox/exp/yolox_tiny_fish.py -c weights/best_tiny3.pth --conf 0.25 --nms 0.45 --tsize 640 --device cpu
    '''
