#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# YOLOX 实验配置构建模块 - 用于加载和管理不同 YOLOX 模型的实验配置

# 导入必要的Python标准库
import importlib  # 用于动态导入模块
import os        # 用于处理文件路径
import sys       # 用于修改Python系统路径


def get_exp_by_file(exp_file):
    """
    通过文件路径加载实验配置
    参数:
        exp_file (str): 实验配置文件的路径
    返回:
        exp: 实验配置对象
    """
    try:
        # 将配置文件所在目录添加到Python的模块搜索路径中
        sys.path.append(os.path.dirname(exp_file))
        
        # 获取配置文件的基本名称（不含路径和扩展名）并导入该模块
        # os.path.basename() 获取文件名
        # split(".")[0] 去除.py扩展名
        # importlib.import_module() 动态导入模块
        current_exp = importlib.import_module(os.path.basename(exp_file).split(".")[0])
        
        # 创建并返回配置类的实例
        exp = current_exp.Exp()
    except Exception:
        # 如果导入失败或找不到Exp类，抛出导入错误
        raise ImportError("配置文件 {} 中未找到 'Exp' 类".format(exp_file))
    
    # 返回配置对象
    return exp


def get_exp_by_name(exp_name):
    """
    通过预定义的模型名称加载实验配置
    参数:
        exp_name (str): 模型名称，如 'yolox-s', 'yolox-m' 等
    """
    # 导入YOLOX包
    import yolox

    # 获取YOLOX包的根目录路径
    yolox_path = os.path.dirname(os.path.dirname(yolox.__file__))
    
    # 定义模型名称到配置文件的映射字典
    filedict = {
        "yolox-s": "yolox_s.py",      # 小型模型配置文件
        "yolox-m": "yolox_m.py",      # 中型模型配置文件
        "yolox-l": "yolox_l.py",      # 大型模型配置文件
        "yolox-x": "yolox_x.py",      # 超大型模型配置文件
        "yolox-tiny": "yolox_tiny.py",# 轻量级模型配置文件
        "yolox-nano": "nano.py",      # 超轻量级模型配置文件
        "yolov3": "yolov3.py",        # YOLOv3模型配置文件
    }
    
    # 根据模型名称获取对应的配置文件名
    filename = filedict[exp_name]
    
    # 构建完整的配置文件路径
    # 配置文件位于 YOLOX包路径/exps/default/ 目录下
    exp_path = os.path.join(yolox_path, "exps", "default", filename)
    
    # 通过文件路径加载并返回配置对象
    return get_exp_by_file(exp_path)


def get_exp(exp_file, exp_name):
    """
    统一的配置获取接口，支持通过文件路径或预定义名称加载配置
    参数:
        exp_file (str): 配置文件路径
        exp_name (str): 预定义的模型名称
    """
    # 确保至少提供了一种配置获取方式
    assert (
        exp_file is not None or exp_name is not None
    ), "请提供配置文件路径或模型名称"
    
    # 优先使用配置文件路径加载配置
    if exp_file is not None:
        return get_exp_by_file(exp_file)
    # 如果没有提供配置文件路径，则使用预定义名称加载配置
    else:
        return get_exp_by_name(exp_name)
