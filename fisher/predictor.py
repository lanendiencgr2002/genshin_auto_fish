import time
from loguru import logger

import os
import torch
import cv2

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import FISH_CLASSES
from yolox.utils import postprocess, vis

class Predictor(object):
    '''
    原神自动钓鱼的图像预测器类
    
    主要功能：
    1. 使用 YOLOX 模型检测游戏画面中的钓鱼相关元素
    2. 处理和预测图像中的目标位置
    3. 提供可视化支持
    
    参数：
        model: 训练好的 YOLOX 模型
        exp: 实验配置对象，包含模型参数设置
        cls_names: 类别名称列表，默认使用 FISH_CLASSES
        trt_file: TensorRT 模型文件路径，用于加速推理（可选）
        decoder: 模型输出解码器（可选）
        device: 运行设备，可选 "cpu" 或 "gpu"，默认为 "cpu"
        fp16: 是否使用半精度浮点数，默认为 False
        legacy: 是否使用旧版本预处理方式，默认为 False
    
    主要方法：
        inference(): 对输入图像进行推理
        image_det(): 检测图像中的目标
        visual(): 可视化检测结果
    '''
    def __init__(
        self,
        model,
        exp,
        cls_names=FISH_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        # 初始化模型和基本参数
        self.model = model  # 设置模型
        self.cls_names = cls_names  # 设置类别名称列表
        self.decoder = decoder  # 设置解码器
        self.num_classes = exp.num_classes  # 类别数量
        self.confthre = exp.test_conf  # 置信度阈值
        self.nmsthre = exp.nmsthre  # 非极大值抑制阈值
        self.test_size = exp.test_size  # 测试图片大小
        self.device = device  # 运行设备（CPU/GPU）
        self.fp16 = fp16  # 是否使用半精度浮点数
        self.preproc = ValTransform(legacy=legacy)  # 图像预处理器

        # 如果提供了 TensorRT 模型文件，则加载优化后的模型
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            # 创建测试输入并运行模型
            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        """对输入图像进行推理"""
        # 初始化图像信息字典
        img_info = {"id": 0}
        
        # 处理输入图像，可以是文件路径或图像数据
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        # 记录图像尺寸信息
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        # 计算缩放比例
        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        # 图像预处理
        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)  # 转换为张量并增加批次维度
        img = img.float()  # 转换为浮点类型

        # GPU相关处理
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # 转换为半精度

        # 执行推理
        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)  # 模型推理
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            # 后处理：非极大值抑制等
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("推理时间: {:.4f}秒".format(time.time() - t0))
        return outputs, img_info

    def image_det(self, img, with_info=False):
        """检测图像中的目标"""
        # 执行推理
        outputs, img_info = self.inference(img)
        ratio = img_info["ratio"]
        obj_list = []

        # 如果没有检测到目标，返回None
        if outputs[0] is None:
            return None

        # 处理每个检测到的目标
        for item in outputs[0].cpu():
            bboxes = item[:4]  # 获取边界框坐标
            bboxes /= ratio  # 还原到原始图像尺寸
            scores = item[4] * item[5]  # 计算置信度得分
            # 将结果添加到列表：[类别名称, 置信度, 边界框坐标]
            obj_list.append([self.cls_names[int(item[6])], scores, [bboxes[0], bboxes[1], bboxes[2], bboxes[3]]])

        # 根据参数决定返回内容
        if with_info:
            return obj_list, outputs, img_info
        else:
            return obj_list

    def visual(self, output, img_info, cls_conf=0.35):
        """可视化检测结果"""
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        
        # 如果没有检测结果，返回原图
        if output is None:
            return img
        
        output = output.cpu()
        bboxes = output[:, 0:4]  # 获取边界框坐标
        bboxes /= ratio  # 还原到原始图像尺寸

        cls = output[:, 6]  # 获取类别
        scores = output[:, 4] * output[:, 5]  # 计算置信度得分

        # 在图像上绘制检测结果
        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res