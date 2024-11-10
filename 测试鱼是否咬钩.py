import numpy as np
import cv2
import os
def psnr(img1, img2):
    """计算两张图片的PSNR psnr是峰值信噪比 归一化后像素差平方和 表示图像的相似度"""
    mse = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def is_bite():
    """检测鱼是否咬钩"""
    # 截取指定区域的图像
    # img = cap(region=[1595, 955, 74, 74],fmt='BGR')
    # 读取钓鱼图片
    当前目录 = os.path.dirname(os.path.abspath(__file__))
    bite = cv2.imread('./imgs/bite.png', cv2.IMREAD_GRAYSCALE)    # 咬钩提示图像
    print('bite.shape',bite.shape)
    # img = cap(region=[1595, 955, 74, 74],fmt='BGR')
    # 指定范围
    img=cv2.imread('./is_bited.png')[951:1025,1568:1642]
    print('img.shape',img.shape)
    # 转换为灰度图并进行边缘检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge_output = cv2.Canny(gray, 50, 150)
    # 通过PSNR值判断是否咬钩
    print('psnr',psnr(bite, edge_output))
    return psnr(bite, edge_output)>10

def 测试():
    print(is_bite())

if __name__ == '__main__':
    测试()

