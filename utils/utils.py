import time
import argparse
import cv2
import pyautogui
import numpy as np
import win32api, win32con, win32gui, win32ui
from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).parent.parent.joinpath("config.yaml")
assert CONFIG_PATH.is_file()


with open(CONFIG_PATH, encoding='utf-8') as f:
    result = yaml.safe_load(f)
    DEFAULT_MONITOR_WIDTH = result.get("windows").get("monitor_width")
    DEFAULT_MONITOR_HEIGHT = result.get("windows").get("monitor_height")
    WINDOW_NAME = result.get("game").get("window_name")

MOUSE_LEFT=0
MOUSE_MID=1
MOUSE_RIGHT=2

mouse_list_down=[win32con.MOUSEEVENTF_LEFTDOWN, win32con.MOUSEEVENTF_MIDDLEDOWN, win32con.MOUSEEVENTF_RIGHTDOWN]
mouse_list_up=[win32con.MOUSEEVENTF_LEFTUP, win32con.MOUSEEVENTF_MIDDLEUP, win32con.MOUSEEVENTF_RIGHTUP]

gvars=argparse.Namespace()
hwnd = win32gui.FindWindow(None, WINDOW_NAME)
gvars.genshin_window_rect = win32gui.GetWindowRect(hwnd)

# def cap(region=None):
#     img = pyautogui.screenshot(region=region) if region else pyautogui.screenshot()
#     return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def cap(region=None ,fmt='RGB'):
    """截图 返回图像  region为截图区域 fmt为图像格式"""
    return cap_raw(gvars.genshin_window_rect_img if region is None else (region[0]+gvars.genshin_window_rect_img[0], region[1]+gvars.genshin_window_rect_img[1], region[2], region[3]), fmt=fmt)

def cap_raw(region=None, fmt='RGB'):
    """
    截取屏幕指定区域的图像
    参数:
        region: 截图区域的坐标和大小 (left, top, width, height)
        fmt: 返回图像的格式，支持 'RGB' 或 'BGR'
    返回:
        返回截取的图像数组
    """
    # 设置截图区域
    if region is not None:
        left, top, w, h = region  # 使用指定的区域参数
    else:
        # 如果没有指定区域，使用默认的显示器尺寸
        w = DEFAULT_MONITOR_WIDTH
        h = DEFAULT_MONITOR_HEIGHT
        left = 0
        top = 0

    # 获取原神游戏窗口句柄
    hwnd = win32gui.FindWindow(None, WINDOW_NAME)
    
    # 创建设备上下文和兼容位图
    wDC = win32gui.GetWindowDC(hwnd)  # 获取窗口的设备上下文
    dcObj = win32ui.CreateDCFromHandle(wDC)  # 创建设备上下文对象
    cDC = dcObj.CreateCompatibleDC()  # 创建兼容的设备上下文
    dataBitMap = win32ui.CreateBitmap()  # 创建位图对象

    # 创建兼容位图并选择到兼容设备上下文中
    dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
    cDC.SelectObject(dataBitMap)
    
    # 将窗口内容复制到位图中
    cDC.BitBlt((0, 0), (w, h), dcObj, (left, top), win32con.SRCCOPY)
    
    # 获取位图数据并转换为numpy数组
    signedIntsArray = dataBitMap.GetBitmapBits(True)
    img = np.fromstring(signedIntsArray, dtype="uint8")
    img.shape = (h, w, 4)  # 重塑数组为图像格式（高度，宽度，4通道RGBA）

    # 释放资源
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())
    
    # 根据指定格式转换图像颜色空间
    if fmt == 'BGR':
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2BGR)
    if fmt == 'RGB':
        return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2RGB)
    else:
        raise ValueError('不支持的图像格式，只能使用 RGB 或 BGR')


def mouse_down(x, y, button=MOUSE_LEFT):
    """模拟鼠标按下 x,y为像素坐标 button为鼠标按键"""
    time.sleep(0.1)
    xx,yy=x+gvars.genshin_window_rect[0], y+gvars.genshin_window_rect[1]
    win32api.SetCursorPos((xx,yy))
    win32api.mouse_event(mouse_list_down[button], xx, yy, 0, 0)

def mouse_move(dx, dy):
    """模拟鼠标移动 dx,dy为像素坐标"""
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, dx, dy, 0, 0)

def mouse_up(x, y, button=MOUSE_LEFT):
    """模拟鼠标抬起 x,y为像素坐标 button为鼠标按键"""
    time.sleep(0.1)
    xx, yy = x + gvars.genshin_window_rect[0], y + gvars.genshin_window_rect[1]
    win32api.SetCursorPos((xx, yy))
    win32api.mouse_event(mouse_list_up[button], xx, yy, 0, 0)

def mouse_click(x, y, button=MOUSE_LEFT):
    """模拟鼠标点击 x,y为像素坐标 button为鼠标按键"""
    mouse_down(x, y, button)
    mouse_up(x, y, button)

def mouse_down_raw(x, y, button=MOUSE_LEFT):
    """模拟鼠标按下 x,y为像素坐标 button为鼠标按键"""
    xx, yy = x + gvars.genshin_window_rect[0], y + gvars.genshin_window_rect[1]
    win32api.mouse_event(mouse_list_down[button], xx, yy, 0, 0)

def mouse_up_raw(x, y, button=MOUSE_LEFT):
    """模拟鼠标抬起 x,y为像素坐标 button为鼠标按键"""
    xx, yy = x + gvars.genshin_window_rect[0], y + gvars.genshin_window_rect[1]
    win32api.mouse_event(mouse_list_up[button], xx, yy, 0, 0)

def mouse_click_raw(x, y, button=MOUSE_LEFT):
    """模拟鼠标点击 x,y为像素坐标 button为鼠标按键"""
    mouse_down_raw(x, y, button)
    mouse_up_raw(x, y, button)

def match_img(img, target, type=cv2.TM_CCOEFF):
    """
    匹配图像 img为原图 target为模板图 type为匹配类型
    返回匹配到的图像的左上角坐标、右下角坐标、中心坐标
    """
    h, w = target.shape[:2]
    res = cv2.matchTemplate(img, target, type)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if type in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        return (
            *min_loc,
            min_loc[0] + w,
            min_loc[1] + h,
            min_loc[0] + w // 2,
            min_loc[1] + h // 2,
        )
    else:
        return (
            *max_loc,
            max_loc[0] + w,
            max_loc[1] + h,
            max_loc[0] + w // 2,
            max_loc[1] + h // 2,
        )


def list_add(li, num):
    """列表元素相加 num为数字或列表"""
    if isinstance(num, int) or isinstance(num, float):
        return [x + num for x in li]
    elif isinstance(num, list) or isinstance(num, tuple):
        return [x + y for x, y in zip(li, num)]


def psnr(img1, img2):
    """计算两张图片的PSNR psnr是峰值信噪比 归一化后像素差平方和 表示图像的相似度"""
    mse = np.mean((img1 / 255.0 - img2 / 255.0) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))


def distance(x1, y1, x2, y2):
    """计算两点之间的距离"""
    return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
