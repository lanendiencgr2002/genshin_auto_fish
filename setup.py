
#!/usr/bin/env python
# 版权声明: Megvii, Inc. 及其附属公司保留所有权利

import re
import setuptools
import glob
from os import path
import torch
from torch.utils.cpp_extension import CppExtension

# 检查 PyTorch 版本要求
torch_ver = [int(x) for x in torch.__version__.split(".")[:2]]
assert torch_ver >= [1, 7], "需要 PyTorch >= 1.7 版本"

def get_extensions():
    """获取 C++ 扩展模块配置"""
    # 获取当前目录和扩展源码目录
    当前目录 = path.dirname(path.abspath(__file__))
    扩展目录 = path.join(当前目录, "yolox", "layers", "csrc")

    # 配置源文件
    主源文件 = path.join(扩展目录, "vision.cpp")
    源文件列表 = glob.glob(path.join(扩展目录, "**", "*.cpp"))
    
    # ... 其余代码保持不变 ...

# 读取版本号
with open("yolox/__init__.py", "r") as f:
    版本号 = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
        f.read(), re.MULTILINE
    ).group(1)

# 读取长描述
with open("README.md", "r", encoding="utf-8") as f:
    详细描述 = f.read()

# 配置安装信息
setuptools.setup(
    name="yolox",
    version=版本号,
    author="basedet team",
    python_requires=">=3.6",
    long_description=详细描述,
    ext_modules=get_extensions(),
    classifiers=["Programming Language :: Python :: 3", "Operating System :: OS Independent"],
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    packages=setuptools.find_packages(),
)
