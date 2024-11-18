"""
依赖包安装脚本

该脚本用于自动化安装项目所需的Python依赖包
主要功能：
1. 支持通过代理服务器安装包（适用于网络受限环境）
2. 支持选择性安装CPU或CUDA版本的PyTorch
3. 使用subprocess替代pip.main以避免潜在的包冲突

关于依赖包覆盖行为：
- pip install 命令默认会更新已安装的包到指定版本
- 如果已安装版本满足版本要求（如 >=5.3.1），则不会重新安装
- 使用 ==1.8.2 这样的精确版本号会强制安装指定版本，可能覆盖现有版本
- 当不指定版本号时（如直接 pip install cython）：
  1. 如果包未安装：安装最新稳定版本
  2. 如果包已安装：保持现有版本不变
  3. 如果要强制更新到最新版：需要使用 pip install --upgrade cython
- 如果需要避免覆盖，可以：
  1. 使用 pip install --no-deps 阻止更新依赖
  2. 在安装前使用 pip freeze 检查现有版本
  3. 添加 --ignore-installed 参数强制重新安装
"""

import subprocess
import sys
import argparse
from typing import List, Optional

def pip_install(proxy: Optional[str], args: List[str]) -> None:
    """
    执行pip安装命令的封装函数
    
    Args:
        proxy: 代理服务器地址，如 http://127.0.0.1:1080
        args: pip install 命令的参数列表
    
    注意：
    - 使用subprocess.run而不是pip.main以避免包管理器的环境污染
    - check=True 确保安装失败时抛出异常，便于错误处理
    - 该函数会遵循pip的默认行为：
      * 对于已安装的包，如果版本满足要求则跳过
      * 如果指定了精确版本，则会更新到指定版本
    """
    if proxy is None:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", *args],
            check=True,
        )
    else:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", f"--proxy={proxy}", *args],
            check=True,
        )

def main():
    """
    主函数：解析命令行参数并安装所需依赖
    
    支持的参数：
    --cuda: 指定CUDA版本，用于选择性安装GPU版PyTorch
    --proxy: 指定HTTP代理服务器
    """
    parser = argparse.ArgumentParser(description="install requirements")
    parser.add_argument("--cuda", default=None, type=str)
    parser.add_argument(
        "--proxy",
        default=None,
        type=str,
        help="specify http proxy, [http://127.0.0.1:1080]",
    )
    args = parser.parse_args()

    # 定义所需的依赖包列表
    # 注意：PyTorch的版本会根据是否指定CUDA版本动态确定
    pkgs = f"""
    cython
    scikit-image
    loguru
    matplotlib
    tabulate
    tqdm
    pywin32
    PyAutoGUI
    PyYAML>=5.3.1
    opencv_python
    keyboard
    Pillow
    pymouse
    numpy>=1.21.1
    torch==1.8.2+{"cpu" if args.cuda is None else "cu" + args.cuda} -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
    torchvision==0.9.2+{"cpu" if args.cuda is None else "cu" + args.cuda} --no-deps -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
    thop --no-deps
    git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
    """

    # 逐行处理依赖包列表
    for line in pkgs.split("\n"):
        line = line.strip()
        if len(line) > 0:  # 跳过空行
            pip_install(args.proxy, line.split())

    print("\nsuccessfully installed requirements!")

if __name__ == "__main__":
    main()
