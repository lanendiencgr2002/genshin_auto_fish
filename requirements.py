# import pip
import subprocess
import sys
import argparse

# use python type hints to make code more readable
from typing import List, Optional


def pip_install(proxy: Optional[str], args: List[str]) -> None:
    if proxy is None:
        # pip.main(["install", f"--proxy={proxy}", *args])
        subprocess.run(
            [sys.executable, "-m", "pip", "install", *args],
            # capture_output=False,
            check=True,
        )
    else:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", f"--proxy={proxy}", *args],
            # capture_output=False,
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(description="install requirements")
    parser.add_argument("--cuda", default=None, type=str)
    parser.add_argument(
        "--proxy",
        default=None,
        type=str,
        help="specify http proxy, [http://127.0.0.1:1080]",
    )
    args = parser.parse_args()

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

    for line in pkgs.split("\n"):
        # 处理每一行
        line = line.strip()  # 去除行首行尾的空白字符

        if len(line) > 0:  # 如果行不为空
            pip_install(args.proxy, line.split())

    print("\nsuccessfully installed requirements!")


if __name__ == "__main__":
    main()
