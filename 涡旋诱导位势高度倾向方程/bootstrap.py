"""让脚本在任意工作目录运行；不依赖当前目录。"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
for name in ('02_单独项', '03_算子', '04_全球计算'):
    sys.path.insert(0, str(ROOT / name))
