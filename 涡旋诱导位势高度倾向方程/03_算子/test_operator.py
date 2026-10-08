from pathlib import Path
import sys
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
import bootstrap
"""独立解析解/守恒/相容性测试，不需要ERA5或spharm。"""
from global_solver import self_test
if __name__=='__main__':self_test()
