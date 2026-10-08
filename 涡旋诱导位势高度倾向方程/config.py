"""统一配置：首先修改路径；物理/数值设置有联动，修改后需要重新核验。"""
from pathlib import Path
RAW_ROOT = Path('/data/ERA5_4xDaily')
BANDPASS = Path('/data/eddy_height_tendency/bandpass')
ROOT = Path('/data/eddy_height_tendency')
BASIC = ROOT / 'basic'
BACKGROUND = ROOT / 'step2/background.1982-2023.nc'
TERMS = ROOT / 'terms_global_rotational'
OUTPUT = ROOT / 'step3_global_rotational_trial'
START_YEAR, END_YEAR = 1982, 2023
TRUNCATION, CHUNK_DAYS = 179, 8
# 固定日采样；NCL 57点 Lanczos，2.5—8日，sigma=1。
FILTER_WEIGHTS, FILTER_LOW, FILTER_HIGH, FILTER_SIGMA = 57, 1/8, 1/2.5, 1.0
