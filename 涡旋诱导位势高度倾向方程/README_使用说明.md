# 涡旋诱导位势高度倾向方程：全球计算使用说明

本包计算热通量贡献 EHF、涡度通量贡献 EVF 及其总和。方法为 Gan 形式算子的全球近似诊断，保留目前使用的日平均 Lanczos 2.5—8 日带通。全球经度周期、极点球谐正则，没有人工南北纬度边界；压力边界仍必须设置。

## 1. 目录和运行顺序

| 文件夹 | 入口 | 工作和输出 |
|---|---|---|
| 01_滤波 | filter_lanczos.py | 原始6小时 u/v/T → 日平均 → Lanczos带通；每变量每月一个文件 |
| 02_单独项 | calculate_temperature.py | 原始未滤波T的月平均，年度文件；保留全部原始压力层 |
| 02_单独项 | rotational_terms.py | 无辐散风重建、热/涡度通量散度月平均；Gaussian网格 terms.YYYY.MM.nc |
| 03_算子 | calculate_background.py | 月气候态纬向平均背景，位温和稳定度 sigma |
| 03_算子 | global_solver.py / test_operator.py | 球谐Galerkin和压力有限体积算子核心；独立解析测试 |
| 04_全球计算 | cal_global_rotational.py | 热/涡度/总强迫独立反演，数值检查，输出 m/day |

`config.py` 是统一路径配置。四个文件夹与根目录文件应一起保留，不要只复制一个入口脚本。入口支持从任意工作目录启动。第三阶段提供算子实现和系数；具体月份的稀疏矩阵由第四阶段调用该实现组装，不保存巨大矩阵文件。

## 2. 准备环境

服务器推荐使用已经成功运行 windspharm/pyspharm 的 Python 环境：

```bash
python -m pip install numpy scipy netCDF4
python check_environment.py
```

第二、四阶段需要 `from spharm import Spharmt`，即 pyspharm 的 SPHEREPACK 后端；不要用名称相似的不同球谐库替代。第一阶段还需要命令行 `ncl`：Python负责解包、日平均和跨月对齐，NCL只生成与旧程序相同的 `filwgts_lanczos` 权重。可在支持的平台用 conda-forge 安装 NCL、pyspharm；优先保留已成功的服务器环境。本包不要求 xarray、numba 或 windspharm Python前端。

设置单线程可避免BLAS过度并行：

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
```

## 3. 首先修改配置、确认输入

编辑根目录 `config.py` 中 RAW_ROOT、BANDPASS、BASIC、BACKGROUND、TERMS、OUTPUT。默认路径为通用示例，使用者必须改成自己的数据目录。默认1982—2023年，共504个月。输出目录建议独立设置。

原始文件需要全球1°、完整00/06/12/18 UTC、全部压力层：

```text
RAW_ROOT/uwnd/uwnd.6h.global.1deg.YYYY.MM.nc   变量u或uwnd，m/s
RAW_ROOT/vwnd/vwnd.6h.global.1deg.YYYY.MM.nc   变量v或vwnd，m/s
RAW_ROOT/air/air.6h.global.1deg.YYYY.MM.nc     变量t或air，K
```

时间坐标兼容time/valid_time，压力兼容lev/level/pressure_level/p，纬经度兼容短名或latitude/longitude。依据单位识别Pa/hPa并同步重排数据，不依据年份猜测。每个原始文件分别自动应用自己的scale_factor/add_offset，禁止二次解包。压力单位缺失时先查清，然后显式加 `--assume-pressure-unit hPa` 或 Pa；不要猜。

第一阶段需要研究时段前后各至少28日原始数据，例如1982—2023需要1981年12月及2024年1月。缺测或缺少邻月直接停止，不填零或镜像补边。

## 4. 路线A：从原始数据完整运行

先测试权重和求解核心：

```bash
python 01_滤波/test_input_reader.py
python 01_滤波/filter_lanczos.py weights-test
python 02_单独项/rotational_terms.py self-test
python 03_算子/test_operator.py
```

最后应有PASS。测试兼容性拒绝时出现 `bad COMPATIBILITY=1` 属于预期测试。

第一阶段，三变量带通：

```bash
nohup python -u 01_滤波/filter_lanczos.py batch --start-year 1982 --end-year 2023 > 01_filter.log 2>&1 &
```

等待结束后，第二阶段分别计算背景所需的未滤波月温度和旋转风通量散度：

```bash
nohup python -u 02_单独项/calculate_temperature.py --start-year 1982 --end-year 2023 > 02_temperature.log 2>&1 &
```

```bash
nohup python -u 02_单独项/rotational_terms.py batch --start-year 1982 --end-year 2023 > 02_terms.log 2>&1 &
```

这两个任务可以分别运行；第三阶段须等待所有年度T_month完成：

```bash
python -u 03_算子/calculate_background.py --clim-start 1982 --clim-end 2023 > 03_background.log 2>&1
```

第四阶段先算单月。所有前置任务完成后执行：

```bash
python -u 04_全球计算/cal_global_rotational.py solve --year 2021 --month 1 > 04_global_202101.log 2>&1
```

检查日志DIAGNOSTICS、ADDITIVITY及SAVED；然后批量：

```bash
nohup python -u 04_全球计算/cal_global_rotational.py batch --start-year 1982 --end-year 2023 > 04_global_1982_2023.log 2>&1 &
tail -f 04_global_1982_2023.log
```

上述后台命令只提交任务，不表示前一步完成。检查进程及日志，不能启动后立刻依次运行全部后续命令。

## 5. 路线B：复用当前服务器已有数据

已有2.5—8日带通、T_month和背景文件时，不必重新运行第一阶段、月温度和第三阶段。配置路径后，直接运行第四阶段self-test、solve或batch即可。第四阶段优先读取阶段二缓存；缓存不存在则现场运行相同的基础项核心，因此不要求先把504个月的Gaussian缓存全部保存。

```bash
python 04_全球计算/cal_global_rotational.py self-test
nohup python -u 04_全球计算/cal_global_rotational.py batch --start-year 1982 --end-year 2023 > global_batch.log 2>&1 &
```

不会读取旧A/B区域高度倾向，也不会将旧全风通量直接作为新全球强迫。

## 6. 覆盖、续跑与空间占用

默认不覆盖已有文件。第一至三阶段发现同名结果会停止；已完成年份/月可调整运行范围后继续，单月基础项用solve。确定重算才加 `--overwrite`。

第四阶段batch支持续跑：重执行同一命令，校验符合当前版本、背景哈希、输入路径、截断和数值检查的完成月份后跳过。只存在强迫、未完成高度倾向的月份会重算。设置冲突会停止。solve重算已有月也需显式--overwrite。

缓存TERMS含float64 Gaussian场，504个月会占较大空间；不要求缓存，路线B现场计算更省空间。强迫快照和高度倾向仍会保存。

续跑元数据检查不是输入文件内容逐字节检查。若在相同路径覆盖了滤波数据或terms，必须重算受影响的最终月份；建议改OUTPUT，避免混合不同输入。改滤波频率时必须同步修改命名、读取模板和方法属性，本包目前固定2.5—8日命名；不能只改截止频率后仍使用旧标签。

## 7. 输出和读取

| 数据 | 名称 | 维度与单位 |
|---|---|---|
| 日带通 | uwnd、vwnd、t | time/lev/lat/lon；m/s、K；压力hPa |
| 月温度 | T_month，n_samples | 12月/全部lev/lat/lon；K；年度文件压力hPa |
| 单独项 | div_T、div_zeta | lev/mu/lon；K/s、s^-2；Gaussian网格，压力Pa |
| 背景 | sigma、theta_background、dtheta_dp | month/全部lev/lat；sigma为m²s^-2Pa^-2；压力Pa |
| 强迫快照 | div_T、div_zeta、sigma | Gaussian网格；未反演 |
| 最终高度倾向 | Ztend_heat、Ztend_vort、Ztend_total、Ztend_total_independent | time/lev/lat/lon；m/day；压力Pa |

最终文件：OUTPUT/Ztend.YYYY.MM.nc。压力100—1000hPa原始13层：100、150、200、250、300、400、500、600、700、850、925、975、1000。全球纬度90至−90，经度0至359。

```python
from netCDF4 import Dataset
import numpy as np
with Dataset('/自己的输出目录/Ztend.2021.01.nc') as nc:
    k=np.flatnonzero(np.isclose(nc['lev'][:],85000))[0]
    ehf=nc['Ztend_heat'][0,k,:,:]  # 已是m/day，不再乘86400
    evf=nc['Ztend_vort'][0,k,:,:]
    lat,lon=nc['lat'][:],nc['lon'][:]
```

EHF是热通量引起的高度倾向，不是div_T本身；EVF同理。回归、指数和绘图不属于这四步，本包输出月度场供后续季节平均与回归使用。

## 8. 出错时定位

找不到文件：核对config和文件模板；压力/时间检查失败：检查原文件元数据、记录完整性及归一坐标。Nonpositive stability：核验背景定义和解包，不能取abs或裁剪。兼容性、残差、角动量检查失败：保存日志及forcing文件定位，不放宽阈值或扣强迫均值。

检查spharm错误应使用原成功环境。调 `--chunk-days 2` 可降低内存，定义不变；降低truncation改变空间尺度，不能当作等价优化。

详见 `方法与验证说明.md`，其中明确列出近似、公式和测试边界。
