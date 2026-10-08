#!/usr/bin/env python3
"""原始6小时数据→完整日平均→57点Lanczos带通；权重由NCL原函数生成。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import argparse,calendar,os,subprocess,tempfile
from datetime import datetime
from contextlib import ExitStack
import numpy as np
from scipy.ndimage import convolve1d
from netCDF4 import Dataset,date2num
from config import *
from audit_readers import InputFile,check_grid,require

NAMES={'uwnd':('u','uwnd'),'vwnd':('v','vwnd'),'air':('t','air')}
def weights():
    """直接调用NCL的filwgts_lanczos，不另造近似权重或重新归一化。"""
    require(FILTER_WEIGHTS==57,'当前跨月窗口固定支持57点；改窗口需同步修改读取范围')
    require(0<FILTER_LOW<FILTER_HIGH<.5,'带通频率必须在日采样Nyquist以内')
    with tempfile.TemporaryDirectory() as folder:
        out=Path(folder)/'weights.txt';code=Path(folder)/'weights.ncl'
        code.write_text(f'begin\n w = filwgts_lanczos({FILTER_WEIGHTS},2,{FILTER_LOW:.17g},{FILTER_HIGH:.17g},{FILTER_SIGMA:.17g})\n asciiwrite(getenv("WEIGHTS_OUT"),w)\nend\n')
        subprocess.run(['ncl','-Q',str(code)],env={**os.environ,'WEIGHTS_OUT':str(out)},check=True)
        w=np.loadtxt(out);require(w.shape==(57,) and np.isfinite(w).all(),'NCL权重输出异常')
    require(np.allclose(w,w[::-1]),'权重不对称');return w

def adjacent(year,month,offset):
    n=year*12+month-1+offset;return n//12,n%12+1

def run(name,year,month,args,w):
    path=args.out/name/f'{name}.daymean.global.1deg.{year}.{month:02d}.BPF_L_2.5_8d.nc'
    require(args.overwrite or not path.exists(),f'{path}已存在；覆盖用--overwrite')
    # 邻月分别自动解包、分别按压力/经纬度排序，防止跨2021/2022错层。
    with ExitStack() as stack:
        src=[]
        for offset in (-1,0,1):
            y,m=adjacent(year,month,offset)
            obj=InputFile(args.raw/name/f'{name}.6h.global.1deg.{y}.{m:02d}.nc',
                NAMES[name],y,m,False,args.assume_pressure_unit,raw_wind=name!='air')
            stack.callback(obj.close);src.append(obj)
        for obj in src:check_grid(src[1],obj)
        nd=calendar.monthrange(year,month)[1];before=src[0].nt//4
        require(before>=28 and src[2].nt//4>=28,'必须有完整前后28日，不填充时间边界')
        ref=src[1];path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_name(path.name+'.partial')
        with Dataset(tmp,'w') as nc:
            for key,a,unit in [('time',np.arange(nd),'hours since 1900-01-01'),('lev',ref.p,'hPa'),('lat',ref.lat,'degrees_north'),('lon',ref.lon,'degrees_east')]:
                nc.createDimension(key,len(a));v=nc.createVariable(key,'f8',(key,));v.units=unit
                v[:]=date2num([datetime(year,month,d+1) for d in range(nd)],unit) if key=='time' else a
            nc['time'].calendar='standard'
            v=nc.createVariable('t' if name=='air' else name,'f4',('time','lev','lat','lon'),zlib=True,complevel=3)
            v.units='K' if name=='air' else 'm/s'
            nc.filter='daily Lanczos 2.5-8 d; NCL filwgts_lanczos; 57 weights; sigma=1'
            nc.source_files=';'.join(str(o.path) for o in src)
            nc.createDimension('filter_lag',len(w));nc.createVariable('filter_weights','f8',('filter_lag',))[:]=w
            for k,p in enumerate(ref.p):
                days=[]
                for obj in src:
                    a=obj.level(k);require(np.isfinite(a).all(),'原始输入缺测，不填零')
                    days.append(a.reshape(obj.nt//4,4,len(ref.lat),len(ref.lon)).mean(axis=1))
                a=np.concatenate(days)
                # 仅写中心月，距读入数据两端>=28日；constant填充不进入正式输出。
                filtered=convolve1d(a,w,axis=0,mode='constant',cval=0.)
                v[:,k]=filtered[before:before+nd]
                print(f'FILTER {name} {year}-{month:02d} {p:g}hPa',flush=True)
        tmp.replace(path);print('SAVED',path,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['solve','batch','weights-test'])
    ap.add_argument('--variable',choices=['all',*NAMES],default='all')
    ap.add_argument('--year',type=int,default=2021);ap.add_argument('--month',type=int,default=1)
    ap.add_argument('--start-year',type=int,default=START_YEAR);ap.add_argument('--end-year',type=int,default=END_YEAR)
    ap.add_argument('--raw',type=Path,default=RAW_ROOT);ap.add_argument('--out',type=Path,default=BANDPASS)
    ap.add_argument('--assume-pressure-unit',choices=['Pa','hPa']);ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args();w=weights()
    if args.stage=='weights-test':print('PASS NCL权重',w);sys.exit(0)
    require(1<=args.month<=12 and args.start_year<=args.end_year,'月份或年份范围错误')
    dates=[(args.year,args.month)] if args.stage=='solve' else [(y,m) for y in range(args.start_year,args.end_year+1) for m in range(1,13)]
    for y,m in dates:
        for name in NAMES if args.variable=='all' else [args.variable]:run(name,y,m,args,w)
