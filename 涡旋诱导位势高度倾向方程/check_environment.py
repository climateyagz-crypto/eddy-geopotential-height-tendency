"""检查依赖；不访问或修改数据。"""
import importlib,shutil,sys
bad=[]
for name in ('numpy','scipy','netCDF4','spharm'):
    try:
        m=importlib.import_module(name);print(name,'OK',getattr(m,'__version__',''))
        if name=='spharm':assert hasattr(m,'Spharmt') and hasattr(m,'gaussian_lats_wts')
    except Exception as e:print(name,'FAILED',e);bad.append(name)
print('ncl:',shutil.which('ncl') or '缺失：仅重用已有滤波时可以跳过第一阶段')
if bad:sys.exit(1)
