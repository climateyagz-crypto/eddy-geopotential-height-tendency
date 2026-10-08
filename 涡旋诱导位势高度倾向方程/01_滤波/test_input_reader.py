"""合成packed int16、Pa、反向压力、跨年别名及同步重排测试；不需NCL。"""
from pathlib import Path
import sys,tempfile,calendar
from datetime import datetime,timedelta
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import bootstrap
import numpy as np
from netCDF4 import Dataset,date2num
from audit_readers import InputFile,check_grid

def run():
    with tempfile.TemporaryDirectory() as root:
        sources=[]
        try:
            for year,month,modern in [(2021,12,False),(2022,1,True)]:
                path=Path(root)/f'{year}.nc';nt=4*calendar.monthrange(year,month)[1]
                pd='pressure_level' if modern else 'level';td='valid_time' if modern else 'time'
                p=np.array([850.,500.,100.]) if modern else np.array([100.,500.,850.])
                lat=np.arange(-90,91.) if modern else np.arange(90,-91,-1.)
                lon=np.arange(-180,180.) if modern else np.arange(360.)
                with Dataset(path,'w') as nc:
                    for name,a,unit in [(pd,p*100 if modern else p,'Pa' if modern else 'hPa'),('latitude',lat,'degrees_north'),('longitude',lon,'degrees_east'),(td,np.arange(nt)*6,'hours since 1900-01-01')]:
                        nc.createDimension(name,len(a));v=nc.createVariable(name,'f8',(name,));v.units=unit
                        v[:]=date2num([datetime(year,month,1)+timedelta(hours=6*i) for i in range(nt)],unit) if name==td else a
                    v=nc.createVariable('u','i2',(td,pd,'latitude','longitude'),zlib=True)
                    v.units='m/s';v.scale_factor=.01 if modern else .02;v.add_offset=5. if modern else -3.
                    for k,pk in enumerate(p):
                        field=pk/100+lat[:,None]/100+np.mod(lon,360)[None,:]/100
                        v[:,k]=field[None,:,:]
                src=InputFile(path,('u',),year,month,False,None,raw_wind=True);sources.append(src)
                for k,pk in enumerate(src.p):
                    expected=pk/100+src.lat[:,None]/100+src.lon[None,:]/100
                    np.testing.assert_allclose(src.level(k)[0],expected,atol=.011,rtol=0)
            check_grid(*sources)
            print('PASS: per-file unpacking, time aliases, Pa/hPa, reordered coordinates and cross-year alignment')
        finally:
            for src in sources:src.close()
if __name__=='__main__':run()
