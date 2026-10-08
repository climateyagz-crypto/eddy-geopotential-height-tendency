#!/usr/bin/env python3
"""单月全球 Gan 形式近似诊断：无辐散风通量、去混叠球谐、守恒压力离散。
这是数值一致性试验，不是宣称赤道处准地转理论成立或完全复现 Gan。
"""
from pathlib import Path
import sys
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
import bootstrap
import argparse
from pathlib import Path
from contextlib import ExitStack
from datetime import datetime,timezone
import json,hashlib,calendar,time,copy
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.fft import next_fast_len
from numpy.polynomial.legendre import leggauss
from netCDF4 import Dataset,date2num,num2date
from audit_readers import InputFile,check_grid,BANDPASS
from global_solver import Solver,basis,need,self_test,RADIUS,OMEGA,G,RD

from config import ROOT, BACKGROUND, OUTPUT, TERMS, TRUNCATION, CHUNK_DAYS, START_YEAR, END_YEAR
VERSION = 'global_rotational_Gan_form_Galerkin_pressure_FV_v1_1_double_flux'


def double_advection(zgrid,tgrid,mu,weights,truncation):
    """Reconstruct a self-consistent double-precision rotational wind and gradients.

    SHT analysis supplies the resolved vorticity/temperature fields. Project
    these fields on one normalized scalar basis; invert zeta=laplacian(psi),
    reconstruct V=k cross grad(psi), and compute div(V*A)=V dot grad(A).
    No angular-momentum mode or forcing mean is deleted.
    """
    nlat,nlon,nt = zgrid.shape
    zf,tf = [np.fft.rfft(np.asarray(x,'f8'),axis=1)/nlon for x in (zgrid,tgrid)]
    spectra = [np.zeros_like(zf) for _ in range(6)]
    c = np.sqrt(1-mu*mu)[:,None]
    for m in range(truncation+1):
        B = basis(mu,m,truncation)
        ell = np.arange(m,truncation+1,dtype=float)
        deriv = -mu[:,None]*B*ell[None,:]
        if B.shape[1]>1:
            n = ell[1:]
            factor = np.sqrt((2*n+1)/(2*n-1)*(n*n-m*m))
            deriv[:,1:] += B[:,:-1]*factor[None,:]
        deriv /= (1-mu*mu)[:,None]
        zcoef = B.T@(weights[:,None]*zf[:,m,:])
        tcoef = B.T@(weights[:,None]*tf[:,m,:])
        inv = np.zeros_like(ell);nz = ell>0
        inv[nz] = -RADIUS**2/(ell[nz]*(ell[nz]+1))
        psicoef = inv[:,None]*zcoef
        spectra[0][:,m,:] = -c/RADIUS*(deriv@psicoef)
        spectra[1][:,m,:] = (B@(1j*m*psicoef))/(RADIUS*c)
        spectra[2][:,m,:] = B@(1j*m*zcoef)
        spectra[3][:,m,:] = deriv@zcoef
        spectra[4][:,m,:] = B@(1j*m*tcoef)
        spectra[5][:,m,:] = deriv@tcoef
    u,v,zlam,zmu,tlam,tmu = [np.fft.irfft(x*nlon,n=nlon,axis=1) for x in spectra]
    cosine = c[:,None,:]
    advz = u*zlam/(RADIUS*cosine)+v*cosine*zmu/RADIUS
    advt = u*tlam/(RADIUS*cosine)+v*cosine*tmu/RADIUS
    return advz,advt


def double_flux_self_test():
    T=16;mu,w=leggauss(2*T+1);mu,w=mu[::-1],w[::-1]
    lon=np.arange(72)[None,:,None]*2*np.pi/72
    s=mu[:,None,None];c=np.sqrt(1-mu*mu)[:,None,None]
    U=10.;phase=np.array([1.,.5])[None,None,:]
    # psi=U*a*(cos(phi)*cos(lambda)+0.2*P2(sin(phi))).
    z=(-2*U/RADIUS*c*np.cos(lon)-6*.2*U/RADIUS*(3*s*s-1)/2)*phase
    temp=2*c*np.cos(lon)*np.ones_like(phase)
    advz,advt=double_advection(z,temp,mu,w,T)
    u=U*(s*np.cos(lon)-.6*s*c)*phase
    v=-U*np.ones_like(s)*np.sin(lon)*phase
    zlam=2*U/RADIUS*c*np.sin(lon)*phase
    zmu=(2*U/RADIUS*s/c*np.cos(lon)-3.6*U/RADIUS*s)*phase
    expected=u*zlam/(RADIUS*c)+v*c*zmu/RADIUS
    np.testing.assert_allclose(advz,expected,atol=1e-22,rtol=1e-10)
    mean=np.sum(w*np.mean(advz,axis=(1,2)))/2
    fm=np.sum(w*mu*np.mean(advz,axis=(1,2)))/2
    scale=np.max(np.abs(advz))
    need(max(abs(mean),abs(fm))/scale<1e-12,'Angular momentum identity failed')
    print('PASS: double-precision rotational advection, analytic derivatives, zero global circulation and angular-momentum tendency',flush=True)


def read(v,key=slice(None)):
    a = np.asarray(np.ma.asarray(v[key],dtype='f8').filled(np.nan))
    need(np.isfinite(a).all() and (np.abs(a)<1e30).all(),f'{v.name}: missing/invalid')
    return a


def backend_test(src,dst,lat,mu,truncation):
    # Independent smooth nondivergent wind from psi=U*a*cos(phi)*cos(lambda).
    U = 10.
    phi = np.deg2rad(lat)[:,None];lam = np.arange(src.nlon)[None,:]*2*np.pi/src.nlon
    u,v = U*np.sin(phi)*np.cos(lam),-U*np.ones_like(phi)*np.sin(lam)
    lg = np.arange(dst.nlon)[None,:]*2*np.pi/dst.nlon
    expected_u = U*mu[:,None]*np.cos(lg)
    expected_v = -U*np.ones((len(mu),1))*np.sin(lg)
    expected_z = -2*U/RADIUS*np.sqrt(1-mu[:,None]**2)*np.cos(lg)
    # The analytic field has degree 1. At full T179, float32 wind roundoff
    # creates tiny high-degree spectra; curl amplifies them by degree/a.
    # Separate a low-degree sign/amplitude test from the full-band error test.
    weights = leggauss(len(mu))[1][::-1]
    metrics = {}
    for label,T in [('low_degree',min(8,truncation)),('full_band',truncation)]:
        zz,dd = src.getvrtdivspec(np.asarray(u,'f4'),np.asarray(v,'f4'),ntrunc=T)
        ug,vg = [np.asarray(x,'f8') for x in dst.getuv(zz,np.zeros_like(dd))]
        zg = np.asarray(dst.spectogrd(zz),'f8')
        eu,ev,ez = ug-expected_u,vg-expected_v,zg-expected_z
        wind_max = float(max(abs(eu).max(),abs(ev).max())/U)
        wind_rms = float(np.sqrt(np.sum(weights*np.mean(eu*eu+ev*ev,axis=1))/2))
        amplitude = 2*U/RADIUS
        curl_max = float(abs(ez).max()/amplitude)
        curl_rms = float(np.sqrt(np.sum(weights*np.mean(ez*ez,axis=1))/2)/amplitude)
        pos = np.unravel_index(np.argmax(abs(ez)),ez.shape)
        # On degree <= T vector fields, ||curl(error)||_2 is bounded by
        # sqrt(T(T+1))/a * ||wind_error||_2. The last term accounts for
        # separate float32 scalar/vector synthesis rounding.
        bound = np.sqrt(T*(T+1))*wind_rms/(RADIUS*amplitude)+32*np.finfo('f4').eps
        metrics[label] = dict(truncation=T,wind_relative_max=wind_max,
            curl_relative_max=curl_max,curl_relative_area_RMS=curl_rms,
            curl_roundoff_bound=float(bound),max_error_latitude=float(np.rad2deg(np.arcsin(mu[pos[0]]))))
        print('BACKEND_METRICS',label,json.dumps(metrics[label]),flush=True)
        need(wind_max<1e-5,f'{label}: spherical vector remapping failed; metrics={metrics[label]}')
        if label=='low_degree':
            need(curl_max<1e-5,f'Low-degree spherical curl sign/normalization failed; metrics={metrics[label]}')
        else:
            need(curl_rms<1e-5 and curl_rms<=bound,
                 f'Full-band spherical curl precision failed; metrics={metrics[label]}')
    print('PASS: installed spherical backend; low-degree sign/normalization and full-band precision bound',flush=True)
    return metrics


def load_background(args,grid,mu):
    with Dataset(args.background) as nc:
        need(getattr(nc,'background_derivative_scheme','')=='theta_derivative_second_order_except_bottom_secant_v2',
             'Unknown background version')
        need(nc['lev'].units=='Pa','Background pressure unit must be Pa')
        p,lat = read(nc['lev']),read(nc['lat'])
        need(np.allclose(p,grid.p*100,rtol=0,atol=1e-5),'Background pressure mismatch')
        need(np.allclose(lat,grid.lat,rtol=0,atol=1e-5),'Background latitude mismatch')
        sigma = read(nc['sigma'],args.month-1)
        need(np.all(sigma>0),'Nonpositive stability; no clipping')
        # Treat sigma as a smooth positive coefficient. Monotone interpolation
        # avoids artificial undershoots; no change to its monthly/latitude definition.
        x = np.sin(np.deg2rad(lat))[::-1]
        sg = PchipInterpolator(x,sigma[:,::-1],axis=1)(mu)
        need(np.isfinite(sg).all() and np.all(sg>0),'Invalid interpolated stability')
        ix = np.flatnonzero((p>=10000)&(p<=100000))
        need(np.isclose(p[ix[0]],10000) and np.isclose(p[ix[-1]],100000),'100/1000 hPa endpoints required')
    return p[ix],ix,sg[ix]


def build_forcing(args):
    """读取阶段2的单独项；没有缓存时现场计算，使用相同数值核心。"""
    from rotational_terms import build_terms
    from types import SimpleNamespace
    path=args.terms/f'terms.{args.year}.{args.month:02d}.nc'
    if path.exists():
        with Dataset(path) as nc:
            need((int(nc.year),int(nc.month))==(args.year,args.month),'单独项月份不匹配')
            need(nc.truncation==args.truncation and nc.method=='double_advection_v1_1','单独项版本不匹配')
            p,mu,w,divt,divz,lat,lon=[read(nc[n]) for n in ('lev','mu','weights','div_T','div_zeta','output_lat','output_lon')]
            meta=json.loads(nc.metadata)
            for key,name in [('source_u','uwnd'),('source_v','vwnd'),('source_T','air')]:
                expected=args.bandpass/name/f'{name}.daymean.global.1deg.{args.year}.{args.month:02d}.BPF_L_2.5_8d.nc'
                need(meta[key]==str(expected),'单独项输入路径不同，请重新计算单独项')
            need(meta['n_samples']==calendar.monthrange(args.year,args.month)[1],'单独项采样不完整')
    else:
        print('单独项缓存不存在：现场计算',flush=True)
        p,mu,w,divt,divz,lat,lon,meta=build_terms(args)
    # 背景必须在完整原始压力层计算，然后截取到反演层。
    with Dataset(args.background) as nc: fullp=read(nc['lev'])
    grid=SimpleNamespace(p=fullp/100,lat=lat)
    bp,ix,sigma=load_background(args,grid,mu)
    need(np.array_equal(p,bp),'单独项与背景压力层不一致')
    return p,mu,w,sigma,divt,divz,lat,lon,meta


def solve(args):
    output = args.out/f'Ztend.{args.year}.{args.month:02d}.nc'
    need(args.overwrite or not output.exists(),f'{output}: exists; use --overwrite to replace')
    forcing_output = args.out/f'forcing_rotational.{args.year}.{args.month:02d}.nc'
    need(args.overwrite or not forcing_output.exists(),f'{forcing_output}: exists; use --overwrite to replace')
    p,mu,w,sigma,divt,divz,lat,lon,meta = build_forcing(args)
    args.out.mkdir(parents=True,exist_ok=True)
    forcing_temp = forcing_output.with_name(forcing_output.name+'.partial')
    with Dataset(forcing_temp,'w') as nc:
        for name,size in [('lev',len(p)),('gaussian_lat',len(mu)),('gaussian_lon',divt.shape[-1])]:nc.createDimension(name,size)
        for name,values,units in [('lev',p,'Pa'),('gaussian_lat',np.rad2deg(np.arcsin(mu)),'degrees_north'),
                                 ('gaussian_lon',np.arange(divt.shape[-1])*360/divt.shape[-1],'degrees_east')]:
            var=nc.createVariable(name,'f8',(name,));var.units=units;var[:]=values
        var=nc.createVariable('area_quadrature_weight','f8',('gaussian_lat',));var[:]=w/2;var.units='1'
        var=nc.createVariable('sigma','f8',('lev','gaussian_lat'));var[:]=sigma;var.units='m2 s-2 Pa-2'
        for name,values,units in [('div_T',divt,'K s-1'),('div_zeta',divz,'s-2')]:
            var=nc.createVariable(name,'f8',('lev','gaussian_lat','gaussian_lon'),zlib=True,complevel=3)
            var[:]=values;var.units=units
        nc.year,nc.month=args.year,args.month
        nc.stage='Rotational eddy forcing only; no height tendency inversion yet'
        nc.version=VERSION;nc.spherical_truncation=args.truncation
        nc.input_metadata=json.dumps(meta);nc.background_file=str(args.background)
        nc.background_sha256=hashlib.sha256(args.background.read_bytes()).hexdigest()
        nc.forcing_mean_removed=0
    forcing_temp.replace(forcing_output)
    print('SAVED FORCING',forcing_output,flush=True)
    op = Solver(p,mu,w,sigma,args.truncation)
    h = RD/(p[:,None,None]*sigma[:,:,None])*divt
    sh = op.heat_source(h)
    sv = -2*OMEGA*mu[None,:,None]/G*divz
    gt,gb = RD/(G*p[0])*divt[0],RD/(G*p[-1])*divt[-1]
    zero = np.zeros_like(gt)
    fields,diag = op.solve_many({'heat':sh,'vort':sv,'total_independent':sh+sv},
        {'heat':(gt,gb),'vort':(zero,zero),'total_independent':(gt,gb)},lat,len(lon))
    diff = fields['total_independent']-fields['heat']-fields['vort']
    add = float(np.abs(diff).max()/max(np.abs(fields['heat']).max()+np.abs(fields['vort']).max(),1e-300))
    need(add<1e-8,f'Independent additivity failed: {add:g}')
    args.out.mkdir(parents=True,exist_ok=True)
    temp = output.with_name(output.name+'.partial')
    with Dataset(temp,'w') as nc:
        for name,n in [('time',1),('lev',len(p)),('lat',len(lat)),('lon',len(lon))]:nc.createDimension(name,n)
        tv = nc.createVariable('time','f8',('time',));tv.units='hours since 1900-01-01';tv.calendar='standard'
        tv[:] = date2num([datetime(args.year,args.month,1)],tv.units)
        for name,value,unit in [('lev',p,'Pa'),('lat',lat,'degrees_north'),('lon',lon,'degrees_east')]:
            var = nc.createVariable(name,'f8',(name,));var.units=unit;var[:]=value
        nc['lev'].positive='down'
        for name,value in [('Ztend_heat',fields['heat']),('Ztend_vort',fields['vort']),
                           ('Ztend_total',fields['heat']+fields['vort']),('Ztend_total_independent',fields['total_independent'])]:
            var = nc.createVariable(name,'f8',('time','lev','lat','lon'),zlib=True,complevel=3)
            var.units='m day-1';var[0]=value*86400
        nc.case='GLOBAL_ROTATIONAL_GAN_FORM_APPROXIMATE';nc.version=VERSION
        nc.interpretation=('Formal global extension of Gan-form height tendency operator with nondivergent eddy winds; '
                           'not exact Gan reproduction and not a validated equatorial balance theory')
        nc.wind_choice='Nondivergent wind reconstructed from relative-vorticity spectra; temperature remains ERA5 bandpass T'
        nc.filter_definition='Existing daily Lanczos 2.5-8 d input reused; temporal filter not changed'
        nc.spherical_truncation=args.truncation
        nc.spherical_method='Normalized associated-Legendre Galerkin; double-precision rotational advection and dealiased Gaussian quadrature'
        nc.pressure_method='Finite-volume pressure face flux; endpoint cells retain their PDE and prescribed boundary flux'
        nc.vertical_boundary='100-1000 hPa; heat chi_p=Rd/(g*p)*div_T; vort chi_p=0'
        nc.global_reference='One spherical-area and pressure-volume weighted 3D mean zero per component'
        nc.forcing_mean_removed=0
        nc.background_file=str(args.background)
        nc.background_sha256=hashlib.sha256(args.background.read_bytes()).hexdigest()
        nc.background_definition='Original calendar-month zonal-mean background, latitude retained; positive PCHIP interpolation in sin(latitude)'
        nc.independent_additivity_relative=add
        nc.solver_diagnostics=json.dumps(diag)
        nc.input_metadata=json.dumps(meta)
        nc.history=datetime.now(timezone.utc).isoformat()
        nc.Rd,nc.g,nc.earth_radius_m,nc.earth_rotation_s_1=RD,G,RADIUS,OMEGA
    temp.replace(output)
    print('DIAGNOSTICS',json.dumps(diag),flush=True)
    print(f'ADDITIVITY={add:.3e}; SAVED {output}',flush=True)


def completed(args,background_digest):
    """Verify result identity/settings and recorded checks before resuming."""
    result=args.out/f'Ztend.{args.year}.{args.month:02d}.nc'
    forcing=args.out/f'forcing_rotational.{args.year}.{args.month:02d}.nc'
    if not result.exists():return False,'no completed result'
    try:
        need(forcing.exists(),'forcing file missing')
        with Dataset(args.background) as bg:
            bp=read(bg['lev']);expected_p=bp[(bp>=10000)&(bp<=100000)]
        with Dataset(result) as nc:
            need(nc.case=='GLOBAL_ROTATIONAL_GAN_FORM_APPROXIMATE','case mismatch')
            need(nc.version==VERSION,'numerical version mismatch')
            need(nc.spherical_truncation==args.truncation,'spherical truncation mismatch')
            need(nc.background_sha256==background_digest,'background contents changed')
            need(int(nc.forcing_mean_removed)==0,'forcing mean was removed')
            need(np.all(np.diff(read(nc['lev']))>0) and nc['lev'].units=='Pa','pressure coordinates invalid')
            need(np.isclose(read(nc['lev'])[0],10000) and np.isclose(read(nc['lev'])[-1],100000),'pressure domain mismatch')
            need(np.array_equal(read(nc['lev']),expected_p),'background pressure layers mismatch')
            need(np.array_equal(read(nc['lat']),np.arange(90,-91,-1)) and
                 np.array_equal(read(nc['lon']),np.arange(360)),'output grid mismatch')
            dates=num2date(read(nc['time']),nc['time'].units,calendar=nc['time'].calendar)
            need(len(dates)==1 and (dates[0].year,dates[0].month)==(args.year,args.month),'output month mismatch')
            diag=json.loads(nc.solver_diagnostics)
            need(set(diag)=={'heat','vort','total_independent'},'missing component diagnostics')
            for item in diag.values():
                need(item['compatibility_relative']<=1e-8 and item['original_Galerkin_residual_relative']<=1e-8
                     and item['forcing_mean_removed'] is False,'recorded numerical checks failed')
            need(nc.independent_additivity_relative<1e-8,'recorded additivity failed')
            meta=json.loads(nc.input_metadata)
            for key,name in [('source_u','uwnd'),('source_v','vwnd'),('source_T','air')]:
                expected=args.bandpass/name/f'{name}.daymean.global.1deg.{args.year}.{args.month:02d}.BPF_L_2.5_8d.nc'
                need(meta[key]==str(expected),'bandpass input path mismatch')
            need(meta['n_samples']==calendar.monthrange(args.year,args.month)[1],'monthly sample count mismatch')
            for name in ('Ztend_heat','Ztend_vort','Ztend_total','Ztend_total_independent'):
                v=nc[name]
                need(v.dimensions==('time','lev','lat','lon') and v.units=='m day-1','result variable invalid')
                read(v,(0,0,0,0));read(v,(0,-1,-1,-1))
        with Dataset(forcing) as nc:
            need(nc.version==VERSION and nc.spherical_truncation==args.truncation,'forcing version/truncation mismatch')
            need((int(nc.year),int(nc.month))==(args.year,args.month),'forcing month mismatch')
            need(int(nc.forcing_mean_removed)==0,'forcing mean was removed')
            need(np.array_equal(read(nc['lev']),expected_p),'forcing pressure layers mismatch')
            need(len(nc.dimensions['gaussian_lat'])==2*args.truncation+1 and
                 len(nc.dimensions['gaussian_lon'])==2*next_fast_len(2*args.truncation+1),'forcing grid mismatch')
            need(nc['div_T'].units=='K s-1' and nc['div_zeta'].units=='s-2','forcing units mismatch')
            read(nc['div_T'],(0,0,0));read(nc['div_zeta'],(-1,-1,-1))
        return True,'compatible completed result'
    except Exception as exc:
        return False,str(exc)


def batch(args):
    need(args.start_year<=args.end_year,'Invalid batch years')
    need(args.background.exists(),f'Background missing: {args.background}')
    digest=hashlib.sha256(args.background.read_bytes()).hexdigest()
    count=12*(args.end_year-args.start_year+1);skipped=done=0
    start=time.monotonic()
    for year in range(args.start_year,args.end_year+1):
        for month in range(1,13):
            local=copy.copy(args);local.year,local.month=year,month
            output=local.out/f'Ztend.{year}.{month:02d}.nc'
            if output.exists() and not args.overwrite:
                ok,reason=completed(local,digest)
                need(ok,f'{output}: cannot resume: {reason}. Inspect this month; do not silently mix results.')
                skipped+=1
                print(f'SKIP {year}-{month:02d}: verified completed result ({done+skipped}/{count})',flush=True)
                continue
            # A failed inversion may have left its forcing diagnostic. Retry only
            # that incomplete month; completed results are still verified/skipped.
            forcing=local.out/f'forcing_rotational.{year}.{month:02d}.nc'
            if not output.exists() and forcing.exists():
                local.overwrite=True
                print(f'RETRY {year}-{month:02d}: regenerate incomplete forcing diagnostic',flush=True)
            print(f'BEGIN {year}-{month:02d} ({done+skipped+1}/{count})',flush=True)
            month_start=time.monotonic()
            try:solve(local)
            except Exception:
                print(f'BATCH STOPPED {year}-{month:02d}; completed earlier months retained',flush=True)
                raise
            done+=1
            print(f'END {year}-{month:02d}: seconds={time.monotonic()-month_start:.1f}',flush=True)
    print(f'BATCH COMPLETED calculated={done}; skipped={skipped}; total={count}; seconds={time.monotonic()-start:.1f}',flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['self-test','solve','batch'])
    ap.add_argument('--year',type=int,default=2021)
    ap.add_argument('--month',type=int,choices=range(1,13),default=1)
    ap.add_argument('--start-year',type=int,default=START_YEAR)
    ap.add_argument('--end-year',type=int,default=END_YEAR)
    ap.add_argument('--truncation',type=int,default=TRUNCATION,help='Triangular spherical truncation, default retains supported 1-degree bandwidth')
    ap.add_argument('--chunk-days',type=int,default=CHUNK_DAYS)
    ap.add_argument('--terms',type=Path,default=TERMS)
    ap.add_argument('--bandpass',type=Path,default=BANDPASS)
    ap.add_argument('--background',type=Path,default=BACKGROUND)
    ap.add_argument('--out',type=Path,default=OUTPUT)
    ap.add_argument('--assume-pressure-unit',choices=['Pa','hPa'])
    ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args()
    need(args.chunk_days>0,'chunk-days must be positive')
    if args.stage=='self-test':self_test();double_flux_self_test()
    elif args.stage=='batch':batch(args)
    else:solve(args)
