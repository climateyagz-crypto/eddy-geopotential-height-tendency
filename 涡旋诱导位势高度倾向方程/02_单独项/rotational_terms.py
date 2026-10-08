from pathlib import Path
import sys
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
import bootstrap
"""单独项：先从涡度重建无辐散风，再求热/涡度通量散度月平均。
同一双精度球谐基底求导，Gaussian去混叠，不删去强迫均值。
"""
from contextlib import ExitStack
import json,calendar,argparse
import numpy as np
from scipy.fft import next_fast_len
from numpy.polynomial.legendre import leggauss
from netCDF4 import Dataset
from config import *
from audit_readers import InputFile,check_grid
from global_solver import basis,need,RADIUS,OMEGA,G,RD
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

def build_terms(args):
    from spharm import Spharmt,gaussian_lats_wts
    T = args.truncation
    with ExitStack() as stack:
        def bp(name,candidates,temperature=False):
            path = args.bandpass/name/f'{name}.daymean.global.1deg.{args.year}.{args.month:02d}.BPF_L_2.5_8d.nc'
            obj = InputFile(path,candidates,args.year,args.month,True,args.assume_pressure_unit,
                            daily_temperature=temperature)
            stack.callback(obj.close)
            return obj
        u,v,t = bp('uwnd',('uwnd','u')),bp('vwnd',('vwnd','v')),bp('air',('t','air'),True)
        check_grid(u,v);check_grid(u,t)
        need(u.dates==v.dates==t.dates,'Daily wind/temperature timestamps differ')
        need(T>=8 and T<=min(len(u.lat)-2,len(u.lon)//2-1),'Unsupported spherical truncation')
        ng = 2*T+1
        nlon = 2*next_fast_len(2*T+1)
        mu,weights = leggauss(ng);mu,weights = mu[::-1],weights[::-1]
        glat,gweights = gaussian_lats_wts(ng)
        need(np.allclose(np.sin(np.deg2rad(glat)),mu,atol=1e-7,rtol=0),'Gaussian grid ordering mismatch')
        src = Spharmt(len(u.lon),len(u.lat),rsphere=RADIUS,gridtype='regular',legfunc='stored')
        dst = Spharmt(nlon,ng,rsphere=RADIUS,gridtype='gaussian',legfunc='stored')
        backend_metrics = backend_test(src,dst,u.lat,mu,T)
        pall=u.p*100
        ix=np.flatnonzero((pall>=10000)&(pall<=100000))
        need(len(ix)>2 and pall[ix[0]]==10000 and pall[ix[-1]]==100000,'需要100/1000hPa端点')
        p=pall[ix]
        divt = np.empty((len(p),ng,nlon));divz = np.empty_like(divt)
        for j,k in enumerate(ix):
            ud,vd,td = u.level(k),v.level(k),t.level(k)
            need(np.isfinite(ud).all() and np.isfinite(vd).all() and np.isfinite(td).all(),
                 'Missing bandpass input; no filling permitted')
            need(max(np.abs(ud).max(),np.abs(vd).max())<500 and np.abs(td).max()<100,
                 'Bandpass magnitude invalid')
            sums = [np.zeros((ng,nlon)) for _ in range(2)]
            for start in range(0,u.nt,args.chunk_days):
                end = min(start+args.chunk_days,u.nt)
                a,b,c = [np.ascontiguousarray(np.moveaxis(x[start:end],0,-1),dtype='f4') for x in (ud,vd,td)]
                zz,dd = src.getvrtdivspec(a,b,ntrunc=T)
                # Removing irrotational wind is a physical diagnostic choice;
                # it is NOT a subtraction of the forcing's global mean.
                zg = dst.spectogrd(zz)
                tg = dst.spectogrd(src.grdtospec(c,ntrunc=T))
                advz,advt = double_advection(zg,tg,mu,weights,T)
                for accum,term in zip(sums,(advt,advz)):
                    accum += np.sum(term,axis=2,dtype='f8')
            divt[j],divz[j] = [x/u.nt for x in sums]
            meanz = float(np.sum(weights*np.mean(divz[j],axis=-1))/2)
            meansv = float(np.sum(weights*(-2*OMEGA*mu/G)*np.mean(divz[j],axis=-1))/2)
            print(f'FORCING {p[j]/100:g}hPa div_zeta_mean={meanz:.3e}; S_vort_mean={meansv:.3e}',flush=True)
        return p,mu,weights,divt,divz,u.lat.copy(),u.lon.copy(),dict(
            source_u=str(u.path),source_v=str(v.path),source_T=str(t.path),n_samples=u.nt,
            gaussian_nlat=ng,gaussian_nlon=nlon,backend_test_metrics=backend_metrics)

def save_terms(args):
    path=args.terms/f'terms.{args.year}.{args.month:02d}.nc'
    need(args.overwrite or not path.exists(),f'{path}已存在；覆盖用--overwrite')
    p,mu,w,dt,dz,lat,lon,meta=build_terms(args)
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.partial')
    with Dataset(tmp,'w') as nc:
        for name,a in [('lev',p),('mu',mu),('lon',np.arange(dt.shape[-1])*360/dt.shape[-1]),('output_lat',lat),('output_lon',lon)]:
            nc.createDimension(name,len(a));nc.createVariable(name,'f8',(name,))[:]=a
        nc['lev'].units='Pa'
        nc.createVariable('weights','f8',('mu',))[:]=w
        for name,a,unit in [('div_T',dt,'K s-1'),('div_zeta',dz,'s-2')]:
            v=nc.createVariable(name,'f8',('lev','mu','lon'),zlib=True,complevel=3);v[:]=a;v.units=unit
        nc.year,nc.month=args.year,args.month;nc.truncation=args.truncation
        nc.method='double_advection_v1_1';nc.metadata=json.dumps(meta)
    tmp.replace(path);print('SAVED TERMS',path,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['solve','batch','self-test'])
    ap.add_argument('--year',type=int,default=2021);ap.add_argument('--month',type=int,default=1)
    ap.add_argument('--start-year',type=int,default=START_YEAR);ap.add_argument('--end-year',type=int,default=END_YEAR)
    ap.add_argument('--bandpass',type=Path,default=BANDPASS);ap.add_argument('--terms',type=Path,default=TERMS)
    ap.add_argument('--truncation',type=int,default=TRUNCATION);ap.add_argument('--chunk-days',type=int,default=CHUNK_DAYS)
    ap.add_argument('--assume-pressure-unit',choices=['Pa','hPa']);ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args();need(args.chunk_days>0 and 1<=args.month<=12,'参数范围错误')
    if args.stage=='self-test':double_flux_self_test()
    elif args.stage=='solve':save_terms(args)
    else:
        need(args.start_year<=args.end_year,'年份范围错误')
        for y in range(args.start_year,args.end_year+1):
            for m in range(1,13):args.year,args.month=y,m;save_terms(args)
