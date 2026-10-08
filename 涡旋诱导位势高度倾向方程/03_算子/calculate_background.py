from pathlib import Path
import sys
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
import bootstrap
"""月气候态纬向平均温度、位温、静力稳定度；完整原始压力层求导。"""
from pathlib import Path
import argparse,calendar
from datetime import datetime,timezone
import numpy as np
from config import ROOT,BASIC,BACKGROUND,START_YEAR,END_YEAR
from audit_readers import ALIASES
RD,CP,P0,G=287.,1004.,100000.,9.80665
RADIUS,OMEGA=6371200.,7.292e-5
BG_SCHEME='theta_derivative_second_order_except_bottom_secant_v2'
TU='hours since 1900-01-01 00:00:00'
def need(ok, message):
    if not ok:
        raise ValueError(message)

def read(v, key=slice(None)):
    a = np.asarray(np.ma.asarray(v[key], dtype='f8').filled(np.nan))
    need(np.all(np.isfinite(a)), f'{v.name}: missing/NaN/Inf; do not fill with zero')
    need(np.all(np.abs(a) < 1e30), f'{v.name}: undeclared fill/sentinel detected')
    return a

def dp(a, p, axis=0):
    """Second-order derivatives on the ORIGINAL nonuniform pressure grid (Pa)."""
    p = np.asarray(p, dtype='f8')
    need(len(p) >= 3 and np.all(np.diff(p) > 0), 'Need >=3 ascending pressure levels')
    return np.gradient(a, p, axis=axis, edge_order=2)

def stability(tb, p):
    # tb: (month, lev, lat). Build theta first, then differentiate theta.
    c = (P0 / p) ** (RD / CP)
    theta = tb * c[None, :, None]
    theta_p = dp(theta, p, axis=1)
    # At the largest pressure, quadratic extrapolation can reverse the sign
    # even when both adjacent layer slopes are negative. Use the bottom-layer
    # secant everywhere at this boundary (not only where the old sign failed).
    # Interior and top derivatives retain the original second-order scheme.
    theta_p[:, -1, :] = (theta[:, -1, :] - theta[:, -2, :]) / (p[-1] - p[-2])
    sigma = -(RD * tb / p[None, :, None]) * theta_p / theta
    return theta, theta_p, sigma

def annual(root, name, year):
    return root / name / f'{name}.monthly.global.1deg.{year}.nc'

class Monthly:
    def __init__(self, path, name, year, assume):
        from netCDF4 import Dataset, num2date
        self.path, self.name = Path(path), name
        self.nc = Dataset(path)
        self.nc.set_auto_maskandscale(True)
        try:
            self.v = self.nc[name]
            need(self.v.ndim == 4, f'{path}: {name} must have four dimensions')
            self.dims, cv = {}, {}
            for kind, aliases in ALIASES.items():
                ds = [d for d in self.v.dimensions if d in aliases]
                need(len(ds) == 1, f'{path}: cannot identify {kind} dimension')
                self.dims[kind] = ds[0]
                cs = [self.nc.variables[n] for n in dict.fromkeys((ds[0],) + aliases)
                      if n in self.nc.variables
                      and self.nc.variables[n].dimensions == (ds[0],)]
                need(bool(cs), f'{path}: no coordinate for {kind}')
                cv[kind] = cs[0]
            p = read(cv['lev'])
            pu = str(getattr(cv['lev'], 'units', '')).strip().lower()
            if not pu:
                need(assume is not None, f'{path}: absent pressure unit; use explicit --assume-pressure-unit')
                pu = assume.lower()
                print(f'NOTE {path}: assuming pressure unit {pu}', flush=True)
            if pu in ('hpa', 'mb', 'mbar', 'millibar', 'millibars'):
                p *= 100
            else:
                need(pu in ('pa', 'pascal', 'pascals'), f'{path}: unknown pressure unit {pu}')
            la, lo = read(cv['lat']), np.mod(read(cv['lon']), 360)
            self.ix = {'lev': np.argsort(p), 'lat': np.argsort(la)[::-1], 'lon': np.argsort(lo)}
            self.p, self.lat, self.lon = p[self.ix['lev']], la[self.ix['lat']], lo[self.ix['lon']]
            need(len(p) >= 3 and np.all(np.diff(self.p) > 0)
                 and self.p[0] > 0 and self.p[-1] <= 110000, f'{path}: invalid pressure')
            need(len(la) >= 5 and np.allclose(self.lat, np.linspace(90, -90, len(la)), atol=1e-5, rtol=0),
                 f'{path}: require pole-inclusive global regular latitude')
            need(len(lo) >= 8 and np.allclose(self.lon, np.arange(len(lo))*360/len(lo), atol=1e-5, rtol=0),
                 f'{path}: require global regular longitude without duplicate endpoint')
            tv = cv['time']
            dates = num2date(read(tv), tv.units, calendar=getattr(tv, 'calendar', 'standard'))
            need([(d.year, d.month) for d in dates] == [(year, m) for m in range(1, 13)],
                 f'{path}: require exactly Jan-Dec in requested year')
            unit = str(getattr(self.v, 'units', '')).lower().replace(' ', '').replace('**', '^')
            allowed = {'ut': ('k*m/s', 'kms-1', 'kms^-1', 'km/s'),
                       'vt': ('k*m/s', 'kms-1', 'kms^-1', 'km/s'),
                       'uzeta': ('ms-2', 'ms^-2', 'm/s2', 'm/s^2'),
                       'vzeta': ('ms-2', 'ms^-2', 'm/s2', 'm/s^2'),
                       'T_month': ('k', 'kelvin')}
            need(unit in allowed[name], f'{path}: unexpected units {unit!r} for {name}')
            if 'n_samples' in self.nc.variables:
                ns = self.nc['n_samples']
                need(ns.dimensions == (self.dims['time'], self.dims['lev']),
                     f'{path}: unexpected n_samples dimensions')
                counts = read(ns)
                expected = np.array([calendar.monthrange(year, m)[1] for m in range(1, 13)])
                if name == 'T_month':
                    expected *= 4
                need(np.all(counts == expected[:, None]), f'{path}: incomplete monthly sampling')
        except Exception:
            self.nc.close()
            raise

    def month(self, m):
        dims = list(self.v.dimensions)
        axis = dims.index(self.dims['time'])
        key = [slice(None)] * 4
        key[axis] = m
        a = read(self.v, tuple(key))
        remaining = [d for i, d in enumerate(dims) if i != axis]
        a = a.transpose([remaining.index(self.dims[k]) for k in ('lev', 'lat', 'lon')])
        for axis, kind in enumerate(('lev', 'lat', 'lon')):
            a = np.take(a, self.ix[kind], axis=axis)
        return a

    def close(self):
        self.nc.close()

def match(grid, src):
    for name in ('p', 'lat', 'lon'):
        a, b = grid[name], getattr(src, name)
        need(a.shape == b.shape and np.allclose(a, b, atol=1e-5, rtol=0),
             f'{src.path}: coordinate mismatch after normalization: {name}')

def make_grid(src):
    return {n: getattr(src, n).copy() for n in ('p', 'lat', 'lon')}

def new_nc(path, grid, overwrite, year=None):
    from netCDF4 import Dataset, date2num
    need(overwrite or not path.exists(), f'{path} exists; use --overwrite to replace')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    nc = Dataset(tmp, 'w', format='NETCDF4')
    td = 'month' if year is None else 'time'
    nc.createDimension(td, 12)
    t = nc.createVariable(td, 'i4' if year is None else 'f8', (td,))
    if year is None:
        t[:] = np.arange(1, 13)
        t.long_name = 'calendar month'
    else:
        t.units, t.calendar = TU, 'standard'
        t[:] = date2num([datetime(year, m, 1) for m in range(1, 13)], TU)
    for name, values, units in [('lev', grid['p'], 'Pa'), ('lat', grid['lat'], 'degrees_north'),
                                ('lon', grid['lon'], 'degrees_east')]:
        nc.createDimension(name, len(values))
        v = nc.createVariable(name, 'f8', (name,))
        v.units, v[:] = units, values
    nc['lev'].positive = 'down'
    nc['lev'].standard_name = 'air_pressure'
    nc.history = datetime.now(timezone.utc).isoformat() + ': cal_eddy_forcing.py'
    nc.Rd, nc.cp, nc.g, nc.earth_radius_m, nc.earth_rotation_s_1 = RD, CP, G, RADIUS, OMEGA
    nc.pressure_derivative = ('actual Pa spacing; background theta: second order except first-order '
                              'secant at largest pressure; outer heat-forcing derivative: second order')
    nc.background_derivative_scheme = BG_SCHEME
    return nc, tmp

def addvar(nc, name, dims, units, description):
    v = nc.createVariable(name, 'f8', dims, zlib=True, complevel=3, fill_value=9.969209968386869e36)
    v.units, v.long_name = units, description
    return v

def bgpath(args):
    return args.background_file or args.out / f'background.{args.clim_start}-{args.clim_end}.nc'

def background(args):
    grid, total = None, None
    for year in range(args.clim_start, args.clim_end + 1):
        src = Monthly(annual(args.basic, 'T_month', year), 'T_month', year, args.assume_pressure_unit)
        try:
            if grid is None:
                grid = make_grid(src)
                total = np.zeros((12, len(src.p), len(src.lat)))
            match(grid, src)
            for m in range(12):
                a = src.month(m)
                need(a.min() > 100 and a.max() < 400, f'{src.path}: temperature outside 100-400 K')
                total[m] += a.mean(axis=2)  # equal-longitude zonal average, NOT a latitude average
            print(f'BACKGROUND {year}', flush=True)
        finally:
            src.close()
    tb = total / (args.clim_end - args.clim_start + 1)  # equal-year calendar-month climatology
    theta, theta_p, sigma = stability(tb, grid['p'])
    valid = (theta_p < 0) & (sigma > 0) & np.isfinite(sigma)
    # Do not silently regularize invalid layers; record them for inspection.
    print(f'BACKGROUND sigma min/max={sigma.min():.6g}/{sigma.max():.6g}; invalid={np.count_nonzero(~valid)}', flush=True)
    path = bgpath(args)
    nc, tmp = new_nc(path, grid, args.overwrite)
    try:
        nc.background_definition = 'calendar-month climatological zonal mean temperature; equal weight per year; no latitude averaging or smoothing'
        nc.climatology_start_year, nc.climatology_end_year = args.clim_start, args.clim_end
        nc.source_directory = str(args.basic / 'T_month')
        for name, value, unit, desc in [
            ('T_background', tb, 'K', 'Climatological zonal mean temperature'),
            ('theta_background', theta, 'K', 'Background potential temperature'),
            ('dtheta_dp', theta_p, 'K Pa-1', 'Pressure derivative of background potential temperature'),
            ('sigma', sigma, 'm2 s-2 Pa-2', 'Static stability: -(Rd*T/p)/theta * dtheta/dp')]:
            addvar(nc, name, ('month', 'lev', 'lat'), unit, desc)[:] = value
        v = nc.createVariable('stable', 'i1', ('month', 'lev', 'lat'))
        v.long_name = '1 where theta_p<0 and sigma>0; otherwise 0'
        v[:] = valid.astype('i1')
    finally:
        nc.close()
    tmp.replace(path)
    print(f'SAVED {path}', flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--basic',type=Path,default=BASIC)
    ap.add_argument('--background-file',type=Path,default=BACKGROUND)
    ap.add_argument('--out',type=Path,default=ROOT/'step2')
    ap.add_argument('--clim-start',type=int,default=START_YEAR)
    ap.add_argument('--clim-end',type=int,default=END_YEAR)
    ap.add_argument('--assume-pressure-unit',choices=['Pa','hPa'])
    ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args();need(args.clim_start<=args.clim_end,'年份范围错误');background(args)
