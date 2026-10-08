"""逐文件自动解包、归一坐标；压力内部hPa，纬度北到南，经度0—359。"""
import calendar
from pathlib import Path
from datetime import datetime
import numpy as np
from netCDF4 import Dataset, num2date, date2num
from config import BANDPASS, RAW_ROOT, BASIC
RAW_AIR=RAW_ROOT/'air'
DEFAULT_OUT=BASIC
RADIUS=6371200.0
TIME_UNITS='hours since 1900-01-01 00:00:00'
ALIASES={'time':('time','valid_time'),'lev':('lev','level','pressure_level','p'),
         'lat':('lat','latitude'),'lon':('lon','longitude')}
def require(condition, message):
    if not condition:
        raise ValueError(message)

def unpack_values(var, key=slice(None)):
    # netCDF4 自动按当前文件属性解包，并识别声明的缺测值。
    # 不再手动乘 scale_factor 或加 add_offset。
    a = np.ma.asarray(var[key], dtype=np.float64)
    a = np.asarray(a.filled(np.nan))

    # 防止未正确声明的 NCL/NetCDF 填充值参与计算。
    require(
        not np.any(np.isfinite(a) & (np.abs(a) > 1.0e30)),
        f"{var.name}: 存在未被缺测属性识别的巨大数值"
    )
    return a

class InputFile:
    def __init__(self, path, candidates, year, month,
                 daily, assume_pressure_unit, daily_temperature=False, raw_wind=False):
        self.path = Path(path)
        self.nc = Dataset(self.path)
        self.nc.set_auto_maskandscale(True)

        try:
            names = [n for n in candidates if n in self.nc.variables]
            require(len(names) == 1,
                    f"{path}: 无法唯一识别变量 {candidates}")
            self.var = self.nc[names[0]]
            require(self.var.ndim == 4,
                    f"{path}: 需要四维数据，实际 {self.var.dimensions}")

            self.dims = {}
            self.coords = {}

            for kind, aliases in ALIASES.items():
                dims = [d for d in self.var.dimensions if d in aliases]
                require(len(dims) == 1,
                        f"{path}: 无法识别 {kind} 维度")
                dim = dims[0]
                self.dims[kind] = dim

                # 同时兼容 dimension=time、coordinate=valid_time。
                possible = list(dict.fromkeys((dim,) + aliases))
                coords = [
                    self.nc.variables[n]
                    for n in possible
                    if n in self.nc.variables
                    and self.nc.variables[n].dimensions == (dim,)
                ]
                require(len(coords) >= 1,
                        f"{path}: {dim} 没有对应一维坐标")
                self.coords[kind] = coords[0]

            pvar = self.coords["lev"]
            p = unpack_values(pvar)
            units = str(getattr(pvar, "units", "")).strip().lower()

            if not units:
                require(
                    assume_pressure_unit is not None,
                    f"{path}: 压力单位缺失；确认后使用 "
                    "--assume-pressure-unit hPa"
                )
                units = assume_pressure_unit.lower()
                print(f"NOTE {path}: 压力单位缺失，明确假定 {units}",
                      flush=True)

            if units in ("pa", "pascal", "pascals"):
                p = p / 100.0
            else:
                require(
                    units in ("hpa", "mb", "mbar",
                              "millibar", "millibars"),
                    f"{path}: 不支持压力单位 {units}"
                )

            lat = unpack_values(self.coords["lat"])
            lon = unpack_values(self.coords["lon"])

            for name, arr in (("pressure", p), ("lat", lat), ("lon", lon)):
                require(np.all(np.isfinite(arr)),
                        f"{path}: {name} 坐标含缺测")

            # 排序索引同时应用到数据，不能只修改坐标标签。
            self.ip = np.argsort(p)
            self.iy = np.argsort(lat)[::-1]
            lon = np.mod(lon, 360.0)
            self.ix = np.argsort(lon)

            self.p = p[self.ip]
            self.lat = lat[self.iy]
            self.lon = lon[self.ix]

            require(
                np.all(np.diff(self.p) > 0)
                and np.all((self.p > 0) & (self.p <= 1100)),
                f"{path}: 压力层重复或数值异常"
            )

            # 按你的全球 1° 经纬网格检查，拒绝局地或缺行数据。
            require(
                len(self.lat) == 181
                and np.allclose(self.lat, np.arange(90, -91, -1),
                                atol=1e-5, rtol=0),
                f"{path}: 预期纬度为完整的 90 ... -90，间隔1°"
            )
            require(
                len(self.lon) == 360
                and np.allclose(self.lon, np.arange(360),
                                atol=1e-5, rtol=0),
                f"{path}: 预期经度0 ... 359，不含重复360°端点"
            )

            tvar = self.coords["time"]
            require(hasattr(tvar, "units"),
                    f"{path}: 时间单位缺失")
            dates = num2date(
                unpack_values(tvar),
                tvar.units,
                calendar=getattr(tvar, "calendar", "standard"),
            )
            self.dates = [
                (d.year, d.month, d.day, d.hour, d.minute, d.second)
                for d in dates
            ]
            ndays = calendar.monthrange(year, month)[1]
            hours = (0,) if daily else (0, 6, 12, 18)

            expected = [
                (year, month, day, hour, 0, 0)
                for day in range(1, ndays + 1)
                for hour in hours
            ]

            if daily:
                # 日资料允许用每天中午等固定时刻作为日期标签。
                require(len(self.dates) == ndays,
                        f"{path}: 日记录数不是 {ndays}")
                require(
                    [d[:3] for d in self.dates]
                    == [d[:3] for d in expected],
                    f"{path}: 日日期缺失、重复或顺序错误"
                )
                require(len(set(d[3:] for d in self.dates)) == 1,
                        f"{path}: 日时间标签不一致")
            else:
                require(self.dates == expected,
                        f"{path}: 原始时间不是完整00/06/12/18 UTC")

            self.nt = len(self.dates)
            self.daily = daily

            # 不改变变量单位；明确检查所期待的单位。
            vu = str(getattr(self.var, "units", "")).lower()
            vu = vu.replace(" ", "").replace("**", "^")
            if raw_wind or (daily and not daily_temperature):
                allowed = ("m/s", "ms-1", "ms^-1", "m.s-1", "m.s^-1")
            else:
                allowed = ("k", "kelvin", "kelvins")
            require(
                vu in allowed,
                f"{path}: 变量单位 {getattr(self.var, 'units', None)!r} "
                f"不是预期的 {'m/s' if daily else 'K'}；请核对元数据"
            )

        except Exception:
            self.nc.close()
            raise

    def close(self):
        self.nc.close()

    def level(self, k):
        """返回一个压力层的 (time, lat, lon) 数组。"""
        dim_order = list(self.var.dimensions)
        key = [slice(None)] * 4
        paxis = dim_order.index(self.dims["lev"])
        key[paxis] = int(self.ip[k])

        a = unpack_values(self.var, tuple(key))
        remaining = [d for j, d in enumerate(dim_order) if j != paxis]
        axes = [
            remaining.index(self.dims[name])
            for name in ("time", "lat", "lon")
        ]
        a = np.transpose(a, axes)
        return a[:, self.iy, :][:, :, self.ix]

def check_grid(ref, other):
    for name in ("p", "lat", "lon"):
        a = getattr(ref, name)
        b = getattr(other, name)
        require(
            a.shape == b.shape
            and np.allclose(a, b, atol=1e-5, rtol=0),
            f"{other.path}: 规范化后 {name} 与基准仍不一致"
        )

def grid_copy(src):
    # 不持有已关闭的输入文件。
    class Grid:
        pass
    out = Grid()
    for name in ("p", "lat", "lon"):
        setattr(out, name, getattr(src, name).copy())
    return out

def create_output(path, name, year, grid, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    nc = Dataset(temporary, "w", format="NETCDF4")

    nc.createDimension("time", 12)
    nc.createDimension("lev", len(grid.p))
    nc.createDimension("lat", len(grid.lat))
    nc.createDimension("lon", len(grid.lon))

    time = nc.createVariable("time", "f8", ("time",))
    time.units = TIME_UNITS
    time.calendar = "standard"
    time[:] = date2num(
        [datetime(year, m, 1) for m in range(1, 13)],
        TIME_UNITS
    )

    for key, values, units in (
        ("lev", grid.p, "hPa"),
        ("lat", grid.lat, "degrees_north"),
        ("lon", grid.lon, "degrees_east"),
    ):
        v = nc.createVariable(key, "f8", (key,))
        v.units = units
        v[:] = values

    nc["lev"].positive = "down"
    nc["lev"].standard_name = "air_pressure"

    v = nc.createVariable(
        name, "f4", ("time", "lev", "lat", "lon"),
        zlib=True, complevel=3, fill_value=np.float32(9.96921e36),
        chunksizes=(1, 1, len(grid.lat), len(grid.lon))
    )
    v.units = "K" if name == "T_month" else "m s-2"
    v.long_name = {
        "uzeta": "Monthly mean zonal transient eddy relative vorticity flux",
        "vzeta": "Monthly mean meridional transient eddy relative vorticity flux",
        "T_month": "Monthly mean unfiltered air temperature",
    }[name]
    v.cell_methods = "time: mean"

    count = nc.createVariable("n_samples", "i4", ("time", "lev"))
    count.long_name = "Number of samples used in each monthly level mean"
    count.units = "1"

    nc.source_bandpass_directory = str(BANDPASS)
    nc.source_raw_temperature_directory = str(RAW_AIR)
    nc.history = f"{datetime.now().isoformat()}: cal_eddy_basic.py"
    nc.earth_radius_m = RADIUS
    nc.temperature_method = "Complete equally spaced raw 6-hourly monthly mean"
    nc.missing_policy = (
        "Fail on any missing input; complete monthly samples required"
    )
    nc.pressure_unit_assumption_if_absent = (
        args.assume_pressure_unit or "none"
    )
    return nc, temporary

def temperature_mean(t):
    require(np.all(np.isfinite(t)),
            "原始温度含缺测/NaN/Inf：停止，不静默使用不完整月平均")
    require(np.min(t) > 100 and np.max(t) < 400,
            "温度超出100—400 K检查范围，请检查单位和解包")
    # 时间已验证为完整、等间隔的六小时资料。
    return np.mean(t, axis=0, dtype=np.float64)

def report(name, field, year, month, pressure, grid):
    iy = (grid.lat >= 20) & (grid.lat <= 70)
    ix = (grid.lon >= 120) & (grid.lon <= 260)
    region = field[np.ix_(iy, ix)]
    rms = np.sqrt(np.mean(region.astype(np.float64) ** 2))
    print(
        f"{year}-{month:02d} {pressure:g}hPa {name}: "
        f"regional mean={region.mean():.6g}, RMS={rms:.6g}, "
        f"absMax={np.abs(region).max():.6g}",
        flush=True,
    )
