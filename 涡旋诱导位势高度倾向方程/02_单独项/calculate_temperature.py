from pathlib import Path
import sys
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))
import bootstrap
"""原始6小时温度月平均：输出年度12个月，保留全部原始压力层。"""
import argparse
from contextlib import ExitStack
import numpy as np
from config import RAW_ROOT,BASIC,START_YEAR,END_YEAR
from audit_readers import *
def run_year(year, args, reference):
    names = (
        ["uzeta", "vzeta"] if args.task == "flux"
        else ["T_month"] if args.task == "temperature"
        else ["uzeta", "vzeta", "T_month"]
    )
    paths = {
        name: args.out / name /
        f"{name}.monthly.global.1deg.{year}.nc"
        for name in names
    }

    if not args.overwrite:
        for path in paths.values():
            require(not path.exists(),
                    f"{path} 已存在；确需重算覆盖时加 --overwrite")

    outputs = {}
    temporaries = {}

    try:
        for month in range(1, 13):
            print(f"PROCESS {year}-{month:02d}", flush=True)
            with ExitStack() as stack:
                inputs = {}

                def open_input(key, path, candidates, daily):
                    obj = InputFile(
                        path, candidates, year, month, daily,
                        args.assume_pressure_unit
                    )
                    stack.callback(obj.close)
                    inputs[key] = obj
                    return obj

                if args.task in ("flux", "all"):
                    u = open_input(
                        "u",
                        BANDPASS / "uwnd" /
                        f"uwnd.daymean.global.1deg.{year}.{month:02d}.BPF_L_2.5_8d.nc",
                        ("uwnd", "u"), True,
                    )
                    v = open_input(
                        "v",
                        BANDPASS / "vwnd" /
                        f"vwnd.daymean.global.1deg.{year}.{month:02d}.BPF_L_2.5_8d.nc",
                        ("vwnd", "v"), True,
                    )
                    require(u.dates == v.dates,
                            "u/v 的逐日时间标签不一致")

                if args.task in ("temperature", "all"):
                    t = open_input(
                        "t",
                        RAW_AIR /
                        f"air.6h.global.1deg.{year}.{month:02d}.nc",
                        ("t", "air"), False,
                    )

                first = next(iter(inputs.values()))
                if reference is None:
                    reference = grid_copy(first)
                for obj in inputs.values():
                    check_grid(reference, obj)

                if not outputs:
                    for name in names:
                        nc, tmp = create_output(
                            paths[name], name, year, reference, args
                        )
                        outputs[name] = nc
                        temporaries[name] = tmp

                for k, pressure in enumerate(reference.p):
                    fields = {}
                    counts = {}

                    if args.task in ("flux", "all"):
                        uz, vz = fluxes(u.level(k), v.level(k))
                        fields.update(uzeta=uz, vzeta=vz)
                        counts.update(uzeta=u.nt, vzeta=u.nt)

                    if args.task in ("temperature", "all"):
                        fields["T_month"] = temperature_mean(t.level(k))
                        counts["T_month"] = t.nt

                    for name, field in fields.items():
                        require(np.all(np.isfinite(field)),
                                f"{name}: 月平均结果含非有限值")
                        outputs[name][name][month - 1, k] = field
                        outputs[name]["n_samples"][month - 1, k] = counts[name]

                        if any(np.isclose(pressure, x)
                               for x in (250, 500, 850)):
                            report(
                                name, field, year, month,
                                pressure, reference
                            )

                for nc in outputs.values():
                    nc.sync()

        # 年度全部成功后才发布正式文件。
        for nc in outputs.values():
            nc.close()
        outputs.clear()
        for name in names:
            temporaries[name].replace(paths[name])
            print(f"SAVED {paths[name]}", flush=True)

    finally:
        for nc in outputs.values():
            nc.close()

    return reference
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--start-year',type=int,default=START_YEAR)
    ap.add_argument('--end-year',type=int,default=END_YEAR)
    ap.add_argument('--out',type=Path,default=BASIC)
    ap.add_argument('--assume-pressure-unit',choices=['Pa','hPa'])
    ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args(); args.task='temperature'
    require(args.start_year<=args.end_year,'年份范围错误')
    reference=None
    for year in range(args.start_year,args.end_year+1):reference=run_year(year,args,reference)
