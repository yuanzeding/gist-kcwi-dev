#!/usr/bin/env python
"""
gist_replot.py -- redraw the GIST maps from finished results, without re-running
any fit. Calls the same plotting routines as the pipeline, non-interactively.

    python gist_replot.py results/LEDA66688bin35_v2              # every module with results
    python gist_replot.py results/LEDA66688bin35_v2 --modules KIN SFH

Maps go to <run dir>/maps/, as from the pipeline. Run it in the GIST
environment (it imports gistPipeline). Each module is tried on its own: one
that fails is reported with its traceback and the others still run.
"""
import argparse
import os
import sys
import traceback

import matplotlib
matplotlib.use('Agg')                       # no display on bifrost


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('rundir', help='the run directory, e.g. results/LEDA66688bin35_v2')
    ap.add_argument('--modules', nargs='+', choices=['KIN', 'GAS', 'SFH', 'LS'],
                    help='default: every module whose results exist')
    a = ap.parse_args()

    from gistPipeline.plotting import gistPlot_kin, gistPlot_lambdar, gistPlot_gas, gistPlot_sfh, gistPlot_ls

    d = a.rundir.rstrip('/')
    run = os.path.basename(d)
    have = lambda suffix: os.path.isfile(os.path.join(d, run + suffix))
    if not have('_table.fits'):
        sys.exit('%s: no %s_table.fits -- is this a GIST run directory?' % (d, run))

    ls_type = None
    if os.path.isfile(os.path.join(d, 'CONFIG')):
        try:
            import yaml
            ls_type = (yaml.safe_load(open(os.path.join(d, 'CONFIG'))) or {}).get('LS', {}).get('TYPE')
        except Exception:
            pass

    jobs = {
        'KIN': (have('_kin.fits'), [lambda: gistPlot_kin.plotMaps('KIN', d), lambda: gistPlot_lambdar.plotMaps(d)]),
        'GAS': (have('_gas_BIN.fits') or have('_gas_SPAXEL.fits'),
                ([lambda: gistPlot_gas.plotMaps(d, 'BIN', True)] if have('_gas_BIN.fits') else []) +
                ([lambda: gistPlot_gas.plotMaps(d, 'SPAXEL', True)] if have('_gas_SPAXEL.fits') else [])),
        'SFH': (have('_sfh.fits'), [lambda: gistPlot_sfh.plotMaps('SFH', d)]),
        'LS': (have('_ls_OrigRes.fits') or have('_ls_AdapRes.fits'),
               [lambda: gistPlot_ls.plotMaps(d, 'ORIGINAL'), lambda: gistPlot_ls.plotMaps(d, 'ADAPTED')] +
               ([lambda: gistPlot_sfh.plotMaps('LS', d)] if ls_type == 'SPP' else [])),
    }
    todo = a.modules or [m for m in ('KIN', 'GAS', 'SFH', 'LS') if jobs[m][0]]
    if not todo:
        sys.exit('no module results found in %s' % d)
    failed = []
    for m in todo:
        ok, calls = jobs[m]
        if not ok:
            print('%s: no results in %s, skipped' % (m, d))
            continue
        try:
            for c in calls:
                c()
            print('%s: maps written to %s' % (m, os.path.join(d, 'maps')))
        except Exception:
            failed.append(m)
            print('%s: FAILED\n%s' % (m, traceback.format_exc()))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
