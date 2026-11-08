#!/usr/bin/env python
"""
gist_regul.py -- choose SFH REGUL_ERR from the data.

The criterion (Press et al. 1992; Cappellari 2017, sect. 3.5; the pPXF population
examples): rescale the noise so that the UNregularised fit has chi2/DOF = 1, then
increase the regularisation until the total chi2 has risen by

    Delta chi2 = sqrt(2 N_good)            (N_good = pixels in the fit)

(x sqrt(ncorr) when the noise of neighbouring pixels is correlated: SFH MATCH_RES
smooths the data, see the SFH module)

above it. That is the smoothest star-formation history the data cannot tell apart
from the best fit. Delta chi2 grows monotonically with regul, so a bisection in
log10(regul) finds it in ~10 pPXF fits per bin.

This script runs that search on a few bins of a finished run (the spectra and, with
SFH FIXED = True, the KIN step must exist), with EXACTLY the set-up of the SFH module
(templates, gas lines, masks, MDEG, fixed kinematics -- it calls the module's own
code), and prints the value for each bin and a recommendation.

    python gist_regul.py configFiles/MasterConfig_v4                   # 5 bins over the S/N range
    python gist_regul.py configFiles/MasterConfig_v4 --bins 0,3,17 --ncpu 3
    python gist_regul.py configFiles/MasterConfig_v4 --nbins 8 --ncpu 8

Run it from the target directory (the one with configFiles/defaultDir), in the GIST
environment, like run_gist.py.

Reading the result
  * One number for the whole run: take the recommended (median) value as REGUL_ERR.
  * The value usually depends on S/N: higher S/N constrains the weights more, so a
    smaller regularisation (larger REGUL_ERR) is needed. If the values spread by more
    than ~0.5 dex over your bins, set REGUL_ERR = auto in [SFH]: the SFH module then
    runs this search in every bin (about 10x the fits) and writes the value it used in
    the REGUL_ERR column of _sfh.fits.
  * flag at_max: the data cannot distinguish even the smoothest solution tried from
    the best fit (low S/N): the SFH of that bin is essentially unconstrained.
    flag at_min: even the weakest regularisation tried is too strong (very high S/N).
"""
from __future__ import print_function

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from astropy.io import fits

_STATE = {}


def _one(i):
    d, cfg = _STATE['d'], _STATE['config']
    t0 = time.time()
    try:
        r, info = _STATE['sfh'].find_regul(
            d['templates'], d['galaxy'][i, :], d['noise'].T[:, i], d['velscale'], d['start'][i, :],
            d['goodPixels'], cfg['SFH']['MOM'], d['dv'], cfg['SFH']['MDEG'], d['fixed'],
            d['velscale_ratio'], d['ncomb'], d['gas'], ncorr=d['opts']['ncorr'])
        info['time'] = time.time() - t0
        return i, r, info
    except Exception as e:
        return i, np.nan, dict(flag='failed: %s' % e, target=np.nan, dchi2=np.nan, nfit=0, time=time.time() - t0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('masterconfig')
    ap.add_argument('--default-dir', default='configFiles/defaultDir')
    ap.add_argument('--row', type=int, default=0, help='MasterConfig run row (default 0)')
    ap.add_argument('--bins', help='comma list of BIN_IDs; default: --nbins bins spread over the S/N range')
    ap.add_argument('--nbins', type=int, default=5, help='number of bins chosen over the S/N range (default 5)')
    ap.add_argument('--ncpu', type=int, default=1, help='bins searched in parallel (default 1)')
    a = ap.parse_args()

    from gistPipeline.initialise import _initialise
    from gistPipeline.starFormationHistories import ppxf as sfh
    config = _initialise.readMasterConfig(a.masterconfig, a.row)
    if config is None:
        sys.exit('could not read %s' % a.masterconfig)
    config = _initialise.addPathsToConfig(config, argparse.Namespace(defaultDir=a.default_dir))
    out = os.path.join(config['GENERAL']['OUTPUT'], config['GENERAL']['RUN_ID'])
    for need in ['_BinSpectra.fits'] + (['_kin.fits'] if config['SFH']['FIXED'] == True else []):
        if not os.path.isfile(out + need):
            sys.exit('%s%s is missing: run GIST up to %s first' % (out, need, 'KIN' if 'kin' in need else 'the spectra'))

    print('preparing the SFH set-up of %s (templates, spectra, masks)...' % config['GENERAL']['RUN_ID'])
    d = sfh.prepare_sfh(config)
    print('LMIN-LMAX %g-%g A, MDEG %s, MOM %s, FIXED %s, gas lines: %s; %d bins'
          % (config['SFH']['LMIN'], config['SFH']['LMAX'], config['SFH']['MDEG'], config['SFH']['MOM'],
             config['SFH']['FIXED'], ', '.join(d['gas']['names']) if d['gas'] else 'none (masked)', d['nbins']))
    if d.get('match_res'):
        print('MATCH_RES: data degraded to the template LSF; correlated noise, ncorr = %.1f: the target is '
              'Delta chi2 = sqrt(2 N_good ncorr)' % d['opts']['ncorr'])

    # bins: spread over the S/N range (SNRBIN of the binning table)
    t = fits.open(out + '_table.fits')[1].data
    snr, nsp = np.full(d['nbins'], np.nan), np.zeros(d['nbins'], int)
    for bid, s, n in zip(t.BIN_ID, t.SNRBIN, t.NSPAX):
        if 0 <= bid < d['nbins']:
            snr[bid], nsp[bid] = s, n
    if a.bins:
        bins = [int(x) for x in a.bins.split(',')]
    else:
        ok = np.flatnonzero(np.isfinite(snr))
        order = ok[np.argsort(snr[ok])]
        idx = np.unique(np.round(np.linspace(0, order.size - 1, min(a.nbins, order.size))).astype(int))
        bins = list(order[idx])

    _STATE.update(d=d, config=config, sfh=sfh)
    sfh.printStatus.progressBar = lambda *x, **k: None
    print('searching REGUL_ERR in bins %s (each ~10 pPXF fits)...' % ', '.join(str(b) for b in bins))
    if a.ncpu > 1:
        with Pool(a.ncpu) as pool:
            res = pool.map(_one, bins)
    else:
        res = [_one(b) for b in bins]

    print('\n%6s %8s %7s %8s %12s %10s %14s %5s %7s  %s' % ('BIN', 'SNRBIN', 'NSPAX', 'N_good', 'REGUL_ERR',
          'regul', 'Dchi2/target', 'fits', 'time', 'flag'))
    good = []
    for i, r, info in res:
        ngood = d['goodPixels'].size
        print('%6d %8.1f %7d %8d %12.4g %10.4g %14.2f %5d %6.0fs  %s' % (i, snr[i], nsp[i], ngood, r,
              1. / r if r == r else np.nan, info['dchi2'] / info['target'] if info['target'] == info['target']
              else np.nan, info['nfit'], info['time'], info['flag']))
        if info['flag'] == 'ok':
            good.append(r)
    print()
    cur = config['SFH']['REGUL_ERR']
    if not good:
        print('no bin gave a value inside the search range; see the flags above. Current REGUL_ERR = %s' % cur)
        return
    lg = np.log10(good)
    rec = 10 ** np.median(lg)
    print('recommended REGUL_ERR = %.3g  (median of %d bins; range %.3g-%.3g, i.e. %.2f dex)'
          % (rec, len(good), min(good), max(good), lg.max() - lg.min()))
    print('current   REGUL_ERR = %s' % cur)
    if lg.max() - lg.min() > 0.5:
        print('the values spread by more than 0.5 dex over the S/N range: consider REGUL_ERR = auto '
              '(chosen per bin by the SFH module)')


if __name__ == '__main__':
    main()
