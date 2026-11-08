#!/usr/bin/env python
"""
gist_fitresid.py -- where, and how, does a GIST fit miss the data?

For a finished run, compares every binned spectrum with its best fit (SFH by
default, or KIN / GAS) as a function of wavelength, split by surface brightness
of the bin (faint / mid / bright thirds), in two forms:

  ratio  data/fit - 1          the same in all thirds -> multiplicative
                               (flux calibration, throughput)
  excess (data - fit)/Nspax    the same in all thirds -> additive per spaxel
                               (sky subtraction, scattered light)

    python gist_fitresid.py results/<RUN>                 # SFH fit, default windows
    python gist_fitresid.py results/<RUN> --fit kin --step 50 --plot resid.png

The printed windows run every --step A (rest) over the fit range; the plot shows
the full curves. Needs _table.fits, _mask.fits, _BinSpectra.fits and the
<fit>-bestfit file.
"""
import argparse
import os
import sys
import warnings

import numpy as np
from astropy.io import fits


def main():
    warnings.simplefilter('ignore', RuntimeWarning)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('rundir')
    ap.add_argument('--fit', default='sfh', help='which best fit: sfh, kin or gas (default sfh)')
    ap.add_argument('--step', type=float, default=100., help='window width for the printed table [A rest]')
    ap.add_argument('--plot', help='write a PNG of the curves')
    a = ap.parse_args()
    d = a.rundir.rstrip('/'); run = os.path.basename(d); p = os.path.join(d, run)
    fn = {'sfh': '_sfh-bestfit.fits', 'kin': '_kin-bestfit.fits', 'gas': '_gas-bestfit_BIN.fits'}[a.fit.lower()]
    if not os.path.isfile(p + fn):
        sys.exit('no %s%s' % (run, fn))
    h = fits.open(p + '_BinSpectra.fits')
    S = np.asarray(h[1].data.SPEC); E = np.asarray(h[1].data.ESPEC)
    lam = np.exp(np.asarray(h['LOGLAM'].data.LOGLAM))
    b = fits.open(p + fn)
    BF = np.asarray(b['BESTFIT'].data.BESTFIT); lf = np.exp(np.asarray(b['LOGLAM'].data.LOGLAM))
    good = np.zeros(lf.size, bool)
    if 'GOODPIX' in b:
        gp = np.asarray(b['GOODPIX'].data.GOODPIX); good[gp[gp < lf.size]] = True
    else:
        good[:] = True
    i0 = int(np.argmin(np.abs(lam - lf[0]))); n = lf.size
    S = S[:, i0:i0 + n]; E = E[:, i0:i0 + n]
    t = fits.open(p + '_table.fits')[1].data
    use = np.asarray(t.BIN_ID) >= 0
    if os.path.isfile(p + '_mask.fits'):
        use &= np.asarray(fits.open(p + '_mask.fits')[1].data.MASK) == 0
    nb = S.shape[0]
    ns = np.bincount(np.asarray(t.BIN_ID)[use], minlength=nb).astype(float)
    ns[ns == 0] = np.nan
    nodata = E > 100 * np.nanmedian(E, axis=1, keepdims=True)
    with np.errstate(all='ignore'):
        ratio = np.where(nodata, np.nan, S / BF - 1)
        excess = np.where(nodata, np.nan, (S - BF) / ns[:, None])
    sb = np.nanmedian(BF, axis=1) / ns
    o = np.argsort(sb)
    groups = [('faint', o[:nb // 3]), ('mid', o[nb // 3:2 * nb // 3]), ('bright', o[2 * nb // 3:])]
    print('%s, %s fit: %d bins; median continuum per spaxel %s' % (
        run, a.fit.upper(), nb, ', '.join('%s %.3g' % (g, np.nanmedian(sb[i])) for g, i in groups)))
    print('%-13s %-29s %s' % ('window [A]', 'data/fit - 1 (faint mid bright)', '(data - fit)/Nspax (faint mid bright)'))
    edges = np.arange(np.floor(lf[0] / a.step) * a.step, lf[-1] + a.step, a.step)
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (lf >= lo) & (lf < hi) & good
        if k.sum() < 5:
            continue
        r = [np.nanmedian(np.nanmedian(ratio[i][:, k], axis=1)) for _, i in groups]
        x = [np.nanmedian(np.nanmedian(excess[i][:, k], axis=1)) for _, i in groups]
        flag = ''
        if max(abs(v) for v in r) > 0.05:
            spread_r = np.ptp(r) / max(abs(np.mean(r)), 1e-9)
            spread_x = np.ptp(x) / max(abs(np.mean(x)), 1e-9)
            flag = '  <- %s' % ('multiplicative' if spread_r < spread_x else 'additive')
        print('%5.0f-%-5.0f   %+.3f %+.3f %+.3f           %+.3g %+.3g %+.3g%s' % ((lo, hi) + tuple(r) + tuple(x) + (flag,)))
    print('"<-": more than 5% off; labelled by which form is more nearly the same in the three thirds.')
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
        for (g, i), c in zip(groups, ['C0', 'C1', 'C3']):
            ax[0].plot(lf, np.where(good, np.nanmedian(ratio[i], axis=0), np.nan), color=c, lw=0.6, label=g)
            ax[1].plot(lf, np.where(good, np.nanmedian(excess[i], axis=0), np.nan), color=c, lw=0.6, label=g)
        for x in ax:
            x.axhline(0, color='k', lw=0.5); x.legend(fontsize=8)
        ax[0].set_ylabel('data/fit - 1'); ax[1].set_ylabel('(data - fit) / Nspax')
        ax[0].set_ylim(-0.4, 0.4)
        lim = np.nanpercentile(np.abs(np.nanmedian(excess, axis=0)[good]), 99) * 1.5
        ax[1].set_ylim(-lim, lim)
        ax[1].set_xlabel(u'rest-frame wavelength [Å]')
        ax[0].set_title('%s: %s fit residuals by surface brightness' % (run, a.fit.upper()))
        fig.tight_layout(); fig.savefig(a.plot, dpi=120)
        print('wrote', a.plot)


if __name__ == '__main__':
    main()
