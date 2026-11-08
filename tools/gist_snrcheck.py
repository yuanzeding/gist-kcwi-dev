#!/usr/bin/env python
"""
gist_snrcheck.py -- is the S/N GIST bins on the real S/N of the binned spectra?

For every bin of a finished run (needs only _table.fits and _BinSpectra_linear.fits),
the noise is measured from the spectrum itself with DER_SNR (Stoehr et al. 2008:
pixel-to-pixel scatter, independent of the variance cube) and compared with the
propagated error, in several wavelength windows, as a function of spaxels per bin.

  ratio = DER_SNR noise / propagated error
    ~1 and flat in N  : errors right, COVARIANCE ~ 0
    grows ~log N      : correlated noise between spaxels -> COVARIANCE = alpha below
    grows ~sqrt N     : something coherent across spaxels (sky residuals) adds up
                        faster than noise -- binning stops helping

    python gist_snrcheck.py results/RUN                       # default windows
    python gist_snrcheck.py results/RUN --win 4600 5300 --win 8400 8600 --plot snr.png

Windows are rest frame [A] (the BinSpectra grid). DER_SNR also counts unmodelled
narrow features (sky-line residuals, emission lines, sharp absorption) as noise,
which is what a fit sees too; pick continuum windows for the cleanest number.
"""
import argparse
import numpy as np
from astropy.io import fits


def der_noise(f):
    f = f[np.isfinite(f)]
    return 1.482602 / np.sqrt(6.) * np.median(np.abs(2 * f[2:-2] - f[:-4] - f[4:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('rundir')
    ap.add_argument('--win', type=float, nargs=2, action='append', metavar=('LMIN', 'LMAX'))
    ap.add_argument('--plot')
    a = ap.parse_args()
    d = a.rundir.rstrip('/'); run = d.split('/')[-1]; p = d + '/' + run
    b = fits.open(p + '_BinSpectra_linear.fits')
    lam = np.asarray(b['LOGLAM'].data.field(0), float)
    if lam.max() < 20:
        lam = np.exp(lam)
    s, e = b[1].data.SPEC, b[1].data.ESPEC                       # ESPEC = sigma
    t = fits.open(p + '_table.fits')[1].data
    first = {}
    for j, bid in enumerate(np.abs(t.BIN_ID)):
        first.setdefault(int(bid), j)
    nb = len(s)
    N = np.array([t.NSPAX[first[i]] for i in range(nb)])
    snrbin = np.array([t.SNRBIN[first[i]] for i in range(nb)])
    wins = [tuple(w) for w in a.win] if a.win else [(4600, 5300), (6400, 6800), (7500, 8500), (8400, 8600)]
    # (--win gives lists, which cannot be dict keys)
    wins = [w for w in wins if ((lam > w[0]) & (lam < w[1])).sum() > 50]
    print('%s: %d bins, %d-%d spaxels, nominal SNRBIN median %.1f (COVARIANCE as in the run)' % (
        run, nb, N.min(), N.max(), np.median(snrbin)))
    R = {}; S = {}
    for w in wins:
        m = (lam > w[0]) & (lam < w[1])
        R[w] = np.array([der_noise(s[i, m]) / np.median(e[i, m]) for i in range(nb)])
        S[w] = np.array([np.median(s[i, m]) / der_noise(s[i, m]) for i in range(nb)])
    edges = np.unique(np.r_[1, np.round(np.logspace(0, np.log10(N.max() + 1), 7)).astype(int), N.max() + 1])
    print('\n%-18s' % 'spaxels/bin' + ''.join('%26s' % ('%g-%g A' % w) for w in wins))
    print('%-18s' % '' + ''.join('%26s' % 'ratio   true S/N' for w in wins))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (N >= lo) & (N < hi)
        if m.sum():
            print('%4d-%-4d (%3d)    ' % (lo, hi - 1, m.sum()) +
                  ''.join('%17.2f %8.1f' % (np.nanmedian(R[w][m]), np.nanmedian(S[w][m])) for w in wins))
    print('\nfit ratio = r1 (1 + alpha log10 N) per window (alpha -> SPATIAL_BINNING COVARIANCE):')
    for w in wins:
        ok = np.isfinite(R[w]) & (R[w] > 0)
        A = np.vstack([np.ones(ok.sum()), np.log10(N[ok])]).T
        c = np.linalg.lstsq(A, R[w][ok], rcond=None)[0]
        k = np.polyfit(np.log10(N[ok]), np.log10(R[w][ok]), 1)[0]
        print('  %g-%g A: r1 = %.2f, alpha = %.2f;  ratio ~ N^%.2f%s' % (
            w[0], w[1], c[0], c[1] / c[0] if c[0] > 0 else np.nan, k,
            '   <- coherent residuals (sky?), not just covariance' if k > 0.3 else ''))
    if a.plot:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        for w in wins:
            ax[0].loglog(N, R[w], '.', ms=4, label='%g-%g A' % w)
            ax[1].semilogx(N, S[w], '.', ms=4)
        ax[0].set_xlabel('spaxels per bin'); ax[0].set_ylabel('DER_SNR noise / propagated error'); ax[0].legend(fontsize=7)
        ax[1].axhline(np.median(snrbin), color='k', lw=0.7, ls='--'); ax[1].set_xlabel('spaxels per bin')
        ax[1].set_ylabel('true S/N per pixel (DER_SNR)'); fig.tight_layout(); fig.savefig(a.plot, dpi=110)
        print('wrote', a.plot)


if __name__ == '__main__':
    main()
