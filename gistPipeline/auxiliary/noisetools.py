"""
Noise handling shared by the KIN and SFH modules.

The error spectra are now used as pPXF weights (they used to be constant), so
three things matter that did not before:

 * pPXF requires noise > 0 and finite. A zero or NaN made it raise, and the
   bare `except` in KIN turned that into a silent all-NaN bin.
 * Pixels with no data (the dichroic gap, long masked runs) are given an
   "infinite" variance by the reader. They should not just be down-weighted
   but removed from `goodpixels`, so that they do not count in chi2/DOF, in
   the formal-error correction sqrt(chi2), or in the MC noise level.
 * Exceptions inside worker processes must be reported, not swallowed.
"""
import logging
import os
import sys
import traceback

import numpy as np
from astropy.io import fits

# A pixel whose noise exceeds this many times the bin's median noise carries
# no data. The reader uses 1e12 x the median variance (1e6 in noise) for those.
NOISE_BAD_FACTOR = 1e3
# Value given to non-finite / non-positive noise, relative to the median
BIG_NOISE_FACTOR = 1e6
# A bin must keep at least this fraction of its goodpixels
MIN_GOOD_FRAC = 0.5


def sanitize(noise):
    """
    Replace non-finite or non-positive noise by BIG_NOISE_FACTOR x the median
    good noise of that spectrum. `noise` is (npix,) or (npix, nbins).
    """
    noise = np.array(noise, dtype=float)
    ok = np.isfinite(noise) & (noise > 0)
    with np.errstate(all='ignore'):
        med = np.nanmedian(np.where(ok, noise, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 1.0)
    nbad = np.count_nonzero(~ok)
    if nbad:
        logging.warning("%d noise values were zero, negative or non-finite; they are set to %.0e x "
                        "the median noise of their spectrum and excluded from the fits" % (nbad, BIG_NOISE_FACTOR))
    return np.where(ok, noise, BIG_NOISE_FACTOR * med)


def usable_pixels(goodPixels, noise):
    """ goodPixels minus the pixels that carry no data in this spectrum. """
    g = np.asarray(goodPixels)
    ref = np.median(noise[g])
    keep = g[noise[g] < NOISE_BAD_FACTOR * ref]
    if keep.size < MIN_GOOD_FRAC * g.size:
        raise ValueError("only %d of %d goodpixels carry data in this spectrum" % (keep.size, g.size))
    return keep


def log_failure(module, i):
    """ Report the exception being handled for bin i (works inside worker processes). """
    msg = "%s: the fit of bin %d failed:\n%s" % (module, i, traceback.format_exc())
    logging.error(msg)
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def read_bin_espec(config, logLam):
    """
    ESPEC (standard deviation) of the binned spectra in _BinSpectra.fits, on the
    LOGLAM pixels `logLam`. The emission-subtracted _gas-cleaned_BIN.fits has
    no ESPEC column and covers only the GAS wavelength range, so the errors
    always come from _BinSpectra.fits and are matched on LOGLAM.
    """
    fn = os.path.join(config['GENERAL']['OUTPUT'], config['GENERAL']['RUN_ID']) + '_BinSpectra.fits'
    with fits.open(fn) as hdu:
        espec = np.array(hdu[1].data.ESPEC)
        ll = np.array(hdu[2].data.LOGLAM)
    i0 = int(np.argmin(np.abs(ll - logLam[0])))
    i1 = i0 + len(logLam)
    if i1 > len(ll) or not np.allclose(ll[i0:i1], logLam, rtol=0, atol=1e-9):
        raise ValueError("the LOGLAM grid of the spectra does not match the one of " + fn)
    return espec[:, i0:i1]


def as_bool(value, default=False):
    """ Config switches may arrive as bool or as string. """
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in ('true', 't', 'yes', 'y', '1')
    return bool(value)


def covariance_factor(n, alpha=0., c=1., nmax=None, table=None):
    """
    Noise of a sum of n spaxels / the noise from the summed variance cube:
        c (1 + alpha log10 n)          n <= nmax
        c (1 + alpha log10 nmax)       n >  nmax  (the correlation length is reached)
    (O'Sullivan & Chen 2020; de los Reyes et al. 2023; Zhuang et al. 2024, eqs. 2-3).
    c: the error of ONE spaxel relative to the variance cube (resampling makes the
    propagated per-pixel variance inexact); alpha: SPATIAL_BINNING COVARIANCE.
    table = (N, ratio) measured curve (SPATIAL_BINNING COV_FILE, from kcwi_covar):
    interpolated in log N instead (constant beyond its ends); the other parameters
    are then ignored. Accepts scalars or arrays.
    """
    n = np.maximum(np.asarray(n, float), 1.)
    if table is not None:
        tn, tr = table
        out = np.interp(np.log10(n), np.log10(tn), tr)
        return out if out.ndim else float(out)
    if nmax is not None and np.isfinite(nmax) and nmax > 0:
        n = np.minimum(n, float(nmax))
    out = float(c) * (1. + float(alpha) * np.log10(n))
    return out if out.ndim else float(out)


def read_cov_table(fn):
    """(N, ratio) from a kcwi_covar curve file (columns N, ratio [, error])."""
    t = np.atleast_2d(np.genfromtxt(fn, comments='#'))
    o = np.argsort(t[:, 0])
    if t.shape[0] < 2 or np.any(t[:, 1] <= 0):
        raise ValueError('%s: not a covariance curve (N, ratio > 0 per line)' % fn)
    return t[o, 0], t[o, 1]


def covariance_options(sb, config_dir=None):
    """
    The covariance model of the SPATIAL_BINNING section: (alpha, c, nmax, table).
    COV_FILE (a kcwi_covar curve, in config_dir) replaces COVARIANCE, COV_C, COV_NMAX.
    """
    def num(k, default):
        v = sb.get(k, None)
        if v is None or v is False or str(v).strip().lower() in ('', 'none', 'false'):
            return default
        return float(v)
    table = None
    f = sb.get('COV_FILE', None)
    if f is not None and str(f).strip().lower() == 'auto':
        raise ValueError("SPATIAL_BINNING COV_FILE = auto is resolved by gist_prep (from its cov_dir): rerun "
                         "gist_prep, or give the curve file")
    if f is not None and f is not False and str(f).strip().lower() not in ('', 'none', 'false'):
        fn = f if (os.path.isabs(str(f)) or config_dir is None) else os.path.join(config_dir, str(f))
        table = read_cov_table(fn)
        logging.info("covariance: measured curve %s (N %g-%g, ratio %.3f-%.3f); COVARIANCE/COV_C/COV_NMAX "
                     "ignored" % (fn, table[0][0], table[0][-1], table[1].min(), table[1].max()))
    return num('COVARIANCE', 0.), num('COV_C', 1.), num('COV_NMAX', None), table
