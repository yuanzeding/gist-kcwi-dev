"""Helpers shared by the gistPlot_* scripts."""
import os

import numpy as np
from astropy.io import fits


def _inside_idx(table_hdu, prefix):
    """
    Spaxels of the binned region: BIN_ID >= 0 and, when <prefix>_mask.fits exists,
    MASK == 0. GIST gives a masked spaxel BIN_ID = -(nearest bin), so one whose
    nearest bin is 0 gets 0 and would otherwise be plotted as part of bin 0.
    """
    b = np.asarray(table_hdu[1].data.BIN_ID)
    keep = b >= 0
    fn = prefix + '_mask.fits'
    if os.path.isfile(fn):
        m = np.asarray(fits.open(fn)[1].data.MASK)
        if m.size == b.size:
            keep &= (m == 0)
    return np.where(keep)[0]


def _log_flux(flux):
    """log10 of the flux for the isophotes; non-positive values set to the faintest positive one."""
    f = np.asarray(flux, float)
    good = np.isfinite(f) & (f > 0)
    floor = f[good].min() if good.any() else 1.
    return np.log10(np.where(good, f, floor))


def _flux_levels(flux, offset, step=0.2):
    """Isophote levels from the positive, finite fluxes only (empty if there are none)."""
    f = np.asarray(flux, float)
    lf = np.log10(f[np.isfinite(f) & (f > 0)])
    if lf.size == 0:
        return np.array([])
    return np.arange(lf.min() + offset, lf.max(), step)
