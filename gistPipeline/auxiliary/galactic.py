"""
Galactic (Milky-Way foreground) extinction for the GIST readers.

  ebv = sfd_ebv(ra, dec, mapdir)          # E(B-V) along a line of sight [mag]
  A   = ebv * f99(wave)                   # A(lambda) [mag], wave in Angstrom
  spec *= 10**(0.4 A);  variance *= 10**(0.8 A)

sfd_ebv reads the Schlegel, Finkbeiner & Davis (1998) maps (SFD_dust_4096_ngp.fits
/ _sgp.fits, Lambert zenithal equal-area projections of the two Galactic hemispheres;
the files the 'dustmaps' and 'sfdmap' packages use) directly -- no extra package --
interpolating bilinearly between pixels, and multiplies by 0.86, the recalibration of
Schlafly & Finkbeiner (2011, ApJ 737, 103), which they derived for the Fitzpatrick (1999)
R_V = 3.1 law -- so f99 is the matching curve (they also found it fits the reddening of
SDSS stars better than CCM89 / O'Donnell 1994; ccm89 and od94 are available for
comparison, READ_DATA EXT_LAW). Below |b| ~ 5 deg SFD is less reliable
(unresolved dust structure, temperature errors): treat such values with caution.
"""
import os

import numpy as np

SF11_SCALE = 0.86


def _poly(x, c):
    return sum(ci * x ** i for i, ci in enumerate(c))


def f99(wave, r_v=3.1):
    """
    A(lambda) / E(B-V) of Fitzpatrick (1999) for R_V = r_v: natural cubic spline through
    the optical/IR anchor points and the two UV points at 2700 and 2600 A, the FM90
    parametrisation shortward of 2700 A (as fm_unred.pro / the 'extinction' package).
    wave: Angstrom (array). Valid 1000 A - 3.5 micron.
    """
    from scipy.interpolate import CubicSpline
    x = 1e4 / np.atleast_1d(np.asarray(wave, float))          # inverse micron
    x0, gamma, c3, c4 = 4.596, 0.99, 3.23, 0.41
    c2 = -0.824 + 4.717 / r_v
    c1 = 2.030 - 3.007 * c2

    def uv(xx):
        y = c1 + c2 * xx + c3 * xx ** 2 / ((xx ** 2 - x0 ** 2) ** 2 + (xx * gamma) ** 2)
        far = xx > 5.9
        y = np.where(far, y + c4 * (0.5392 * (xx - 5.9) ** 2 + 0.05644 * (xx - 5.9) ** 3), y)
        return y + r_v

    xuv = 1e4 / np.array([2700., 2600.])
    xop = np.r_[0., 1e4 / np.array([26500., 12200., 6000., 5470., 4670., 4110.])]
    yop = np.r_[np.array([0., 0.26469, 0.82925]) * r_v / 3.1,
                _poly(r_v, [-4.22809e-01, 1.00270, 2.13572e-04]),
                _poly(r_v, [-5.13540e-02, 1.00216, -7.35778e-05]),
                _poly(r_v, [7.00127e-01, 1.00184, -3.32598e-05]),
                _poly(r_v, [1.19456, 1.01707, -5.46959e-03, 7.97809e-04, -4.45636e-05])]
    spl = CubicSpline(np.r_[xop, xuv], np.r_[yop, uv(xuv)], bc_type='natural')
    xcut = 1e4 / 2700.
    out = np.where(x < xcut, spl(np.minimum(x, xcut)), uv(np.maximum(x, xcut)))
    return out if np.ndim(wave) else float(out[0])


def _ccm_like(wave, r_v, ca, cb):
    """Cardelli, Clayton & Mathis (1989) form, A(lambda)/E(B-V) = R_V a(x) + b(x), with the
    optical polynomials ca, cb (CCM89 or O'Donnell 1994); 0.3-10 micron^-1."""
    x = 1e4 / np.atleast_1d(np.asarray(wave, float))
    a = np.zeros_like(x); b = np.zeros_like(x)
    ir = x < 1.1
    a[ir] = 0.574 * x[ir] ** 1.61; b[ir] = -0.527 * x[ir] ** 1.61
    op = (x >= 1.1) & (x < 3.3)
    y = x[op] - 1.82
    a[op] = np.polyval(ca[::-1], y); b[op] = np.polyval(cb[::-1], y)
    uv = x >= 3.3
    xu = x[uv]
    fa = np.where(xu >= 5.9, -0.04473 * (xu - 5.9) ** 2 - 0.009779 * (xu - 5.9) ** 3, 0.)
    fb = np.where(xu >= 5.9, 0.2130 * (xu - 5.9) ** 2 + 0.1207 * (xu - 5.9) ** 3, 0.)
    a[uv] = 1.752 - 0.316 * xu - 0.104 / ((xu - 4.67) ** 2 + 0.341) + fa
    b[uv] = -3.090 + 1.825 * xu + 1.206 / ((xu - 4.62) ** 2 + 0.263) + fb
    out = r_v * a + b
    return out if np.ndim(wave) else float(out[0])


def ccm89(wave, r_v=3.1):
    """A(lambda)/E(B-V), Cardelli, Clayton & Mathis (1989)."""
    return _ccm_like(wave, r_v, [1., 0.17699, -0.50447, -0.02427, 0.72085, 0.01979, -0.77530, 0.32999],
                     [0., 1.41338, 2.28305, 1.07233, -5.38434, -0.62251, 5.30260, -2.09002])


def odonnell94(wave, r_v=3.1):
    """A(lambda)/E(B-V), CCM89 with the optical update of O'Donnell (1994)."""
    return _ccm_like(wave, r_v, [1., 0.104, -0.609, 0.701, 1.137, -1.718, -0.827, 1.647, -0.505],
                     [0., 1.952, 2.908, -3.989, -7.985, 11.102, 5.491, -10.805, 3.347])


LAWS = {'f99': f99, 'ccm89': ccm89, 'od94': odonnell94}


def curve(law='f99'):
    """The A(lambda)/E(B-V) function of READ_DATA EXT_LAW (f99, ccm89, od94)."""
    key = 'f99' if law is None else str(law).strip().lower()
    if key in ('none', ''):
        key = 'f99'
    if key not in LAWS:
        raise ValueError('extinction law %r: use one of %s' % (law, ', '.join(LAWS)))
    return LAWS[key]


def default_mapdirs():
    """Folders searched for the SFD maps, in order."""
    here = os.path.dirname(os.path.abspath(__file__))
    out = []
    for env in ('SFD_DIR', 'SFD_MAPS', 'DUSTMAPS_DIR'):
        v = os.environ.get(env)
        if v:
            out += [v, os.path.join(v, 'sfd')]
    out += [os.path.expanduser('~/software/dustmaps/sfd'), os.path.expanduser('~/.dustmaps/sfd'),
            # gist-development/gistPipeline/auxiliary -> software/dustmaps/sfd
            os.path.normpath(os.path.join(here, '..', '..', '..', 'dustmaps', 'sfd'))]
    return out


def find_maps(mapdir=None):
    """The folder holding SFD_dust_4096_ngp.fits and _sgp.fits."""
    for d in ([mapdir] if mapdir else []) + ([] if mapdir else default_mapdirs()):
        if d and os.path.isfile(os.path.join(d, 'SFD_dust_4096_ngp.fits')) and \
                os.path.isfile(os.path.join(d, 'SFD_dust_4096_sgp.fits')):
            return d
    raise IOError('SFD maps (SFD_dust_4096_ngp.fits, _sgp.fits) not found in %s' % (
        mapdir or ', '.join(default_mapdirs())))


def sfd_ebv(ra, dec, mapdir=None, scale=SF11_SCALE, frame='icrs'):
    """
    E(B-V) [mag] at (ra, dec) [deg] from the SFD (1998) maps x scale (default 0.86,
    Schlafly & Finkbeiner 2011; scale = 1 for the original SFD values).
    Returns (ebv, info) with info: Galactic l, b [deg], raw SFD value, map folder.
    """
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    import astropy.units as u
    from scipy.ndimage import map_coordinates
    d = find_maps(mapdir)
    g = SkyCoord(ra=float(ra) * u.deg, dec=float(dec) * u.deg, frame=frame).galactic
    l, b = np.radians(g.l.deg), np.radians(g.b.deg)
    fn = os.path.join(d, 'SFD_dust_4096_%s.fits' % ('ngp' if b >= 0 else 'sgp'))
    with fits.open(fn) as h:
        hdr = h[0].header
        n = float(hdr.get('LAM_NSGP', 1 if b >= 0 else -1))
        scal = float(hdr['LAM_SCAL'])
        # Lambert zenithal equal-area: 0-based pixel coordinates (SFD98 appendix C)
        x = hdr['CRPIX1'] - 1. + scal * np.cos(l) * np.sqrt(1. - n * np.sin(b))
        y = hdr['CRPIX2'] - 1. - n * scal * np.sin(l) * np.sqrt(1. - n * np.sin(b))
        raw = float(map_coordinates(h[0].data, [[y], [x]], order=1, mode='nearest')[0])
    return raw * scale, dict(l=float(np.degrees(l)), b=float(np.degrees(b)), sfd=raw, mapdir=d, file=fn)


def deredden_factor(wave, ebv, r_v=3.1, law='f99'):
    """10**(0.4 A(wave)) for E(B-V) = ebv (law: f99 default, ccm89, od94): multiply fluxes by
    it, variances by its square."""
    if not ebv:
        return np.ones_like(np.asarray(wave, float))
    return 10 ** (0.4 * float(ebv) * curve(law)(wave, r_v))
