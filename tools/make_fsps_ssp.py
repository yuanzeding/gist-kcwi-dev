#!/usr/bin/env python
"""
make_fsps_ssp.py -- young (or any) single stellar populations from FSPS, written as
E-MILES-style FITS files that GIST's 'miles' template reader accepts, so they can sit in
one template folder with E-MILES and extend it below its 30 Myr floor.

    # FSPS 1-20 Myr at the E-MILES metallicities, into the E-MILES folder
    python make_fsps_ssp.py --dest spectralTemplates/EMILES_BASTI_UN     # after get_ssp
    # first check the join: FSPS vs E-MILES at ages both have
    python make_fsps_ssp.py --compare spectralTemplates/EMILES_BASTI_UN --ages 0.03,0.06,0.1,0.3

Needs python-fsps (and SPS_HOME pointing to the FSPS data, ~/software/fsps here), so it
runs in its own environment, not in gist-dev (python-fsps needs a newer Python):
    conda create -n fsps python=3.11 numpy astropy matplotlib -y && conda activate fsps
    export SPS_HOME=~/software/fsps && pip install fsps

The FSPS set-up (python-fsps default build: MIST isochrones + MILES stellar library,
WMBasic hot stars above 25 kK, Wolf-Rayet spectra):
  * SSPs (sfh = 0), Salpeter IMF with the E-MILES mass limits 0.1-100 Msun
    (imf_type = 0, imf_lower_limit = 0.1, imf_upper_limit = 100)
  * no nebular emission or continuum, no dust: the gas is fitted (or masked) by GIST,
    and dust is absorbed by the multiplicative polynomial
  * metallicity: logzsol = [M/H] (MIST, scaled-solar), the same values as the E-MILES files
  * spectrum per solar mass formed, L_sun/A (E-MILES units), wavelengths converted from
    FSPS's vacuum to air (as E-MILES and the KCWI cubes), linearly resampled to 0.9 A
    (the E-MILES optical sampling), 3525-7500 A only: the MILES-resolution part of FSPS
    (2.5 A FWHM, same lsf_MILES as E-MILES). Outside it FSPS falls back to low-resolution
    libraries. The hottest stars (WMBasic) have less line detail than MILES: for < ~10 Myr
    the continuum is right, individual line profiles only approximate.
Mixing with E-MILES in ONE folder (what GIST needs to use both in one fit) -- GIST's miles.py
reader assumes, and this script therefore enforces:
  * the same wavelength grid in every file (it uses the first file's CRVAL1/CDELT1/NAXIS1 for
    all): the FSPS spectra are written on the grid of the E-MILES files found in --dest. Outside
    3525-7500 A they come from FSPS's low-resolution libraries -- harmless as long as the fit
    range +-150 A (GIST's template margin) stays inside it, e.g. LMIN >= 3675 for E-MILES+FSPS
  * file names that sort by age within a metallicity: E-MILES's own pattern,
        Eun1.30Z<m|p>X.XXT<age in Gyr, 7 chars>_iTp0.00_baseFe_FSPS.fits
    (the _FSPS tag keeps them recognisable; GIST reads age and [M/H] from the name)
  * a complete grid: every metallicity of the folder at every age, no age twice -- FSPS ages
    must be younger than the youngest E-MILES age there, and the metallicities default to
    exactly those of the E-MILES files in --dest.
"""
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
from astropy.io import fits

DEFAULT_AGES = '0.001,0.002,0.003,0.005,0.007,0.010,0.015,0.020'   # Gyr; E-MILES (BaSTI) starts at 0.030
DEFAULT_METALS = '-0.66,-0.35,0.06,0.26,0.40'
LAM0, LAM1, DLAM = 3525., 7500., 0.9


def vac2air(v):
    s2 = (1e4 / v) ** 2
    return v / (1. + 0.0000834254 + 0.02406147 / (130. - s2) + 0.00015998 / (38.9 - s2))


def name(age, z, tag='baseFe'):
    return 'Eun1.30Z%s%.2fT%07.4f_iTp0.00_%s_FSPS.fits' % ('m' if z < 0 else 'p', abs(z), age, tag)


def emiles_in(folder):
    """E-MILES files (not ours) in folder: (ages, metallicities, grid (crval, cdelt, naxis), alpha
    tags) or None."""
    import glob
    import re
    files = sorted(f for f in glob.glob(os.path.join(folder, '*.fits')) if not f.endswith('_FSPS.fits'))
    pat = re.compile(r'Z([mp])(\d+\.\d+)T(\d+\.\d+)')
    apat = re.compile(r'_(E[mp]\d\.\d\d)')
    ages, mets, tags = set(), set(), set()
    for f in files:
        g = pat.search(os.path.basename(f))
        if g:
            mets.add(round((-1 if g.group(1) == 'm' else 1) * float(g.group(2)), 2))
            ages.add(float(g.group(3)))
            t = apat.search(os.path.basename(f))
            tags.add(t.group(1) if t else 'baseFe')
    if not ages:
        return None
    h = fits.getheader(files[0])
    if h.get('CRPIX1', 1.) != 1.:
        sys.exit('%s: CRPIX1 != 1 -- GIST assumes the grid starts at CRVAL1' % files[0])
    return sorted(ages), sorted(mets), (float(h['CRVAL1']), float(h['CDELT1']), int(h['NAXIS1'])), sorted(tags)


def make_sp():
    try:
        import fsps
    except ImportError:
        sys.exit('python-fsps is not installed in this environment (see the header of this script)')
    sp = fsps.StellarPopulation(zcontinuous=1, sfh=0, imf_type=0, imf_lower_limit=0.1, imf_upper_limit=100.,
                                add_neb_emission=False, add_neb_continuum=False, add_dust_emission=False,
                                dust_type=0, dust1=0., dust2=0., logzsol=0.)
    libs = getattr(sp, 'libraries', None)
    if libs is not None:
        print('FSPS libraries:', libs)
        lb = [x.decode() if isinstance(x, bytes) else str(x) for x in libs]
        if not any('miles' in x.lower() for x in lb):
            sys.exit('this python-fsps was not built with the MILES library: the optical would be low resolution')
    return sp


def spectrum(sp, age, z, grid=None):
    """FSPS SSP on a linear air-wavelength grid: grid = (crval, cdelt, naxis), default 3525-7500 A."""
    sp.params['logzsol'] = z
    wv, fl = sp.get_spectrum(tage=age, peraa=True)          # vacuum A, L_sun/A per Msun formed
    wa = vac2air(np.asarray(wv, float))
    if grid is None:
        g = np.arange(LAM0, LAM1 + DLAM / 2, DLAM)
    else:
        g = grid[0] + grid[1] * np.arange(grid[2])
    return g, np.interp(g, wa, np.asarray(fl, float), left=0., right=0.)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dest', default='spectralTemplates/FSPS_MIST_UN', help='folder for the FITS files')
    ap.add_argument('--ages', default=DEFAULT_AGES, help='ages in Gyr, comma-separated (default 1-20 Myr)')
    ap.add_argument('--metals', help='[M/H] values (default: those of the E-MILES files in --dest, else %s)'
                    % DEFAULT_METALS)
    ap.add_argument('--compare', metavar='EMILES_DIR',
                    help='do not write: compare FSPS with the E-MILES files in this folder at the ages both have')
    argv = sys.argv[1:]
    for k in range(len(argv) - 1):
        if argv[k] == '--metals' and argv[k + 1].startswith('-'):
            argv[k:k + 2] = ['--metals=' + argv[k + 1]]
            break
    a = ap.parse_args(argv)
    ages = [float(x) for x in a.ages.split(',') if x.strip()]
    em = emiles_in(a.compare or a.dest) if os.path.isdir(a.compare or a.dest) else None
    if a.metals:
        metals = [float(x) for x in a.metals.split(',') if x.strip()]
    else:
        metals = em[1] if em else [float(x) for x in DEFAULT_METALS.split(',')]

    if a.compare:
        sp = make_sp()
        compare(sp, a.compare, ages, metals)
        return

    grid = None
    if em:
        e_ages, e_mets, grid, tags = em
        if any(t >= e_ages[0] for t in ages):
            sys.exit('FSPS ages must be younger than the youngest E-MILES age in %s (%g Gyr): a duplicate age '
                     'breaks GIST\'s template grid' % (a.dest, e_ages[0]))
        if sorted(round(z, 2) for z in metals) != e_mets:
            sys.exit('the metallicities must be exactly those of the E-MILES files in %s: %s (GIST needs a complete '
                     'age x metallicity grid)' % (a.dest, ', '.join('%+.2f' % z for z in e_mets)))
        lo, hi = grid[0], grid[0] + grid[1] * (grid[2] - 1)
        print('E-MILES in %s: %d ages from %g Gyr, [M/H] %s; writing FSPS on its grid (%.1f-%.1f A, %.3f A)'
              % (a.dest, len(e_ages), e_ages[0], ', '.join('%+.2f' % z for z in e_mets), lo, hi, grid[1]))
        if tags != ['baseFe']:
            print('alpha-variable E-MILES (%s): the FSPS spectra (scaled-solar MIST) are written once per tag so '
                  'the grid stays complete -- at < 30 Myr the light is dominated by hot stars whose spectra '
                  'barely depend on [alpha/Fe], but the young [alpha/Fe] is then NOT measured' % ', '.join(tags))
    else:
        tags = ['baseFe']
    sp = make_sp()

    if not os.path.isdir(a.dest):
        os.makedirs(a.dest)
    for z in metals:
        for t in ages:
            g, f = spectrum(sp, t, z, grid)
            h = fits.Header()
            h['CRPIX1'], h['CRVAL1'], h['CDELT1'] = 1., g[0], g[1] - g[0]
            h['CTYPE1'] = 'AWAV'
            h['BUNIT'] = 'Lsun/A/Msun'
            h['COMMENT'] = 'FSPS SSP (MIST + MILES), Salpeter 0.1-100 Msun, no nebular emission/dust'
            h['AGE_GYR'], h['MH'] = t, z
            for tag in tags:
                fits.writeto(os.path.join(a.dest, name(t, z, tag)), f.astype('f4'), h, overwrite=True)
        print('[M/H] %+.2f: %d ages written' % (z, len(ages)))
    print('wrote %d files into %s/ (LSF_TEMP: lsf_MILES, 2.51 A)' % (len(ages) * len(metals) * len(tags), a.dest))


def compare(sp, edir, ages, metals):
    """FSPS vs E-MILES at common ages: both smoothed to 2.5 A (they already are), normalised
    at 4500-5500 A; prints the continuum ratio (blue/red) and the rms of the line residuals."""
    import glob
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import median_filter
    fig, axs = plt.subplots(len(ages), 1, figsize=(12, 2.6 * len(ages)), squeeze=False)
    z = min(metals, key=lambda m: abs(m - 0.06))
    for ax, t in zip(axs[:, 0], ages):
        cand = [c for c in glob.glob(os.path.join(edir, '*Z%s%.2fT%07.4f*.fits' % ('m' if z < 0 else 'p', abs(z), t)))
                if not c.endswith('_FSPS.fits')]
        if not cand:
            ax.set_title('no E-MILES file at %g Gyr, [M/H] %+.2f' % (t, z)); continue
        e = fits.open(cand[0])[0]
        el = e.header['CRVAL1'] + e.header['CDELT1'] * (np.arange(e.data.size) + 1 - e.header.get('CRPIX1', 1))
        g, f = spectrum(sp, t, z)
        ef = np.interp(g, el, np.ravel(e.data))
        k = (g > 4500) & (g < 5500)
        f, ef = f / np.median(f[k]), ef / np.median(ef[k])
        blue = (g > 3800) & (g < 4100)
        n = int(100 / DLAM) | 1
        resid = (f / median_filter(f, n)) - (ef / median_filter(ef, n))
        use = (g > 3700) & (g < 5400)
        print('%6.3f Gyr [M/H] %+.2f: FSPS/E-MILES continuum at 3800-4100 A = %.3f; rms of line residuals '
              '(3700-5400 A) = %.3f' % (t, z, np.median(f[blue] / ef[blue]), np.std(resid[use])))
        ax.plot(g, ef, lw=0.6, label='E-MILES (BaSTI)'); ax.plot(g, f, lw=0.6, label='FSPS (MIST)')
        ax.set_xlim(3600, 5500); ax.set_title('%g Gyr, [M/H] %+.2f' % (t, z), fontsize=9); ax.legend(fontsize=8)
    axs[-1, 0].set_xlabel('air wavelength [A]')
    fig.tight_layout(); fig.savefig('fsps_vs_emiles.png', dpi=110)
    print('wrote fsps_vs_emiles.png')


if __name__ == '__main__':
    main()
