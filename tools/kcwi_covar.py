#!/usr/bin/env python
"""
kcwi_covar -- measure the noise covariance of a KcwiKit stack with mock-noise stacks
(de los Reyes et al. 2023; Zhuang et al. 2024, sect. 2.3.2), WITHOUT restacking the data.

Why. kcwi_stack drizzles every exposure onto the output grid (Montage mProjectCube) and
projects the VARIANCE cube the same way, as a weighted mean -- not with the squared
weights a correct propagation needs, and with no record of the covariance it creates
between neighbouring output spaxels. So (i) the variance of one stacked spaxel is not
exactly right and (ii) the noise of a sum of N spaxels grows faster than sqrt(sum of
the variances), up to the correlation length, beyond which it stops growing:

    sigma_true(N) = C [1 + alpha log10(min(N, N_max))] sigma_from_variance_cube(N)

C, alpha, N_max depend only on the geometry (slicer, pixel scales, drizzle factor,
dithers / position angles, the stacking options) -- not on the sky -- so they can be
measured by pushing pure noise through the SAME stacking: every exposure cube is
replaced by Gaussian noise with that exposure's own variance (UNCERT), only a few
narrow wavelength windows are kept (the covariance is spatial; a window of ~200 A costs
~1/20 of a full stack), and kcwi_stack is run on them with the stack's own .par, shift,
preshift, flux and astrometry lists and the same keyword arguments. The stacked mock
cube is then pure noise whose propagated variance is exactly what GIST would read, and
the ratio above is measured on k x k boxes.

    # in the stack folder (with IRAS03582+6012_skl.list/.par/.shift.list), in the
    # environment where kcwi_stack works (kcwikit, Montage):
    kcwi_covar mock IRAS03582+6012_skl.list --preset blue         # mock stacks -> covar_mock/
    kcwi_covar measure covar_mock/IRAS03582+6012_skl              # C, alpha, N_max (+ plot)
    kcwi_covar run IRAS03582+6012_skl.list --preset blue          # both
    kcwi_covar run IRAS03582+6012-red.list --preset red --windows 6600-6800,7700-7900
    kcwi_covar batch --dry-run                                    # every source of this folder: the plan
    kcwi_covar batch --clean                                      # ... blue and red, summary in
                                                                  #   covar_mock/covar_summary.csv
    kcwi_covar batch --sources IRAS03582+6012,ESO353-9 --channels blue

--preset gives the keyword arguments of the stacking notebook (blue: crr_split frame,
drizzle 0.9; red: crr_split auto); --stack-kw '{"drizzle": 0.7, ...}' adds or overrides
any of them. Use exactly what the real stack used, or the mock measures a different
covariance. The real data and stack products are never touched: everything is written
under --out (default covar_mock/<list name>/).

What to do with the result (gist_prep parameter file, [SPATIAL_BINNING]), either
    COV_FILE = covar_<name>.txt        the measured curve (copy it to configFiles/), or
    COVARIANCE = alpha     COV_C = C     COV_NMAX = N_max      its three-number fit
(GIST: S/N of an N-spaxel bin = S/N(variance cube) / (C (1 + alpha log10 min(N, N_max)))).
The curve is the more faithful of the two (the real one bends smoothly towards its
asymptote; the fit is a broken line in log N).
The measurement uses square boxes; GIST's bins are irregular but compact, which is
close. KCWI slices make the covariance anisotropic (wider across the slices): the
strip test printed by 'measure' shows how much.
"""
from __future__ import print_function

import argparse
import glob
import importlib
import json
import os
import shutil
import sys

import numpy as np

PRESETS = {
    # the kcwi_stack calls of stack-multiAGN.ipynb (blue: *_skl.list, red: *-red.list)
    'blue': dict(crr_split=True, crr_split_mode='frame', crr_mask_dir='./crmask', crr_reuse_mask=False,
                 crr_save_mask=False, drizzle=0.9, montagepy=False, keep_trim=False, keep_mont=False,
                 use_astrom=False, overwrite=True),
    'red': dict(crr_split=True, crr_split_mode='auto', crr_mask_dir='./crmask', crr_reuse_mask=False,
                crr_save_mask=False, drizzle=0.9, montagepy=False, keep_trim=False, keep_mont=False,
                use_astrom=False, overwrite=True),
    'none': dict(overwrite=True),
}
DEFAULT_WINDOWS = {'blue': '4150-4350,5050-5250', 'red': '6600-6800,7900-8100'}
PAR_WAVE_KEYS = ('wave_ref', 'nwave', 'dwave')        # .par keys that would fix the output wavelength grid
SIDE_LISTS = ('.shift.list', '.preshift.list', '.pre.list', '.flx.list', '.astrom.list', '.par')


# ---------------------------------------------------------------------------
# mock
# ---------------------------------------------------------------------------
def _wave_axis(h):
    cd = h.get('CD3_3', h.get('CDELT3'))
    return h['CRVAL3'] + (np.arange(h['NAXIS3']) + 1 - h['CRPIX3']) * cd


def _read_list(fn):
    rows = []
    for line in open(fn):
        t = line.split()
        if t and not t[0].startswith('#'):
            rows.append(t)
    return rows


def _cube_files(base):
    """(suffix, i/v/m/e file names) of a list entry, as kcwi_stack finds them."""
    for suf in ('cubes', 'cubed'):
        if os.path.isfile(base + '_i' + suf + '.fits'):
            return suf, dict((k, base + '_' + k + suf + '.fits') for k in 'ivme')
    raise IOError('%s_icubes.fits (or _icubed) not found' % base)


def _slice_hdu(hdu, k0, k1, nw):
    """A 3-D HDU cut to planes k0:k1 (its wavelength WCS shifted); other HDUs unchanged."""
    from astropy.io import fits
    d = hdu.data
    if d is None or d.ndim != 3 or d.shape[0] != nw:
        return hdu.copy()
    h = hdu.header.copy()
    if 'CRPIX3' in h:
        h['CRPIX3'] = h['CRPIX3'] - k0
    h['NAXIS3'] = k1 - k0
    cls = fits.PrimaryHDU if isinstance(hdu, fits.PrimaryHDU) else fits.ImageHDU
    out = cls(np.array(d[k0:k1]), header=h)
    if not isinstance(hdu, fits.PrimaryHDU):
        out.name = hdu.name
    return out


def _noise_like(flux, sigma, rng):
    """Gaussian noise with the given sigma where the exposure has data; elsewhere the
    original value (0 / NaN), so that every mask of kcwi_stack acts identically."""
    ok = np.isfinite(flux) & (flux != 0) & np.isfinite(sigma) & (sigma > 0)
    out = np.array(flux, dtype=np.float64)
    out[ok] = rng.standard_normal(int(ok.sum())) * sigma[ok]
    return out.astype(flux.dtype if flux.dtype.kind == 'f' else np.float32), ok


def make_mock_exposure(base, outdir, lo, hi, rng):
    """Mock noise version of one exposure (all its i/v/m/e cubes), planes lo-hi A only."""
    from astropy.io import fits
    suf, fns = _cube_files(base)
    with fits.open(fns['i']) as hl:
        wave = _wave_axis(hl[0].header)
        nw = wave.size
        k = np.flatnonzero((wave >= lo) & (wave <= hi))
        if k.size < 10:
            raise ValueError('%s: only %d planes in %g-%g A' % (fns['i'], k.size, lo, hi))
        k0, k1 = int(k[0]), int(k[-1]) + 1
        new = fits.HDUList([_slice_hdu(h, k0, k1, nw) for h in hl])
        if len(hl) >= 4:                                  # pyDRP: UNCERT / FLAGS extensions
            if 'UNCERT' not in [h.name for h in hl]:
                raise ValueError('%s has no UNCERT extension' % fns['i'])
            sig = np.array(new['UNCERT'].data, float)
            new[0].data, ok = _noise_like(new[0].data, sig, rng)
            extra = []
        else:                                             # IDL DRP: separate _vcubes / _mcubes
            with fits.open(fns['v']) as hv:
                var = np.array(hv[0].data[k0:k1], float)
                hvh = hv[0].header.copy(); hvh['CRPIX3'] -= k0
            new[0].data, ok = _noise_like(new[0].data, np.sqrt(np.clip(var, 0, None)), rng)
            extra = [('v', fits.PrimaryHDU(var.astype(np.float32), hvh))]
            if os.path.isfile(fns['m']):
                with fits.open(fns['m']) as hm:
                    hmh = hm[0].header.copy(); hmh['CRPIX3'] -= k0
                    extra.append(('m', fits.PrimaryHDU(np.array(hm[0].data[k0:k1]), hmh)))
    new[0].header['HISTORY'] = 'kcwi_covar: Gaussian noise with the UNCERT/variance of this exposure'
    nb = os.path.join(outdir, os.path.basename(base))
    new.writeto(nb + '_i' + suf + '.fits', overwrite=True)
    for kk, h in extra:
        h.writeto(nb + '_' + kk + suf + '.fits', overwrite=True)
    reg = fns['i'].replace('.fits', '.thum.reg')        # region masks used by kcwi_stack
    if os.path.isfile(reg):
        shutil.copy(reg, nb + '_i' + suf + '.thum.reg')
    return nb, wave[k0], wave[k1 - 1], float(ok.mean())


def _copy_side_files(listfn, outdir, base_map):
    """The .par (without fixed wavelength-grid keys) and shift/preshift/flux/astrometry
    lists of the stack, with original paths replaced by the mock ones."""
    stem = listfn[:-len('.list')]
    name = os.path.basename(stem)
    for ext in SIDE_LISTS:
        src = stem + ext
        if not os.path.isfile(src):
            continue
        out = []
        for line in open(src):
            t = line.split()
            if ext == '.par' and t and t[0] in PAR_WAVE_KEYS:
                continue                                  # the mock covers a window only
            for old, new in base_map.items():
                line = line.replace(old, new)
            out.append(line)
        with open(os.path.join(outdir, name + ext), 'w') as f:
            f.writelines(out)


def _stack_module(name):
    for m in ([name] if name else ['kcwikit.kcwi.kcwi', 'kcwi']):
        try:
            return importlib.import_module(m)
        except ImportError:
            continue
    sys.exit('cannot import the KcwiKit stacking module (%s): run in the environment where kcwi_stack works, '
             'or give --module' % (name or 'kcwikit.kcwi.kcwi / kcwi'))


def list_info(listfn):
    """(camera 'blue'/'red', good wavelength range) from the first exposure of a stack list."""
    from astropy.io import fits
    rows = _read_list(listfn)
    if not rows:
        raise ValueError('%s is empty' % listfn)
    suf, fns = _cube_files(rows[0][0])
    h = fits.getheader(fns['i'])
    cam = str(h.get('CAMERA', '')).strip().lower()
    if cam not in ('blue', 'red'):
        cam = 'red' if os.path.basename(listfn).endswith('-red.list') else 'blue'
    w = _wave_axis(h)
    lo, hi = h.get('WAVGOOD0', w[0]), h.get('WAVGOOD1', w[-1])
    return cam, (float(max(lo, w[0])), float(min(hi, w[-1])))


def auto_windows(good, width=200., frac=(0.3, 0.7), margin=60.):
    """Two windows of `width` A inside the good range, centred at 30% and 70% of it. The
    mock noise has no sky lines, so only the range (and the DAR/vachelio edges) matters."""
    lo, hi = good[0] + margin, good[1] - margin
    if hi - lo < width:
        raise ValueError('good range %.0f-%.0f A too short for a %g A window' % (good[0], good[1], width))
    out = []
    for f in frac:
        c = min(max(lo + f * (hi - lo), lo + width / 2.), hi - width / 2.)
        out.append((round(c - width / 2.), round(c + width / 2.)))
    return out


def mock(listfn, windows, out, stack_kw, module=None, seed=1, margin=20.):
    """Write and stack the mock-noise exposures; returns the stacked cube bases."""
    rows = _read_list(listfn)
    name = os.path.basename(listfn)[:-len('.list')]
    kc = _stack_module(module)
    rng = np.random.RandomState(seed)
    here = os.getcwd()
    made = []
    for iw, (lo, hi) in enumerate(windows):
        wdir = os.path.abspath(os.path.join(out, 'w%d_%g-%g' % (iw, lo, hi)))
        if not os.path.isdir(wdir):
            os.makedirs(wdir)
        bmap, lines = {}, []
        for t in rows:
            base = t[0]
            nb, w0, w1, frac = make_mock_exposure(base, wdir, lo - margin, hi + margin, rng)
            bmap[base] = nb
            lines.append(' '.join([nb] + t[1:]) + '\n')
            print('  %s: %.1f-%.1f A, noise in %.0f%% of the voxels' % (os.path.basename(base), w0, w1, 100 * frac))
        with open(os.path.join(wdir, name + '.list'), 'w') as f:
            f.writelines(lines)
        _copy_side_files(os.path.abspath(listfn), wdir, bmap)
        print('stacking the mock exposures of %g-%g A in %s with %s' % (lo, hi, wdir, json.dumps(stack_kw)))
        os.chdir(wdir)
        try:
            kc.kcwi_stack(name + '.list', **stack_kw)
        except TypeError as e:
            os.chdir(here)
            sys.exit('kcwi_stack refused the keyword arguments (%s): this KcwiKit is older/newer than the '
                     'preset; give the ones your stack used with --preset none --stack-kw \'{...}\'' % e)
        finally:
            os.chdir(here)
        made.append(os.path.join(wdir, name))
    return made


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------
def _stack_files(base):
    """i/v/m/e cubes of a stacked product (base = path without _icubes.fits)."""
    out = {}
    for k in 'ivme':
        for suf in ('cubes', 'cubed'):
            for tail in ('_wcs', ''):
                fn = '%s_%s%s%s.fits' % (base, k, suf, tail if k == 'i' else '')
                if os.path.isfile(fn) and k not in out:
                    out[k] = fn
    if 'i' not in out or 'v' not in out:
        raise IOError('%s_icubes.fits / _vcubes.fits not found' % base)
    return out


def _box_sums(a, k):
    """Sums of a (nw, ny, nx) over every k x k box; (nw, ny-k+1, nx-k+1)."""
    c = np.zeros((a.shape[0], a.shape[1] + 1, a.shape[2] + 1))
    c[:, 1:, 1:] = np.cumsum(np.cumsum(a, axis=1), axis=2)
    return c[:, k:, k:] - c[:, :-k, k:] - c[:, k:, :-k] + c[:, :-k, :-k]


def _strip_sums(a, k, axis):
    c = np.cumsum(np.concatenate([np.zeros_like(np.take(a, [0], axis=axis)), a], axis=axis), axis=axis)
    n = a.shape[axis]
    return np.take(c, np.arange(k, n + 1), axis=axis) - np.take(c, np.arange(0, n - k + 1), axis=axis)


def ratio_curve(D, V, G, sizes, chan_step=1):
    """r(N) = std(sum D / sqrt(sum V)) over non-overlapping k x k boxes fully inside G."""
    out = []
    for k in sizes:
        Ds, Vs, Gs = _box_sums(D, k), _box_sums(V, k), _box_sums(G.astype(float), k)
        sel = (Gs[:, ::k, ::k] > k * k - 0.5) & (Vs[:, ::k, ::k] > 0)
        z = (Ds[:, ::k, ::k] / np.sqrt(np.where(Vs[:, ::k, ::k] > 0, Vs[:, ::k, ::k], 1.)))[sel]
        z = z[::chan_step] if chan_step > 1 else z
        if z.size < 30:
            break
        med = np.median(z)
        mad = 1.4826 * np.median(np.abs(z - med))
        z = z[np.abs(z - med) < 6 * mad]
        r = float(np.std(z))
        nbox = int(sel[0].sum())
        out.append((k * k, r, r / np.sqrt(2. * max(z.size / 2., 1.)), nbox))   # /2: channels correlated
    return np.array(out)


def strip_ratio(D, V, G, k, axis):
    """Same for 1 x k strips along x (axis 2) or y (axis 1), non-overlapping."""
    Ds, Vs, Gs = _strip_sums(D, k, axis), _strip_sums(V, k, axis), _strip_sums(G.astype(float), k, axis)
    sl = [slice(None)] * 3; sl[axis] = slice(None, None, k); sl = tuple(sl)
    sel = (Gs[sl] > k - 0.5) & (Vs[sl] > 0)
    z = (Ds[sl] / np.sqrt(np.where(Vs[sl] > 0, Vs[sl], 1.)))[sel]
    return float(np.std(z)) if z.size > 30 else np.nan


def fit_model(N, r, e, floor=0.02):
    """Least-squares C, alpha for each candidate N_max (the measured N, or none). A
    relative error floor keeps the very precise small-N points from dominating: the
    model is only an approximation of the real curve (which approaches its asymptote
    smoothly), and GIST's bins have N ~ 10-1000."""
    e = np.sqrt(np.asarray(e) ** 2 + (floor * np.asarray(r)) ** 2)
    best = None
    cands = list(N[1:]) + [np.inf]
    for nm in cands:
        x = np.log10(np.minimum(N, nm))
        A = np.vstack([np.ones_like(x), x]).T / e[:, None]
        sol, *_ = np.linalg.lstsq(A, r / e, rcond=None)
        c, ca = sol                                     # r = c + (c alpha) x
        chi2 = float(np.sum(((c + ca * x - r) / e) ** 2))
        if best is None or chi2 < best[3] - 1e-9:
            best = (c, ca / c if c else np.nan, nm, chi2)
    x = np.log10(N)
    # GIST's original form (C = 1, no cap) for comparison: r = 1 + alpha log10 N
    a1 = float(np.sum((r - 1) * x / e ** 2) / np.sum(x ** 2 / e ** 2))
    return dict(C=best[0], alpha=best[1], nmax=best[2], chi2=best[3], dof=len(N) - 3, alpha_c1=a1)


def measure(bases, sizes, min_exp=0.5, plot=None, table=None):
    from astropy.io import fits
    results = []
    for base in bases:
        fns = _stack_files(base)
        D = np.array(fits.getdata(fns['i']), float)
        V = np.array(fits.getdata(fns['v']), float)
        G = np.isfinite(D) & np.isfinite(V) & (V > 0) & (D != 0)
        if 'm' in fns:
            G &= np.array(fits.getdata(fns['m'])) == 0
        if 'e' in fns and min_exp > 0:
            E = np.array(fits.getdata(fns['e']), float)
            G &= E >= min_exp * np.nanmedian(E[E > 0])
        D = np.where(G, D, 0.); V = np.where(G, V, 0.)
        cur = ratio_curve(D, V, G, sizes)
        f = fit_model(cur[:, 0], cur[:, 1], cur[:, 2])
        sx = [strip_ratio(D, V, G, k, 2) for k in (2, 4, 8)]
        sy = [strip_ratio(D, V, G, k, 1) for k in (2, 4, 8)]
        results.append(dict(base=base, curve=cur, fit=f, strips=(sx, sy)))
        print('\n%s  (%d planes; %d spaxels with data)' % (base, D.shape[0], int(G[G.shape[0] // 2].sum())))
        print('   N    r = sigma_true / sigma_var    +-     boxes')
        for n, r, e, nb in cur:
            print('%5d   %8.3f                   %6.3f   %6d' % (n, r, e, nb))
        print('fit: C = %.3f, alpha = %.3f, N_max = %s  (chi2 %.1f for %d dof); with C = 1 and no cap: '
              'alpha = %.3f' % (f['C'], f['alpha'], 'none' if not np.isfinite(f['nmax']) else '%d' % f['nmax'],
                                f['chi2'], f['dof'], f['alpha_c1']))
        print('strips 1x2, 1x4, 1x8 along x: %s; along y: %s (anisotropy of the slicer/drizzle)' % (
            ', '.join('%.3f' % v for v in sx), ', '.join('%.3f' % v for v in sy)))
    if len(results) > 1:
        Cs = np.array([r['fit']['C'] for r in results]); As = np.array([r['fit']['alpha'] for r in results])
        print('\nwindows agree to C %.3f +- %.3f, alpha %.3f +- %.3f' % (Cs.mean(), Cs.std(), As.mean(), As.std()))
    if results:
        # one curve from all windows together (the geometry is the same)
        allc = np.concatenate([r['curve'] for r in results])
        Ns = np.unique(allc[:, 0])
        rr = np.array([np.average(allc[allc[:, 0] == n, 1], weights=1 / allc[allc[:, 0] == n, 2] ** 2) for n in Ns])
        ee = np.array([1 / np.sqrt(np.sum(1 / allc[allc[:, 0] == n, 2] ** 2)) for n in Ns])
        f = fit_model(Ns, rr, ee)
        nm = 'None' if not np.isfinite(f['nmax']) else '%d' % f['nmax']
        print('\nall windows: C = %.3f, alpha = %.3f, N_max = %s' % (f['C'], f['alpha'], nm))
        print('gist_prep [SPATIAL_BINNING]:  COVARIANCE = %.3f   COV_C = %.3f   COV_NMAX = %s' % (f['alpha'], f['C'], nm))
        print('(previous single-parameter form, C = 1, no cap: COVARIANCE = %.3f)' % f['alpha_c1'])
        if table:
            with open(table, 'w') as fo:
                fo.write('# kcwi_covar: sigma_true / sigma_from_variance_cube for sums of N spaxels (k x k boxes)\n'
                         '# stacks: %s\n# fit: C = %.4f alpha = %.4f N_max = %s\n# N  ratio  error\n'
                         % (', '.join(r['base'] for r in results), f['C'], f['alpha'], nm))
                for n, r_, e_ in zip(Ns, rr, ee):
                    fo.write('%6d  %.4f  %.4f\n' % (n, r_, e_))
            print('wrote %s -- the measured curve itself: copy it to configFiles/ and set SPATIAL_BINNING '
                  'COV_FILE = %s (used instead of the three numbers)' % (table, os.path.basename(table)))
        if plot:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 4))
            for r in results:
                c = r['curve']
                ax.errorbar(c[:, 0], c[:, 1], c[:, 2], fmt='o', ms=4, label=os.path.basename(os.path.dirname(r['base'])))
            n = np.logspace(0, np.log10(Ns.max()), 100)
            from_n = f['C'] * (1 + f['alpha'] * np.log10(np.minimum(n, f['nmax'])))
            ax.plot(n, from_n, 'k-', label='C=%.2f, alpha=%.2f, N_max=%s' % (f['C'], f['alpha'], nm))
            ax.plot(n, 1 + f['alpha_c1'] * np.log10(n), 'k:', label='1 + %.2f log N' % f['alpha_c1'])
            ax.set_xscale('log'); ax.set_xlabel('N spaxels (k x k boxes)'); ax.set_ylabel('sigma_true / sigma_var')
            ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(plot, dpi=120)
            print('wrote', plot)
        return dict(results=results, fit=f, curve=(Ns, rr, ee), table=table)
    return dict(results=results, fit=None, curve=None, table=None)


# ---------------------------------------------------------------------------
def _windows(s):
    out = []
    for w in s.split(','):
        lo, hi = w.split('-')
        out.append((float(lo), float(hi)))
    return out


SIDE_EXT = ('.shift.list', '.preshift.list', '.pre.list', '.flx.list', '.astrom.list')


def find_stacks(folder, sources=None, channels=('blue', 'red'), blue_suffix='_skl'):
    """
    The stack lists of a stack folder, one per source and channel:
      blue: <src><blue_suffix>.list if it exists (the version actually stacked), else <src>.list
      red : <src><blue_suffix>-red.list if it exists, else <src>-red.list
    Side lists (.shift/.preshift/.pre/.flx/.astrom), copies ('Copy' in the name) and
    backups are ignored. Returns [(source, channel, list path)].
    """
    names = []
    for fn in sorted(glob.glob(os.path.join(folder, '*.list'))):
        b = os.path.basename(fn)
        if b.endswith(SIDE_EXT) or 'Copy' in b or '.bak' in b:
            continue
        names.append(b[:-len('.list')])
    have = set(names)
    srcs = set()
    for n in names:
        s = n[:-len('-red')] if n.endswith('-red') else n
        if blue_suffix and s.endswith(blue_suffix):
            s = s[:-len(blue_suffix)]
        srcs.add(s)
    want = None if not sources else set(sources)
    out = []
    for s in sorted(srcs):
        if want is not None and s not in want:
            continue
        for ch in channels:
            cand = ([s + blue_suffix, s] if ch == 'blue' else [s + blue_suffix + '-red', s + '-red'])
            for c in cand:
                if c in have:
                    out.append((s, ch, os.path.join(folder, c + '.list')))
                    break
    if want:
        miss = sorted(want - set(x[0] for x in out))
        if miss:
            print('no stack list for: %s' % ', '.join(miss))
    return out


def check_stack(listfn):
    """None if the list can be mock-stacked, else the reason it cannot."""
    stem = listfn[:-len('.list')]
    for ext in ('.par', '.shift.list'):
        if not os.path.isfile(stem + ext):
            return 'no %s' % os.path.basename(stem + ext)
    rows = _read_list(listfn)
    if not rows:
        return 'empty list'
    for t in rows:
        try:
            _cube_files(t[0])
        except IOError:
            return 'exposure not found: %s_icubes.fits (disk not mounted?)' % t[0]
    return None


def batch(a, sizes):
    """Mock-stack and measure every source of a stack folder; one summary table."""
    import csv
    import time
    folder = os.path.abspath(a.folder)
    srcs = [x.strip() for x in (a.sources or '').split(',') if x.strip()]
    if a.sources_file:
        srcs += [l.split()[0] for l in open(a.sources_file) if l.strip() and not l.startswith('#')]
    chans = tuple(x.strip() for x in a.channels.split(',') if x.strip())
    jobs = find_stacks(folder, srcs or None, chans, a.blue_suffix)
    out_top = os.path.join(folder, a.out or 'covar_mock')
    curves = os.path.join(out_top, 'curves')
    rows, todo = [], []
    for src, ch, lst in jobs:
        name = os.path.basename(lst)[:-len('.list')]
        table = os.path.join(out_top, name, 'covar_%s.txt' % name)
        why = check_stack(lst)
        if why:
            rows.append(dict(source=src, channel=ch, list=os.path.basename(lst), status='skipped: ' + why)); continue
        if os.path.isfile(table) and not a.redo:
            rows.append(dict(source=src, channel=ch, list=os.path.basename(lst), status='done earlier', curve=table))
            continue
        todo.append((src, ch, lst, name, table))
    print('%d stack lists in %s: %d to process, %d done earlier, %d skipped' % (
        len(jobs), folder, len(todo), sum(r['status'] == 'done earlier' for r in rows),
        sum(r['status'].startswith('skipped') for r in rows)))
    for r in rows:
        if r['status'].startswith('skipped'):
            print('  skip %-28s %-4s %s' % (r['source'], r['channel'], r['status'][9:]))
    for src, ch, lst, name, table in todo:
        print('  todo %-28s %-4s %s' % (src, ch, os.path.basename(lst)))
    if a.dry_run or not todo:
        return
    extra = json.loads(a.stack_kw)
    here = os.getcwd()
    os.chdir(folder)                         # the lists hold absolute paths; side files sit here
    try:
        for k, (src, ch, lst, name, table) in enumerate(todo):
            t0 = time.time()
            print('\n=== [%d/%d] %s (%s): %s' % (k + 1, len(todo), src, ch, os.path.basename(lst)))
            row = dict(source=src, channel=ch, list=os.path.basename(lst))
            try:
                cam, good = list_info(lst)
                preset = a.preset or cam
                kw = dict(PRESETS[preset]); kw.update(extra)
                wins = auto_windows(good, a.width) if a.windows == 'auto' else _windows(a.windows)
                out = os.path.join(out_top, name)
                bases = mock(lst, wins, out, kw, a.module, a.seed, a.margin)
                res = measure(bases, sizes, a.min_exp, os.path.join(out, 'covar.png'), table)
                f = res['fit']
                if not os.path.isdir(curves):
                    os.makedirs(curves)
                shutil.copy(table, os.path.join(curves, os.path.basename(table)))
                row.update(status='ok', camera=cam, windows=' '.join('%g-%g' % w for w in wins),
                           C='%.4f' % f['C'], alpha='%.4f' % f['alpha'],
                           N_max='None' if not np.isfinite(f['nmax']) else '%d' % f['nmax'],
                           alpha_C1='%.4f' % f['alpha_c1'], curve=table, minutes='%.1f' % ((time.time() - t0) / 60.))
                if a.clean:                  # keep the stacked mock cubes and the curve, drop the mock exposures
                    for b in bases:
                        for fn in glob.glob(os.path.join(os.path.dirname(b), '*_i*cube*.fits')) + \
                                glob.glob(os.path.join(os.path.dirname(b), '*_[vme]cube*.fits')):
                            if not os.path.basename(fn).startswith(name):
                                os.remove(fn)
            except SystemExit as e:
                row.update(status='failed: %s' % e)
            except Exception as e:
                row.update(status='failed: %s: %s' % (type(e).__name__, e))
            print('  -> %s' % row['status'])
            rows.append(row)
            _write_summary(os.path.join(out_top, 'covar_summary.csv'), rows)
    finally:
        os.chdir(here)
    summ = os.path.join(out_top, 'covar_summary.csv')
    _write_summary(summ, rows)
    print('\n%-28s %-4s %7s %7s %6s  %s' % ('source', 'ch', 'C', 'alpha', 'N_max', 'status'))
    for r in rows:
        print('%-28s %-4s %7s %7s %6s  %s' % (r['source'], r['channel'], r.get('C', ''), r.get('alpha', ''),
                                              r.get('N_max', ''), r['status']))
    print('\nwrote %s; curves collected in %s (copy each to its target\'s configFiles/ and set '
          'SPATIAL_BINNING COV_FILE = covar_<list>.txt)' % (summ, curves))


def _write_summary(fn, rows):
    import csv
    keys = ['source', 'channel', 'list', 'status', 'C', 'alpha', 'N_max', 'alpha_C1', 'camera', 'windows',
            'minutes', 'curve']
    old = []
    if os.path.isfile(fn):                   # keep earlier results of lists not in this run
        with open(fn) as f:
            redone = set(x['list'] for x in rows if x['status'] != 'done earlier')
            old = [r for r in csv.DictReader(f) if r.get('list') not in redone]
    if not os.path.isdir(os.path.dirname(fn)):
        os.makedirs(os.path.dirname(fn))
    with open(fn, 'w') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in old + [r for r in rows if r['status'] != 'done earlier']:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd')
    for c in ('mock', 'run', 'batch'):
        hlp = {'mock': 'make and stack mock-noise exposures of one list', 'run': 'mock + measure for one list',
               'batch': 'mock + measure every source of a stack folder (blue and red), with a summary table'}[c]
        p = sub.add_parser(c, help=hlp)
        if c == 'batch':
            p.add_argument('folder', nargs='?', default='.', help='stack folder (default: here)')
            p.add_argument('--sources', help='comma list of sources (default: every list in the folder)')
            p.add_argument('--sources-file', help='file with one source per line (e.g. the kcwi_skylevel batch file)')
            p.add_argument('--channels', default='blue,red', help='blue, red or blue,red (default)')
            p.add_argument('--blue-suffix', default='_skl', help='use <src><suffix>.list when it exists (default '
                           '_skl, the version the notebook stacks); "" = plain lists only')
            p.add_argument('--preset', choices=sorted(PRESETS), help='stack options for every list (default: '
                           'blue/red preset from each list\'s CAMERA)')
            p.add_argument('--width', type=float, default=200., help='window width for --windows auto [A]')
            p.add_argument('--redo', action='store_true', help='redo lists that already have a curve')
            p.add_argument('--clean', action='store_true', help='delete the mock EXPOSURE cubes after measuring '
                           '(the stacked mock cubes and the curve stay)')
            p.add_argument('--dry-run', action='store_true', help='only list what would be done')
        else:
            p.add_argument('list', help='the stack list, e.g. IRAS03582+6012_skl.list (run in its folder)')
            p.add_argument('--preset', choices=sorted(PRESETS), help='kcwi_stack keyword arguments of the stacking '
                           'notebook (default: blue/red from the CAMERA of the first exposure)')
            p.add_argument('--out', help='output folder (default covar_mock/<list name>)')
        p.add_argument('--windows', default='auto', help='observed windows [A], e.g. 4150-4350,5050-5250; auto '
                       '(default) = two 200 A windows at 30%% and 70%% of the good range of the first exposure')
        p.add_argument('--stack-kw', default='{}', help='JSON dict of kcwi_stack keyword arguments, added to / '
                       'overriding the preset')
        p.add_argument('--module', help='module with kcwi_stack (default kcwikit.kcwi.kcwi, then kcwi)')
        p.add_argument('--seed', type=int, default=1)
        p.add_argument('--margin', type=float, default=20., help='extra A kept on each side of a window')
        if c == 'batch':
            p.add_argument('--out', help='output folder inside the stack folder (default covar_mock)')
    m = sub.add_parser('measure', help='measure C, alpha, N_max on stacked mock cubes')
    m.add_argument('bases', nargs='+', help='stacked mock products: covar_mock/<name> (all windows) or '
                   'covar_mock/<name>/w0_.../<name>')
    for p in (sub.choices['run'], sub.choices['batch'], m):
        p.add_argument('--kmax', type=int, default=20, help='largest box side [spaxels] (default 20)')
        p.add_argument('--min-exp', type=float, default=0.5, help='use spaxels with exposure >= this x the median '
                       '(default 0.5: drops the thin edges)')
        if p is not sub.choices['batch']:
            p.add_argument('--plot', default='auto', help='PNG of the ratio curve (default <out>/covar.png; none = off)')
    a = ap.parse_args()
    if not a.cmd:
        ap.print_help(); return
    sizes = [k for k in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 17, 20, 25, 30) if k <= getattr(a, 'kmax', 20)]
    if a.cmd == 'batch':
        batch(a, sizes)
        return
    if a.cmd in ('mock', 'run'):
        name = os.path.basename(a.list)[:-len('.list')]
        out = a.out or os.path.join('covar_mock', name)
        cam, good = list_info(a.list)
        kw = dict(PRESETS[a.preset or cam]); kw.update(json.loads(a.stack_kw))
        wins = auto_windows(good) if a.windows == 'auto' else _windows(a.windows)
        print('%s: %s camera, good range %.0f-%.0f A; windows %s; preset %s' % (
            name, cam, good[0], good[1], ', '.join('%g-%g' % w for w in wins), a.preset or cam))
        bases = mock(a.list, wins, out, kw, a.module, a.seed, a.margin)
        if a.cmd == 'mock':
            print('\nnow: kcwi_covar measure %s' % out); return
        plot = None if a.plot == 'none' else (os.path.join(out, 'covar.png') if a.plot == 'auto' else a.plot)
        measure(bases, sizes, a.min_exp, plot, os.path.join(out, 'covar_%s.txt' % name))
        return
    bases = []
    for b in a.bases:
        if os.path.isdir(b):                       # covar_mock/<name>: every window inside
            name = os.path.basename(os.path.normpath(b))
            bases += sorted(os.path.join(d, name) for d in glob.glob(os.path.join(b, 'w*')) if os.path.isdir(d))
        else:
            bases.append(b[:-len('_icubes.fits')] if b.endswith('_icubes.fits') else b)
    top = a.bases[0] if os.path.isdir(a.bases[0]) else os.path.dirname(os.path.abspath(bases[0]))
    plot = None if a.plot == 'none' else (os.path.join(top, 'covar.png') if a.plot == 'auto' else a.plot)
    measure(bases, sizes, a.min_exp, plot,
            os.path.join(top, 'covar_%s.txt' % os.path.basename(os.path.normpath(bases[0]))))


if __name__ == '__main__':
    main()
