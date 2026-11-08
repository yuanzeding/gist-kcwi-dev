"""
Hierarchical, morphology-aware spatial binning for GIST.

  SPATIAL_BINNING  METHOD hierarchical   TARGET_SNR 35   COVARIANCE 0.0
  optional keys (add columns to MasterConfig; defaults in brackets):
    SB_TOL      [0.10]  largest surface-brightness difference allowed between two
                        regions that merge, in dex (0.1 dex = 26%)
    COMPACT     [0.3]   weight of the shape penalty that keeps bins compact
    MAX_ELONG   [2.5]   hard limit on a bin's rms radius relative to a disc of the
                        same area: stops bins from wrapping round the galaxy along
                        an isophote, which would average the two sides of a
                        rotating disc
    MAX_DPA     [0=off] largest span in position angle of a bin, seen from the
                        centre (READ_DATA ORIGIN), in degrees; not applied within
    CORE_RADIUS [1.0]   arcsec of the centre. Always on, whatever the centre:
                        a bin must contain its own centroid, so no bin is a ring
                        or a horseshoe round a nucleus (several nuclei included)
    SB_RMS      [SB_TOL/2] largest rms of log SB inside a merged bin [dex] (0 = off).
                        The SB_TOL test compares region MEANS only, so a region
                        growing along a gradient could drift and end up spanning
                        ~2 x SB_TOL; this caps the spread within the bin itself.
                        Scaled up with the tolerance in the relaxed passes.
    SPEC_WEIGHT [0]     weight of the spectral-similarity term in the cost (0 = off)
    SPEC_Z      [0=off] spectral consistency TEST: a merge is refused when the two
                        regions' normalised spectra differ by more than the noise,
                        chi2_nu > 1 + SPEC_Z sqrt(2/nu)  (SPEC_Z ~ 3). Unlike
                        SPEC_WEIGHT it needs no tuning against the SB term. Kept
                        in every S/N-growth pass; only the final absorption of
                        leftovers (< MIN_SNR_FRAC x target) ignores it.
    SPEC_LMIN/SPEC_LMAX [whole spectrum]  rest-frame range for the spectral term [A]
    SPEC_WINDOWS [none] rest-frame feature windows, e.g. 3920-3990,4080-4120,4840-4885
                        (one word, no spaces); each is cut into chunks of SPEC_DLAM.
                        Without it the range is cut into SPEC_NCHUNK equal chunks.
    SPEC_FEATURES [none] named stellar absorption features to compare, e.g. CaHK,Mgb,CaT,
                        or 'stellar' = CaHK,Hd,Gband,Fe4383,Hb,Mgb,Fe52,CaT (windows in
                        STELLAR_FEATURES; NaD on request: it is also interstellar). With
                        it, SPEC_NORM defaults to 'local'. Can be combined with SPEC_WINDOWS.
    SPEC_MAXMASK [0.5]  a feature / window with more than this fraction of its channels
                        masked by SPEC_MASK (sky, lines) or without data (dichroic gap,
                        outside the cube) is left out of the comparison entirely
    SPEC_DLAM   [10]    chunk width inside SPEC_FEATURES / SPEC_WINDOWS [A]
    SPEC_NCHUNK [48]    number of chunks when SPEC_WINDOWS is not given
    SPEC_NORM   [S/N window; 'local' with SPEC_FEATURES]  'local' = every feature window
                        divided by its own total, so each feature's depth and shape are
                        compared and dust, flux calibration and the slope between the
                        features are not. Otherwise rest-frame continuum windows that define the amplitude
                        the spectra are divided by, e.g. 4400-4500,5400-5500. Default:
                        READ_DATA LMIN_SNR-LMAX_SNR with READ_DATA SNR_MASK applied
                        (as for the S/N itself); 'total' = the total flux of the
                        chunks (then a strong emission line sets the normalisation).
                        The windows may overlap the feature chunks (covariance kept).
    SPEC_MASK   [KIN SPEC_MASK] specMask file (in CONFIG_DIR) whose intervals are left
                        out of the spectral term: emission lines, sky residuals,
                        telluric bands, gap. False = no mask (e.g. to separate
                        regions by their emission lines on purpose).
    SB_FWHM     [3]     smoothing of the surface-brightness map, in spaxels
    MIN_SNR_FRAC[0.5]   a region left below this fraction of TARGET_SNR at the
                        end is absorbed by its most similar neighbouring bin

How it works
  Every unmasked spaxel starts as its own region; a spaxel already above the
  target S/N is a finished bin. Adjacent regions are merged pairwise, cheapest
  first (agglomerative clustering on the spaxel adjacency graph). The cost of
  a merge is
      |log SB_a - log SB_b| / SB_TOL                (follow the isophotes)
    + COMPACT x (r_rms / r_disc - 1)                (stay round-ish)
    + SPEC_WEIGHT x (chi2_nu(spectra a vs b) - 1)   (optional: similar spectra)
  with SB = the mean of a lightly smoothed surface-brightness map over the
  region. Merges with |dlog SB| > SB_TOL, with an rms of log SB above SB_RMS
  inside the merged bin, or (SPEC_Z > 0) with inconsistent spectra are not
  allowed. A region stops
  growing as soon as it reaches the target S/N. Regions that cannot reach it
  under the SB tolerance (because the light changes too fast there) are then
  merged with relaxed tolerance; what is still below MIN_SNR_FRAC x target is
  absorbed by the most similar neighbouring bin.

  Spectral term: each spaxel's spectrum is summed in chunks (flux and
  variance; voxels with an 'infinite' variance -- long masked runs, the
  dichroic gap -- are left out and flag their chunk as incomplete). Two
  regions are compared on the chunks complete in all their members, after
  dividing by the amplitude (SPEC_NORM windows, or the chunk total), with
  the SPEC_MASK intervals left out, and with the
  first-order variance of that ratio (amplitude error and its covariance
  with the chunk included): chi2_nu = mean (x_a - x_b)^2 / (var_a + var_b).
  The chi2 is diagonal, i.e. approximate; spatially correlated noise is
  allowed for with the COVARIANCE factor, squared, on each region's variance.

  The result: bins follow the isophotes (elongated along them, narrow across
  them), the nucleus and any knot stay separate from the disc around them,
  and the S/N target is met where the light allows. The surface-brightness
  criterion is the one of the Pipe3D continuum-segmentation binning (Sanchez
  et al. 2016, sect. 2.1); the agglomerative merging and the optional
  spectral term follow the hierarchical clustering of CAPIVARA (de Souza et
  al. 2024), with spatial contiguity and an S/N stopping rule added.

Output: the same _table.fits as the voronoi module (masked spaxels get the
negative ID of the nearest bin), so all later modules work unchanged.
"""
import heapq
import logging
import os

import numpy as np
from astropy.io import fits
from scipy import ndimage

from printStatus import printStatus

from gistPipeline.spatialBinning.voronoi import save_table, find_nearest_voronoibin
from gistPipeline.auxiliary import noisetools as _noise


def _opt(config, key, default):
    v = config['SPATIAL_BINNING'].get(key, default)
    return default if v is None else v


def _grid(x, y, pixelsize):
    """Integer (row, col) of every spaxel on its regular grid."""
    i = np.rint((y - y.min()) / pixelsize).astype(int)
    j = np.rint((x - x.min()) / pixelsize).astype(int)
    return i, j


def _windows(v):
    """'3920-3990,4080-4120' (MasterConfig: one word) or a list of pairs -> [(lo, hi), ...]; None if unset."""
    if v is None or (isinstance(v, str) and v.strip().lower() in ('', 'none', 'false')):
        return None
    if isinstance(v, str):
        out = []
        for tok in v.replace(' ', '').split(','):
            if tok:
                lo, hi = tok.split('-')
                out.append((float(lo), float(hi)))
    else:
        out = [(float(a), float(b)) for a, b in v]
    for lo, hi in out:
        if not hi > lo:
            raise ValueError('wavelength window %g-%g is empty' % (lo, hi))
    return out or None


# Rest-frame windows of stellar absorption features, each wide enough to hold the feature
# and its pseudo-continua (Lick bands; Cenarro et al. 2001 for the Ca triplet), so that a
# window normalised by its own flux measures the depth/shape of the feature.
STELLAR_FEATURES = [
    ('CaHK', 3900., 4010.),      # Ca II K 3934, H 3969 (+ He / [NeIII] masked by SPEC_MASK)
    ('Hd', 4040., 4161.),        # H delta (Lick HdA with its continua)
    ('Gband', 4266., 4336.),     # CH G band (Lick G4300)
    ('Fe4383', 4359., 4456.),    # Lick Fe4383
    ('Hb', 4827., 4892.),        # H beta absorption (emission core masked)
    ('Mgb', 5142., 5207.),       # Mg b triplet (Lick Mg b)
    ('Fe52', 5233., 5367.),      # Fe5270 + Fe5335
    ('NaD', 5860., 5925.),       # Na D -- also interstellar: not in 'stellar'
    ('CaT', 8474., 8690.),       # Ca II triplet 8498, 8542, 8662 with continua
]
FEATURE_SETS = {'stellar': ['CaHK', 'Hd', 'Gband', 'Fe4383', 'Hb', 'Mgb', 'Fe52', 'CaT']}


def _features(v):
    """SPEC_FEATURES: 'stellar', or names from STELLAR_FEATURES, comma-separated
    (one word, e.g. CaHK,Mgb,CaT) -> [(name, lo, hi), ...]; None if unset."""
    if v is None or (isinstance(v, str) and v.strip().lower() in ('', 'none', 'false')):
        return None
    table = dict((n.lower(), (n, lo, hi)) for n, lo, hi in STELLAR_FEATURES)
    names = []
    for tok in (v if not isinstance(v, str) else v.replace(' ', '').split(',')):
        if not tok:
            continue
        if tok.lower() in FEATURE_SETS:
            names += FEATURE_SETS[tok.lower()]
        else:
            names.append(tok)
    out = []
    for nm in names:
        if nm.lower() not in table:
            raise ValueError('SPEC_FEATURES: unknown feature %r (known: %s, or %s)' % (
                nm, ', '.join(n for n, _, _ in STELLAR_FEATURES), ', '.join(FEATURE_SETS)))
        if table[nm.lower()] not in out:
            out.append(table[nm.lower()])
    return out or None


BAD_VAR = 1e6   # a voxel with variance > BAD_VAR x the median variance is a missing voxel


def _mask_file(fn, wave, z):
    """True where a GIST specMask file (centre, width, comment) masks wave (rest frame;
    a comment 'sky' means an observed centre, as in GIST's spectralMasking)."""
    cut = np.zeros(wave.size, bool)
    for line in open(fn):
        r = line.split('#')[0].split()
        if len(r) < 2:
            continue
        try:
            c, w = float(r[0]), float(r[1])
        except ValueError:
            continue
        if len(r) > 2 and r[2].lower() == 'sky':
            c = c / (1 + z)
        cut |= (wave >= c - w / 2.) & (wave <= c + w / 2.)
    return cut


def _spectral_features(cube, idx, lmin, lmax, nchunk=48, windows=None, dlam=10., norm=None,
                       fmask=None, nmask=None, maxmask=0.5, local=False):
    """
    Per-spaxel chunked spectra for the spectral term.

    lmin, lmax  range of the feature chunks; windows: feature windows inside it, as (lo, hi)
                or (name, lo, hi) (else nchunk equal chunks); dlam: chunk width inside windows
    maxmask     a window with more than this fraction of its channels masked (SPEC_MASK)
                or without data (gap, outside the cube) is left out entirely
    local       each window is normalised by its own total (norm is then ignored): the
                comparison sees the shape of each feature, not the slope between them
    norm        None: amplitude = total of the feature chunks; else list of windows,
                one amplitude chunk each (they may overlap the features: handled)
    fmask       channels masked everywhere (SPEC_MASK); nmask: also masked in the
                amplitude windows (SNR_MASK)

    Returns F, V   (nspax, nchk)  flux and variance summed over the good voxels of each chunk
            G      (nchk,)        window index of each feature chunk (-1: amplitude chunk)
            B      (nspax, nchk)  True where the chunk has a missing voxel in that spaxel
            CV     (nspax, nchk)  variance shared by a feature chunk and the amplitude
                                  windows (None when norm is None: then it is V itself)
            is_feat, is_norm (nchk,)  chunk enters the shape comparison / the amplitude
            info   dict for the log
    Missing voxels: non-finite, variance <= 0, or the reader's 'infinite' variance
    (long masked runs, the dichroic gap). Channels missing in most spaxels are dropped.
    """
    w = np.asarray(cube['wave'])
    rng = np.ones(w.size, bool)
    if lmin is not None:
        rng &= w >= lmin
    if lmax is not None:
        rng &= w <= lmax
    inorm = np.zeros(w.size, bool)
    for lo, hi in (norm or []):
        inorm |= (w >= lo) & (w <= hi)
    need = np.where(rng | inorm)[0]
    if need.size == 0:
        raise ValueError('no channel in SPEC_LMIN-SPEC_LMAX or the amplitude windows')

    # median variance per channel -> the 'infinite variance' threshold; bad fraction per channel
    medvar = np.full(w.size, np.nan); fbad = np.ones(w.size)
    for s0 in range(0, need.size, 256):
        kb = need[s0:s0 + 256]
        e = cube['error'][kb][:, idx]
        ok = np.isfinite(e) & (e > 0)
        medvar[kb] = np.array([np.median(r[o]) if o.any() else np.nan for r, o in zip(e, ok)])
    good = np.isfinite(medvar)
    vcut = BAD_VAR * np.median(medvar[good]) if good.any() else np.inf
    for s0 in range(0, need.size, 256):
        kb = need[s0:s0 + 256]
        s = cube['spec'][kb][:, idx]; e = cube['error'][kb][:, idx]
        with np.errstate(invalid='ignore'):
            bad = ~np.isfinite(s) | ~np.isfinite(e) | ~(e > 0) | (e > vcut)
        fbad[kb] = bad.mean(axis=1)
    usable = fbad < 0.5                       # dichroic gap, trimmed ends: no information anywhere
    fm = np.zeros(w.size, bool) if fmask is None else np.asarray(fmask, bool)
    nm = np.zeros(w.size, bool) if nmask is None else np.asarray(nmask, bool)
    sel = rng & usable & ~fm
    nsel = inorm & usable & ~fm & ~nm

    def chunks_of(mask, n=None, width=None):
        kk = np.where(mask)[0]
        if kk.size == 0:
            return []
        if n is None:
            n = max(1, int(np.ceil((w[kk[-1]] - w[kk[0]]) / width)))
        n = min(n, kk.size)
        edges = np.linspace(0, kk.size, n + 1).astype(int)
        return [kk[edges[c]:edges[c + 1]] for c in range(n) if edges[c + 1] > edges[c]]

    report = []
    if windows:
        wl = [tuple(x) if len(x) == 3 else ('%g-%g' % tuple(x),) + tuple(x) for x in windows]
        # merge overlapping windows (each channel compared once)
        wl.sort(key=lambda t: t[1])
        merged = []
        for nm, lo, hi in wl:
            if merged and lo <= merged[-1][2]:
                merged[-1] = (merged[-1][0] + '+' + nm, merged[-1][1], max(hi, merged[-1][2]))
            else:
                merged.append((nm, lo, hi))
        feat, gid, kept = [], [], 0
        dw = np.median(np.diff(w)) if w.size > 1 else 1.
        for nm, lo, hi in merged:
            inwin = rng & (w >= lo) & (w <= hi)
            nexp = max(int(round((hi - lo) / dw)), 1)     # channels the window should have
            frac = 1. - float(np.sum(inwin & usable & ~fm)) / nexp
            ok = frac <= maxmask
            report.append((nm, lo, hi, min(max(frac, 0.), 1.), ok))
            if not ok:
                continue
            ch = chunks_of(inwin & sel, width=dlam)
            if local and len(ch) < 2:
                ch = chunks_of(inwin & sel, n=2)      # a locally normalised window needs >= 2 chunks
            feat += ch; gid += [kept] * len(ch); kept += 1
        if not feat:
            raise ValueError('no feature window survives SPEC_MAXMASK = %g: %s' % (maxmask, ', '.join(
                '%s %.0f%% masked' % (r[0], 100 * r[3]) for r in report)))
    else:
        feat = chunks_of(sel, n=int(nchunk))
        gid = [0] * len(feat)
    if local:
        norm = None
    if not feat:
        raise ValueError('no usable channel for the spectral term (SPEC_WINDOWS / SPEC_LMIN-SPEC_LMAX, '
                         'after SPEC_MASK)')
    nrm = []
    if norm:
        for lo, hi in norm:
            nrm += chunks_of(nsel & (w >= lo) & (w <= hi), n=1)
        if not nrm:
            raise ValueError('the amplitude windows (SPEC_NORM) have no usable channel after masking')
    groups = feat + nrm
    nf = len(feat)
    G = np.array(gid + [-1] * len(nrm), int)
    is_feat = np.arange(len(groups)) < nf
    is_norm = ~is_feat if norm else is_feat.copy()

    F = np.zeros((idx.size, len(groups))); V = np.zeros_like(F); B = np.zeros(F.shape, bool)
    CV = np.zeros_like(F) if norm else None
    for c, kk in enumerate(groups):
        s = cube['spec'][kk][:, idx]; e = cube['error'][kk][:, idx]
        with np.errstate(invalid='ignore'):
            bad = ~np.isfinite(s) | ~np.isfinite(e) | ~(e > 0) | (e > vcut)
        B[:, c] = bad.any(axis=0)
        e0 = np.where(bad, 0., e)
        F[:, c] = np.where(bad, 0., s).sum(axis=0)
        V[:, c] = e0.sum(axis=0)
        if norm and c < nf:
            CV[:, c] = e0[nsel[kk]].sum(axis=0)      # channels also in the amplitude
    info = dict(nfeat=nf, nnorm=len(nrm), nchan=int(sum(len(g) for g in feat)),
                nchan_norm=int(sum(len(g) for g in nrm)),
                dropped_gap=int(np.sum(~usable[need])), masked=int(np.sum((rng | inorm) & usable & fm)),
                norm_overlap=int(np.sum(sel & nsel)), incomplete=float(B[:, :nf].mean()),
                windows=report, local=bool(local))
    return F, V, B, CV, G, is_feat, is_norm, info


def hierarchical_bins(x, y, signal, noise, pixelsize, target, covar=0., sb_tol=0.1, cov_c=1., cov_nmax=None,
                      cov_table=None,
                      compact=0.3, spec_weight=0., spec_f=None, spec_v=None,
                      sb_fwhm=3., min_snr_frac=0.5, max_elong=2.5, max_dpa=0., core_radius=1.0,
                      smooth_passes=3, sb_rms=None, spec_z=0., spec_b=None, spec_feat=None,
                      spec_norm=None, spec_cv=None, spec_group=None, spec_local=False, stats=None):
    """
    Bin spaxels at (x, y) [arcsec] with signal/noise into contiguous regions.
    Returns binNum (0..nbin-1 per spaxel), xBar, yBar, sn, npix per bin.

    covar, cov_c, cov_nmax: noise of an N-spaxel sum = cov_c (1 + covar log10 N) x the
             summed-variance noise, N capped at cov_nmax (noisetools.covariance_factor;
             SPATIAL_BINNING COVARIANCE, COV_C, COV_NMAX); cov_table = (N, ratio) measured
             curve instead (COV_FILE)

    sb_rms   max rms of log SB inside a merged bin [dex]; None -> sb_tol/2, 0 -> off
    spec_z   > 0: refuse merges with chi2_nu > 1 + spec_z sqrt(2/nu)
    spec_f, spec_v, spec_b, spec_cv, spec_feat, spec_norm: output of _spectral_features
             (spec_b/feat/norm may be None: no missing chunks, every chunk is both;
              spec_cv None: the amplitude is the total of the feature chunks;
              spec_local: each window (spec_group) normalised by its own total)
    stats    optional dict, filled with diagnostics (rejected merges per reason, ...)
    """
    cf = lambda n: _noise.covariance_factor(n, covar, cov_c, cov_nmax, cov_table)
    if sb_rms is None:
        sb_rms = 0.5 * sb_tol
    n = len(x)
    ii, jj = _grid(x, y, pixelsize)
    ny, nx = ii.max() + 1, jj.max() + 1
    where = -np.ones((ny, nx), int)
    where[ii, jj] = np.arange(n)

    # lightly smoothed surface brightness, for the morphology term only
    img = np.zeros((ny, nx)); msk = np.zeros((ny, nx))
    img[ii, jj] = signal; msk[ii, jj] = 1.
    if sb_fwhm and sb_fwhm > 0:
        s = sb_fwhm / 2.3548
        sm = ndimage.gaussian_filter(img, s) / np.maximum(ndimage.gaussian_filter(msk, s), 1e-12)
    else:
        sm = img
    sb = sm[ii, jj]
    floor = np.nanpercentile(sb[sb > 0], 1) if np.any(sb > 0) else 1.
    lsb = np.log10(np.maximum(sb, floor))        # faint / negative sky noise: clipped

    use_spec = (spec_weight > 0 or spec_z > 0) and spec_f is not None
    if use_spec:
        nchk = spec_f.shape[1]
        is_feat = np.ones(nchk, bool) if spec_feat is None else np.asarray(spec_feat, bool)
        is_norm = is_feat.copy() if spec_norm is None else np.asarray(spec_norm, bool)
        shared = is_feat & is_norm          # chunk is also part of the amplitude: covariance term
    rej = dict(sb=0, sb_rms=0, elong=0, centroid=0, dpa=0, spec=0)

    # azimuth seen from the centre (GIST's x, y are relative to READ_DATA ORIGIN):
    # a bin may not span more than max_dpa in position angle outside core_radius,
    # so it never wraps round the galaxy and averages the two sides of a disc
    th = np.arctan2(y, x)
    wa = (np.hypot(x, y) > core_radius).astype(float)
    CA, SA, WA = wa * np.cos(th), wa * np.sin(th), wa.copy()
    half = np.deg2rad(max_dpa) / 2. if max_dpa else 0.
    x0, y0 = x.min(), y.min()
    rbar_min = np.sin(half) / half if half > 0 else 1.     # mean resultant length of a uniform arc

    # region state (arrays indexed by region id; a merge keeps the smaller id)
    S = signal.astype(float).copy(); V = noise.astype(float) ** 2
    N = np.ones(n); LS = lsb.copy(); LS2 = lsb ** 2
    X1, Y1 = x.astype(float).copy(), y.astype(float).copy()
    X2, Y2 = X1 ** 2, Y1 ** 2
    F = spec_f.astype(float).copy() if use_spec else None
    FV = spec_v.astype(float).copy() if use_spec else None
    BB = None
    CVR = spec_cv.astype(float).copy() if (use_spec and spec_cv is not None and not spec_local) else None
    if use_spec and spec_local:
        GRP = np.asarray(spec_group, int)
        win_ids = [g for g in np.unique(GRP) if g >= 0]
    if use_spec:   # BB[r, c]: chunk c is incomplete in at least one member of region r
        BB = np.zeros(spec_f.shape, bool) if spec_b is None else np.asarray(spec_b, bool).copy()
    alive = np.ones(n, bool); ver = np.zeros(n, int)
    members = [[k] for k in range(n)]
    nbr = [set() for _ in range(n)]
    # 8-connected adjacency (diagonal neighbours too: fewer isolated islands)
    pad = -np.ones((ny + 2, nx + 2), int); pad[1:-1, 1:-1] = where
    for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = pad[1:-1, 1:-1]; b = pad[1 + di:ny + 1 + di, 1 + dj:nx + 1 + dj]
        ok = (a >= 0) & (b >= 0)
        for p, q in zip(a[ok].tolist(), b[ok].tolist()):
            nbr[p].add(q); nbr[q].add(p)
    owner = np.arange(n)

    def sn(r):
        return S[r] / np.sqrt(V[r]) / cf(N[r]) if V[r] > 0 else 0.

    done = np.array([sn(k) >= target for k in range(n)])

    def spec_chi2(a, b):
        """(chi2_nu, nu) of the two regions' amplitude-normalised chunk spectra; None if not comparable."""
        u = ~(BB[a] | BB[b])                 # chunks complete in every member of both regions
        f = u & is_feat
        nf = int(f.sum())
        if use_spec and spec_local:
            # each feature window normalised by its own total: depth/shape of each feature,
            # insensitive to dust, flux calibration and the slope between features
            ka = cf(N[a]) ** 2
            kb = cf(N[b]) ** 2
            tot, nu = 0., 0
            for g in win_ids:
                fg = f & (GRP == g)
                n = int(fg.sum())
                if n < 2:
                    continue
                Fa, Fb, Va, Vb = F[a][fg], F[b][fg], FV[a][fg], FV[b][fg]
                Aa, Ab = Fa.sum(), Fb.sum()
                if not (Aa > 0 and Ab > 0):
                    continue
                xa, xb = Fa / Aa, Fb / Ab
                # exact first-order covariance of x = F / sum(F) in this window,
                #   Cov = [diag(V) - x V^T - V x^T + sum(V) x x^T] / A^2,
                # singular (sum x = 1): drop one chunk, solve for the other n - 1
                Ca = ka * (np.diag(Va) - np.outer(xa, Va) - np.outer(Va, xa) + Va.sum() * np.outer(xa, xa)) / Aa ** 2
                Cb = kb * (np.diag(Vb) - np.outer(xb, Vb) - np.outer(Vb, xb) + Vb.sum() * np.outer(xb, xb)) / Ab ** 2
                d = (xa - xb)[:-1]
                try:
                    c2 = float(d.dot(np.linalg.solve((Ca + Cb)[:-1, :-1], d)))
                except np.linalg.LinAlgError:
                    continue
                if not (np.isfinite(c2) and c2 >= 0.):
                    continue
                tot += c2
                nu += n - 1                  # sum x = 1 in each window
            if nu < 2:
                return None
            return tot / nu, nu
        if CVR is None:                      # amplitude = total of the common complete feature chunks
            m = f
        else:                                # amplitude windows: all of them must be complete
            m = is_norm
            if not u[is_norm].all():
                return None
        if nf < 3 or not m.any():
            return None
        Aa, Ab = F[a][m].sum(), F[b][m].sum()
        if not (Aa > 0 and Ab > 0):
            return None
        AVa, AVb = FV[a][m].sum(), FV[b][m].sum()
        xa, xb = F[a][f] / Aa, F[b][f] / Ab
        # correlated neighbouring spaxels (drizzled cubes): the summed variance underestimates
        # the true one by the same factor COVARIANCE corrects the S/N for
        ka = cf(N[a]) ** 2
        kb = cf(N[b]) ** 2
        d = xa - xb
        if CVR is None:
            # amplitude = total of the chunks: its error is small and spread over all of
            # them; diagonal first-order variance (calibrated: chi2_nu ~ 1 +- sqrt(2/nu))
            va = ka * (FV[a][f] * (1. - 2. * xa) + xa * xa * AVa) / Aa ** 2
            vb = kb * (FV[b][f] * (1. - 2. * xb) + xb * xb * AVb) / Ab ** 2
            nu = max(nf - 1, 1)                    # sum x = 1 for both: one d.o.f. less
            return float(np.sum(d * d / np.maximum(va + vb, 1e-300))) / nu, nu
        # amplitude windows: one common amplitude error couples all chunks, so use the
        # full first-order covariance of x = F/A for each region,
        #   Cov = [diag(V) - x cv^T - cv x^T + var(A) x x^T] / A^2   (cv = cov(F_c, A)),
        # i.e. diagonal + rank 2, and invert diagonal + rank 4 with the Woodbury identity
        D = ka * FV[a][f] / Aa ** 2 + kb * FV[b][f] / Ab ** 2
        U = np.column_stack([xa, CVR[a][f], xb, CVR[b][f]])
        Minv = np.zeros((4, 4))
        Minv[:2, :2] = Aa ** 2 / ka * np.array([[0., -1.], [-1., -AVa]])
        Minv[2:, 2:] = Ab ** 2 / kb * np.array([[0., -1.], [-1., -AVb]])
        Wd = d / D; WU = U / D[:, None]
        q = U.T.dot(Wd)
        try:
            chi2 = float(d.dot(Wd) - q.dot(np.linalg.solve(Minv + U.T.dot(WU), q)))
        except np.linalg.LinAlgError:
            chi2 = np.nan
        if not (np.isfinite(chi2) and chi2 >= 0.):
            return None
        return chi2 / nf, nf

    def cost(a, b, tol, spec_gate=True):
        d_sb = abs(LS[a] / N[a] - LS[b] / N[b]) / tol
        if d_sb > 1.:
            rej['sb'] += 1
            return None
        nn = N[a] + N[b]
        if sb_rms > 0 and tol < 1e8:
            lim = sb_rms * tol / sb_tol           # relaxed together with the tolerance
            mu = (LS[a] + LS[b]) / nn
            if (LS2[a] + LS2[b]) / nn - mu * mu > lim * lim:
                rej['sb_rms'] += 1
                return None                       # the merged bin would span too wide an SB range
        mx, my = (X1[a] + X1[b]) / nn, (Y1[a] + Y1[b]) / nn
        r2 = (X2[a] + X2[b]) / nn - mx * mx + (Y2[a] + Y2[b]) / nn - my * my
        r_disc2 = nn * pixelsize ** 2 / (2. * np.pi)            # <r^2> of a filled disc
        elong = np.sqrt(max(r2, 0.) / r_disc2)
        if elong > max_elong:
            rej['elong'] += 1
            return None                     # no long strips
        if nn >= 9:
            # the bin must contain its own centroid: no rings or horseshoes round
            # a nucleus (which would average both sides of its rotation)
            ci = int(round((my - y0) / pixelsize)); cj = int(round((mx - x0) / pixelsize))
            blk = where[max(ci - 1, 0):ci + 2, max(cj - 1, 0):cj + 2]
            blk = blk[blk >= 0]
            if blk.size == 0 or not np.any((owner[blk] == a) | (owner[blk] == b)):
                rej['centroid'] += 1
                return None
        w = WA[a] + WA[b]
        if half > 0 and w >= 3:
            rbar = np.hypot(CA[a] + CA[b], SA[a] + SA[b]) / w
            if rbar < rbar_min:
                rej['dpa'] += 1
                return None                 # would span more than max_dpa round the centre
        c = d_sb + compact * max(0., elong - 1.)
        if use_spec:
            r = spec_chi2(a, b)             # None: too few common complete chunks -> no evidence
            if r is not None:
                chi2nu, nu = r
                if spec_gate and spec_z > 0 and chi2nu > 1. + spec_z * np.sqrt(2. / nu):
                    rej['spec'] += 1
                    return None             # spectra differ by more than the noise
                c += spec_weight * max(0., chi2nu - 1.)
        return c

    def merge(keep, gone):
        S[keep] += S[gone]; V[keep] += V[gone]; N[keep] += N[gone]
        LS[keep] += LS[gone]; LS2[keep] += LS2[gone]
        X1[keep] += X1[gone]; Y1[keep] += Y1[gone]; X2[keep] += X2[gone]; Y2[keep] += Y2[gone]
        CA[keep] += CA[gone]; SA[keep] += SA[gone]; WA[keep] += WA[gone]
        if use_spec:
            F[keep] += F[gone]; FV[keep] += FV[gone]; BB[keep] |= BB[gone]
            if CVR is not None:
                CVR[keep] += CVR[gone]
        owner[members[gone]] = keep
        members[keep] += members[gone]; members[gone] = []
        alive[gone] = False

    def run(tol, allow_done_partner=False, min_sn=None, spec_gate=True):
        heap = []
        def push(a):
            for b in nbr[a]:
                if not alive[b]:
                    continue
                if done[a] and done[b]:
                    continue
                if (done[a] or done[b]) and not allow_done_partner:
                    continue
                c = cost(a, b, tol, spec_gate)
                if c is not None:
                    heapq.heappush(heap, (c, a, b, ver[a], ver[b]))
        for a in np.where(alive & ~done)[0]:
            push(a)
        while heap:
            c, a, b, va, vb = heapq.heappop(heap)
            if not (alive[a] and alive[b]) or ver[a] != va or ver[b] != vb:
                continue
            if done[a] and done[b]:
                continue
            if allow_done_partner and min_sn is not None:
                # absorb phase: only regions below min_sn are absorbed
                small = a if not done[a] else b
                if sn(small) >= min_sn:
                    continue
            keep, gone = (a, b) if a < b else (b, a)
            merge(keep, gone)
            nbr[keep] |= nbr[gone]; nbr[keep].discard(keep); nbr[keep].discard(gone)
            for q in nbr[gone]:
                if q != keep:
                    nbr[q].discard(gone); nbr[q].add(keep)
            nbr[gone] = set()
            done[keep] = done[keep] or done[gone] or sn(keep) >= target
            ver[keep] += 1
            if not done[keep] or not allow_done_partner:
                push(keep)

    run(sb_tol)                                        # isophote-following growth
    run(2. * sb_tol)                                   # relaxed, for steep gradients
    run(1e9)                                           # any neighbour, still S/N-limited
    # (the spectral test stays on above: a spectrally distinct region that cannot reach
    #  the target stays a bin of its own if it has >= MIN_SNR_FRAC x target)
    run(1e9, allow_done_partner=True, min_sn=min_snr_frac * target,
        spec_gate=False)                               # absorb leftovers

    # Islands (spaxels cut off by the mask) have no neighbour to merge with: absorb
    # what is still below MIN_SNR_FRAC x target into the spatially nearest region.
    from scipy.spatial import cKDTree
    tree = cKDTree(np.c_[x, y])
    while True:
        roots = np.where(alive)[0]
        if roots.size < 2:
            break
        sns = np.array([sn(r) for r in roots])
        low = roots[sns < min_snr_frac * target]
        if low.size == 0:
            break
        r = low[np.argmin(sns[sns < min_snr_frac * target])]
        mem = np.array(members[r])
        best = None
        for k in (8, 32, 128, n):
            d, j = tree.query(np.c_[x[mem], y[mem]], k=min(k, n))
            d, j = np.atleast_2d(d), np.atleast_2d(j)
            other = owner[j] != r
            if other.any():
                dd = np.where(other, d, np.inf)
                p = np.unravel_index(np.argmin(dd), dd.shape)
                best = owner[j[p]]
                break
        if best is None:
            break
        keep, gone = (r, best) if r < best else (best, r)
        merge(keep, gone)

    roots = np.where(alive)[0]
    binNum = np.empty(n, int)
    for b, r in enumerate(roots):
        binNum[members[r]] = b
    if stats is not None:
        # diagnostics before boundary smoothing: rms of log SB per bin, spectral rejections
        rms = np.array([np.sqrt(max(LS2[r] / N[r] - (LS[r] / N[r]) ** 2, 0.)) for r in roots])
        stats.update(rejected=dict(rej), sb_rms_median=float(np.median(rms)),
                     sb_rms_max=float(rms.max()), sb_rms_limit=float(sb_rms), use_spec=bool(use_spec))
    binNum = _smooth_boundaries(binNum, ii, jj, signal, noise, lsb, target, cf, sb_tol, passes=smooth_passes)
    nb = binNum.max() + 1
    cnt = np.bincount(binNum, minlength=nb).astype(float)
    xBar = np.bincount(binNum, x, nb) / cnt; yBar = np.bincount(binNum, y, nb) / cnt
    ssum = np.bincount(binNum, signal, nb); vsum = np.bincount(binNum, noise ** 2, nb)
    snr = ssum / np.sqrt(vsum) / cf(cnt)
    return binNum, xBar, yBar, snr, cnt.astype(int)


def _smooth_boundaries(b, ii, jj, signal, noise, lsb, target, cf, sb_tol, passes=3):
    """
    Majority filter on the bin map: a spaxel whose 8 neighbours are mostly (>= 5)
    in one other bin moves there, if its surface brightness fits that bin and the
    bin it leaves stays above the target S/N (or does not drop by more than 5%).
    Removes the ragged edges that merging in noisy data leaves. A move that would
    split the bin it leaves into two pieces is not made.
    """
    b = b.copy()
    ny, nx = ii.max() + 1, jj.max() + 1
    nb = b.max() + 1
    S = np.bincount(b, signal, nb); V = np.bincount(b, noise ** 2, nb)
    C = np.bincount(b, minlength=nb).astype(float); L = np.bincount(b, lsb, nb)
    sn = lambda k, s, v, c: s / np.sqrt(v) / cf(max(c, 1.)) if v > 0 and c > 0 else 0.
    for _ in range(passes):
        im = -np.ones((ny + 2, nx + 2), int); im[ii + 1, jj + 1] = b
        moved = 0
        for p in range(len(b)):
            nb8 = im[ii[p]:ii[p] + 3, jj[p]:jj[p] + 3].ravel()
            nb8 = nb8[(nb8 >= 0)]
            vals, cts = np.unique(nb8, return_counts=True)
            own = b[p]
            k = np.argmax(np.where(vals == own, -1, cts))
            new, c_new = vals[k], cts[k]
            if new == own or c_new < 5 or C[own] <= 1:
                continue
            if abs(lsb[p] - L[new] / C[new]) > sb_tol:
                continue
            # do not cut the bin it leaves in two: its other spaxels around p must stay
            # 8-connected without p (local test, conservative)
            ring = im[ii[p]:ii[p] + 3, jj[p]:jj[p] + 3] == own
            ring[1, 1] = False
            if ring.any() and ndimage.label(ring, structure=np.ones((3, 3), int))[1] > 1:
                continue
            before = sn(own, S[own], V[own], C[own])
            after = sn(own, S[own] - signal[p], V[own] - noise[p] ** 2, C[own] - 1)
            if after < target and after < 0.95 * before:
                continue
            for arr, val in ((S, signal[p]), (V, noise[p] ** 2), (C, 1.), (L, lsb[p])):
                arr[own] -= val; arr[new] += val
            b[p] = new; im[ii[p] + 1, jj[p] + 1] = new
            moved += 1
        if moved == 0:
            break
    # relabel 0..n-1 (a bin may have emptied)
    u, inv = np.unique(b, return_inverse=True)
    return inv


def generateSpatialBins(config, cube):
    """GIST interface: same inputs and outputs as voronoi.generateSpatialBins."""
    printStatus.running("Defining the hierarchical bins")
    logging.info("Defining the hierarchical (morphology-aware) bins")

    maskfile = os.path.join(config['GENERAL']['OUTPUT'], config['GENERAL']['RUN_ID']) + "_mask.fits"
    mask = fits.open(maskfile)[1].data.MASK
    idxUnmasked = np.where(mask == 0)[0]
    idxMasked = np.where(mask == 1)[0]
    if idxUnmasked.size == 0:
        printStatus.updateFailed("Defining the hierarchical bins")
        logging.error("No unmasked spaxels to bin.")
        return "SKIP"

    target = float(config['SPATIAL_BINNING']['TARGET_SNR'])
    covar, cov_c, cov_nmax, cov_table = _noise.covariance_options(config['SPATIAL_BINNING'],
                                                                  config['GENERAL']['CONFIG_DIR'])
    sb_tol = float(_opt(config, 'SB_TOL', 0.10))
    compact = float(_opt(config, 'COMPACT', 0.3))
    max_elong = float(_opt(config, 'MAX_ELONG', 2.5))
    max_dpa = float(_opt(config, 'MAX_DPA', 0.))
    core_radius = float(_opt(config, 'CORE_RADIUS', 1.0))
    spec_weight = float(_opt(config, 'SPEC_WEIGHT', 0.))
    spec_z = float(_opt(config, 'SPEC_Z', 0.))
    sb_rms = _opt(config, 'SB_RMS', None)
    sb_rms = 0.5 * sb_tol if sb_rms is None else float(sb_rms)
    sb_fwhm = float(_opt(config, 'SB_FWHM', 3.))
    min_frac = float(_opt(config, 'MIN_SNR_FRAC', 0.5))
    spec_f = spec_v = spec_b = spec_cv = spec_feat = spec_norm = spec_group = None
    spec_local = False
    if spec_weight > 0 or spec_z > 0:
        z = float(config['GENERAL'].get('REDSHIFT', 0.) or 0.)
        cdir = config['GENERAL'].get('CONFIG_DIR', '.')
        wave = np.asarray(cube['wave'])
        # SPEC_MASK: default = the KIN mask (emission lines, sky, telluric, gap); False = no mask
        smask = _opt(config, 'SPEC_MASK', None)
        if smask is None and 'KIN' in config:
            smask = config['KIN'].get('SPEC_MASK', None)
        fmask = None
        if smask not in (None, False) and str(smask).lower() not in ('none', 'false', ''):
            fmask = _mask_file(os.path.join(cdir, str(smask)), wave, z)
        # windows compared: SPEC_FEATURES (named stellar features) and/or SPEC_WINDOWS
        feats = _features(_opt(config, 'SPEC_FEATURES', None))
        wins = _windows(_opt(config, 'SPEC_WINDOWS', None))
        windows = (feats or []) + [('%g-%g' % w,) + tuple(w) for w in (wins or [])] or None
        # SPEC_NORM: 'local' = each window by its own total (default with SPEC_FEATURES);
        # unset = the S/N window (READ_DATA LMIN_SNR-LMAX_SNR) with READ_DATA SNR_MASK, as
        # for the S/N itself; 'total' = total of all the chunks; or explicit windows
        sn_raw = _opt(config, 'SPEC_NORM', None)
        nmask = None
        spec_local = (str(sn_raw).lower() == 'local') or (sn_raw is None and feats is not None)
        if spec_local and not windows:
            raise ValueError("SPEC_NORM = local needs SPEC_FEATURES or SPEC_WINDOWS")
        if spec_local or (sn_raw is not None and str(sn_raw).lower() == 'total'):
            norm = None
        else:
            norm = _windows(sn_raw)
            if norm is None:
                norm = [(float(config['READ_DATA']['LMIN_SNR']), float(config['READ_DATA']['LMAX_SNR']))]
            snrm = config['READ_DATA'].get('SNR_MASK', None)
            if snrm not in (None, False) and str(snrm).lower() not in ('none', 'false', ''):
                nmask = _mask_file(os.path.join(cdir, str(snrm)), wave, z)
        maxmask = float(_opt(config, 'SPEC_MAXMASK', 0.5))
        spec_f, spec_v, spec_b, spec_cv, spec_group, spec_feat, spec_norm, info = _spectral_features(
            cube, idxUnmasked, _opt(config, 'SPEC_LMIN', None), _opt(config, 'SPEC_LMAX', None),
            nchunk=int(_opt(config, 'SPEC_NCHUNK', 48)), windows=windows,
            dlam=float(_opt(config, 'SPEC_DLAM', 10.)), norm=norm, fmask=fmask, nmask=nmask,
            maxmask=maxmask, local=spec_local)
        if info['windows']:
            for nm, lo, hi, frac, ok in info['windows']:
                line = ("spectral term window %-10s %6.0f-%6.0f A: %3.0f%% masked or without data -> %s"
                        % (nm, lo, hi, 100. * frac, 'used' if ok else 'LEFT OUT (SPEC_MAXMASK %g)' % maxmask))
                logging.info(line)
                print("             " + line)
        msg = ("spectral term: %d feature chunks (%d channels; %d channels masked by %s); amplitude: %s; "
               "%d channels dropped as missing in most spaxels; %.1f%% of spaxel-chunks incomplete"
               % (info['nfeat'], info['nchan'], info['masked'], smask if fmask is not None else 'no SPEC_MASK',
                  'each window by its own total (local)' if spec_local else
                  'total of the chunks' if norm is None else '%s (%d channels%s)' % (
                      ','.join('%g-%g' % w for w in norm), info['nchan_norm'],
                      ', lines removed with SNR_MASK' if nmask is not None else ''),
                  info['dropped_gap'], 100. * info['incomplete']))
        logging.info(msg)
        print("             " + msg)

    sig, noi = cube['signal'][idxUnmasked], cube['noise'][idxUnmasked]
    tot = np.nansum(sig) / np.sqrt(np.nansum(noi ** 2))
    if tot < target:
        logging.warning('the S/N of all %d unmasked spaxels together is %.1f, below TARGET_SNR = %.1f: '
                        'everything ends in one or a few bins. Check the cube coverage and the mask.'
                        % (idxUnmasked.size, tot, target))
    stats = {}
    binNum, xBar, yBar, sn, npix = hierarchical_bins(
        cube['x'][idxUnmasked], cube['y'][idxUnmasked], sig, noi, cube['pixelsize'], target,
        covar=covar, cov_c=cov_c, cov_nmax=cov_nmax, cov_table=cov_table, sb_tol=sb_tol, compact=compact, spec_weight=spec_weight, spec_f=spec_f,
        spec_v=spec_v, sb_fwhm=sb_fwhm, min_snr_frac=min_frac, max_elong=max_elong,
        max_dpa=max_dpa, core_radius=core_radius, sb_rms=sb_rms, spec_z=spec_z, spec_b=spec_b,
        spec_feat=spec_feat, spec_norm=spec_norm, spec_cv=spec_cv, spec_group=spec_group,
        spec_local=spec_local, stats=stats)

    nb = binNum.max() + 1
    low = int(np.sum(sn < target))
    printStatus.updateDone("Defining the hierarchical bins")
    print("             %d hierarchical bins generated (%d below the target S/N)" % (nb, low))
    rj = stats['rejected']
    print("             rms of log SB within bins: median %.3f, max %.3f dex (SB_RMS %.3f); "
          "merges refused by the spectral test: %d" % (stats['sb_rms_median'], stats['sb_rms_max'],
                                                        sb_rms, rj['spec']))
    logging.info("%d hierarchical bins generated; median S/N %.1f, %d below the target; "
                 "SB_TOL=%.3f dex, SB_RMS=%.3f dex, COMPACT=%.2f, SPEC_WEIGHT=%.2f, SPEC_Z=%.2f"
                 % (nb, np.median(sn), low, sb_tol, sb_rms, compact, spec_weight, spec_z))
    logging.info("within-bin rms of log SB: median %.3f, max %.3f dex; merge evaluations refused: %s"
                 % (stats['sb_rms_median'], stats['sb_rms_max'],
                    ', '.join('%s %d' % kv for kv in sorted(rj.items()))))

    binNum_outside = find_nearest_voronoibin(cube['x'], cube['y'], idxMasked, xBar, yBar) if idxMasked.size else []
    binNum_long = np.zeros(len(cube['x']), dtype=int)
    binNum_long[idxUnmasked] = binNum
    if idxMasked.size:
        binNum_long[idxMasked] = -1 * np.asarray(binNum_outside)
    save_table(config, cube['x'], cube['y'], cube['signal'], cube['snr'], binNum_long, np.arange(nb),
               xBar, yBar, sn, npix, cube['pixelsize'])
    return None
