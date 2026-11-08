from   astropy.io          import fits
import numpy               as np

import os
import logging
from astropy.convolution import Gaussian2DKernel, convolve

from printStatus import printStatus

from gistPipeline.readData    import der_snr as der_snr


# --------------------------------------------------------------------------
# Read-in for KCWI blue + red cube pairs (<INPUT>_icubes_wcs.fits and
# <INPUT>-red_icubes_wcs.fits, with matching _vcubes / _mcubes).
#
# Optional READ_DATA columns SUFFIX_BLUE / SUFFIX_RED (None = no suffix) select
# other versions of the stacks, e.g. after kcwi_skylevel:
#     blue = <INPUT><SUFFIX_BLUE>          IRAS03582+6012_skl_icubes_wcs.fits
#     red  = <INPUT><SUFFIX_RED>-red       IRAS03582+6012-red_icubes_wcs.fits
# A MasterConfig without these columns reads <INPUT> and <INPUT>-red as before.
#
# What changed relative to the previous version, and why:
#
#  1. ONE UNIFORM WAVELENGTH GRID. The blue and red spectra used to be
#     concatenated with the dichroic gap left out. GIST log-rebins assuming
#     the wavelength array is uniform, so every pixel redward of the gap was
#     given the wrong wavelength: +280 / -425 A either side of the gap for
#     IRAS03582+6012 (a 14.7% stretch), +61 / -107 A for SDSSJ1027+1749.
#     Emission lines in the fitted spectra sat exactly where that predicts,
#     and the J1027 KIN velocities carried the predicted +1200 km/s offset.
#     The channels are now placed on a single uniform grid. The gap is filled
#     by linear interpolation only so that log_rebin sees finite values; it is
#     given an effectively infinite variance and must ALSO be listed in the
#     SPEC_MASK files (the range is printed in the log).
#
#  2. THE MASK CUBES ARE USED. Masked voxels are stored as flux 0, variance 0
#     (3% of the blue and 19% of the red voxels in IRAS03582's footprint) and
#     were read as genuine zero-flux measurements. Spatial bins sum spaxels,
#     so a spaxel masked in one channel and not the other made the bin's
#     spectrum jump at the dichroic; zero variance also made pPXF reject a KIN
#     bin (noise must be > 0), which the bare except turned into a silent NaN.
#     Now: each channel is trimmed to the planes where most of the footprint
#     is unmasked (the DRP's WAVGOOD1 included fully-masked red planes above
#     ~9190 A); short masked runs inside a spaxel are interpolated with an
#     inflated variance; long runs get an effectively infinite variance; and a
#     spaxel unusable in either channel is zeroed so spatialMasking drops it.
#
#  3. THE CUBES MUST DESCRIBE THE SAME GRID. kcwi_stack writes _vcubes /
#     _mcubes without the _wcs suffix, so re-running a stack can silently pair
#     one run's flux with another run's variance and mask: their shape and
#     wavelength axis are compared (not CRPIX/CRVAL, which kcwi_astrometry
#     changes in the _wcs file only). Blue and red must put each spaxel at the
#     same sky position; that is compared through the WCS.
# --------------------------------------------------------------------------

BIG_VAR_FACTOR = 1e12     # "infinite" variance, relative to the median good variance
MAX_INTERP_RUN = 6        # masked runs up to this many pixels are interpolated
INTERP_VAR_FACTOR = 4.    # variance inflation for an interpolated voxel
MAX_MASKED_FRAC = 0.2     # a spaxel more masked than this in a channel is unusable
PLANE_GOOD_FRAC = 0.5     # keep planes where >= this fraction (of the best plane) is unmasked
SKY_GRID_TOL = 0.05       # [pixels] blue and red WCS must agree to this


# ======================================
# Routine to set DEBUG mode
# ======================================
def set_debug(cube, xext, yext):
    logging.info("DEBUG mode is activated. Instead of the entire cube, only one line of spaxels is used.")
    cube['x']      = cube['x'     ][  int(yext/2)*xext:(int(yext/2)+1)*xext]
    cube['y']      = cube['y'     ][  int(yext/2)*xext:(int(yext/2)+1)*xext]
    cube['snr']    = cube['snr'   ][  int(yext/2)*xext:(int(yext/2)+1)*xext]
    cube['signal'] = cube['signal'][  int(yext/2)*xext:(int(yext/2)+1)*xext]
    cube['noise']  = cube['noise' ][  int(yext/2)*xext:(int(yext/2)+1)*xext]

    cube['spec']   = cube['spec'  ][:,int(yext/2)*xext:(int(yext/2)+1)*xext]
    cube['error']  = cube['error' ][:,int(yext/2)*xext:(int(yext/2)+1)*xext]

    return(cube)


def _wave(h):
    return h['CRVAL3'] + (np.arange(h['NAXIS3']) + 1 - h['CRPIX3']) * h['CD3_3']


def _check_same_grid(h_ref, h, label):
    """
    Stop if a variance / mask cube does not have the flux cube's shape and
    wavelength axis. Their SPATIAL WCS keywords may legitimately differ:
    kcwi_astrometry rewrites CRPIX/CRVAL of the _icubes_wcs file only, so
    those are only reported.
    """
    for k in ('NAXIS1', 'NAXIS2', 'NAXIS3', 'CRPIX3'):
        if h_ref.get(k) != h.get(k):
            raise ValueError('%s: %s differs (%s vs %s)' % (label, k, h_ref.get(k), h.get(k)))
    for k in ('CRVAL3', 'CD3_3'):
        if not np.isclose(float(h_ref[k]), float(h[k]), rtol=0, atol=1e-9):
            raise ValueError('%s: %s differs (%r vs %r) -- the variance/mask cube is from a '
                             'different stack than the flux cube' % (label, k, h_ref[k], h[k]))
    if any(not np.isclose(float(h_ref.get(k, 0)), float(h.get(k, 0)), rtol=0, atol=1e-9)
           for k in ('CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2')):
        logging.info('%s: spatial WCS keywords differ from the flux cube (expected after '
                     'kcwi_astrometry); the pixel grids are assumed identical' % label)


def _sky_offset_pix(hb, hr):
    """Largest distance [pixels] between where the blue and red WCS put the same pixel."""
    from astropy import wcs as awcs
    nx, ny = hb['NAXIS1'], hb['NAXIS2']
    p = np.array([[0, 0], [nx - 1, 0], [0, ny - 1], [nx - 1, ny - 1], [(nx - 1) / 2., (ny - 1) / 2.]], float)
    wb, wr = awcs.WCS(hb).celestial, awcs.WCS(hr).celestial
    if wb.naxis == 2 and wr.naxis == 2:
        return float(np.abs(wr.all_world2pix(wb.all_pix2world(p, 0), 0) - p).max())
    # no celestial CTYPEs: linear CRVAL + CD (p + 1 - CRPIX)
    def cd(h):
        return np.array([[h.get('CD1_1', 0.), h.get('CD1_2', 0.)], [h.get('CD2_1', 0.), h.get('CD2_2', 0.)]])
    world = (cd(hb) @ (p + 1 - [hb['CRPIX1'], hb['CRPIX2']]).T).T + [hb['CRVAL1'], hb['CRVAL2']]
    pr = (np.linalg.solve(cd(hr), (world - [hr['CRVAL1'], hr['CRVAL2']]).T)).T - 1 + [hr['CRPIX1'], hr['CRPIX2']]
    return float(np.abs(pr - p).max())


def _read_channel(base, label, pad=10.):
    """Flux, variance and bad-voxel mask of one channel, trimmed to usable planes.

    Returns wave (nw,), flux (nw, ny, nx), var (nw, ny, nx), bad (nw, ny, nx).
    """
    hi = fits.open(base + '_icubes_wcs.fits')
    hv = fits.open(base + '_vcubes.fits')
    hm = fits.open(base + '_mcubes.fits')
    h = hi[0].header
    _check_same_grid(h, hv[0].header, label + ' variance cube')
    _check_same_grid(h, hm[0].header, label + ' mask cube')
    wave = _wave(h)
    inside = (wave > h['WAVGOOD0'] + pad) & (wave < h['WAVGOOD1'] - pad)
    flux = hi[0].data; var = hv[0].data; msk = hm[0].data
    bad = (msk != 0) | ~np.isfinite(flux) | ~np.isfinite(var) | (var <= 0)
    footprint = (~bad[inside]).any(axis=0)
    good_frac = (~bad[:, footprint]).mean(axis=1)
    keep = inside & (good_frac >= PLANE_GOOD_FRAC * good_frac[inside].max())
    k = np.where(keep)[0]
    if k.size == 0:
        raise ValueError('%s: no usable wavelength planes' % label)
    k0, k1 = k[0], k[-1] + 1          # contiguous range: interior dips are handled per voxel
    logging.info('%s: WAVGOOD %.1f-%.1f A, kept %.1f-%.1f A (planes where >= %.0f%% of the '
                 'footprint is unmasked)' % (label, h['WAVGOOD0'], h['WAVGOOD1'], wave[k0],
                                              wave[k1 - 1], 100 * PLANE_GOOD_FRAC))
    return (h, wave[k0:k1], np.array(flux[k0:k1], dtype=float),
            np.array(var[k0:k1], dtype=float), bad[k0:k1])


def _repair_masked(flux, var, bad, big_var):
    """Fill masked voxels along wavelength, spaxel by spaxel (arrays are (nw, nspax)).

    Runs of up to MAX_INTERP_RUN pixels are linearly interpolated and given an
    inflated variance; longer runs are interpolated too (so log_rebin stays
    finite) but get big_var. Returns the per-spaxel masked fraction.
    """
    nw, ns = flux.shape
    frac = bad.mean(axis=0)
    for j in np.where(bad.any(axis=0) & (frac < 1))[0]:
        b = bad[:, j]; g = ~b; idx = np.arange(nw)
        flux[b, j] = np.interp(idx[b], idx[g], flux[g, j])
        vint = np.interp(idx[b], idx[g], var[g, j]) * INTERP_VAR_FACTOR
        # run lengths of the masked stretches
        edges = np.diff(np.r_[0, b.astype(int), 0]); st = np.where(edges == 1)[0]; en = np.where(edges == -1)[0]
        longrun = np.zeros(nw, bool)
        for s, e in zip(st, en):
            if e - s > MAX_INTERP_RUN:
                longrun[s:e] = True
        var[b, j] = np.where(longrun[b], big_var, vint)
    return frac


# ======================================
# Routine to load KCWI red and blue cubes
# ======================================
def _snr_mask(fn, wave, z):
    """
    True where a specMask-format file (centre, width, comment; rest frame, except
    comment 'sky' = observed centre, as in GIST's spectralMasking) masks wave.
    Used to keep emission lines, sky lines, telluric bands etc. out of the S/N.
    """
    rows = [l.split('#')[0].split() for l in open(fn)]
    rows = [r for r in rows if len(r) >= 2]
    cut = np.zeros(wave.size, bool)
    for r in rows:
        c, w = float(r[0]), float(r[1])
        if len(r) > 2 and r[2].lower() == 'sky':
            c = c / (1 + z)
        cut |= (wave >= c - w / 2.) & (wave <= c + w / 2.)
    return cut


def _galactic_ebv(config):
    """READ_DATA EBV as a number (0 when unset / None / False)."""
    v = config['READ_DATA'].get('EBV', None)
    if v is None or v is False or str(v).strip().lower() in ('', 'none', 'false', '0'):
        return 0.
    v = float(v)
    if not (0 <= v < 5):
        raise ValueError('READ_DATA EBV = %g: not a Galactic E(B-V)' % v)
    if v > 0 and config.get('GAS', {}).get('EBmV', None) not in (None, False, 0, 0.):
        raise ValueError('READ_DATA EBV (%g) and GAS EBmV (%s) are both set: the GAS spectra would be '
                         'de-reddened twice. Set GAS EBmV = None.' % (v, config['GAS']['EBmV']))
    return v


def _suffix(v):
    return '' if v is None or v is False or str(v).strip() in ('', 'None', 'none', 'False', 'false', '-') else str(v).strip()


def channel_bases(config):
    """Base names (without _icubes_wcs.fits) of the blue and red stacks."""
    rd = config.get('READ_DATA', {})
    inp = config['GENERAL']['INPUT']
    return inp + _suffix(rd.get('SUFFIX_BLUE')), inp + _suffix(rd.get('SUFFIX_RED')) + '-red'


def readCube(config):

    loggingBlanks = (len( os.path.splitext(os.path.basename(__file__))[0] ) + 33) * " "

    printStatus.running("Reading the KECK-KCWI cube")
    base_b, base_r = channel_bases(config)
    logging.info("Reading the KECK-KCWI cubes: blue " + base_b + "_icubes_wcs.fits, red " + base_r + "_icubes_wcs.fits")
    print(loggingBlanks + "blue: " + os.path.basename(base_b) + "_icubes_wcs.fits, red: " + os.path.basename(base_r) + "_icubes_wcs.fits")

    hdrb, wb, fb, vb, bb = _read_channel(base_b, 'blue')
    hdrr, wr, fr, vr, br = _read_channel(base_r, 'red')
    # GIST joins the channels spaxel by spaxel, so both must be on one pixel grid ON THE
    # SKY (compared through the WCS: the same grid can be written with different
    # CRPIX/CRVAL pairs, e.g. by kcwi_astrometry). Identical headers do not prove the
    # CONTENT is registered -- check that with `kcwi_xalign wcs`.
    if (hdrb['NAXIS1'], hdrb['NAXIS2']) != (hdrr['NAXIS1'], hdrr['NAXIS2']):
        raise ValueError('blue and red cubes have different spatial sizes')
    off = _sky_offset_pix(hdrb, hdrr)
    if off > SKY_GRID_TOL:
        raise ValueError('the blue and red WCS put the same spaxel %.2f pixels apart on the sky: the '
                         'cubes are not on one pixel grid, and joining them spaxel by spaxel would mix '
                         'different positions. Stack both on the same grid (same stack_ad / dimension '
                         '/ xpix / ypix / orientation), e.g. with kcwi_xalign shift lists.' % off)
    logging.info('blue and red WCS agree to %.3f pixels' % off)

    # resolve an overlap, if any: blue up to the middle of it, red from there
    if wb[-1] >= wr[0]:
        cut = 0.5 * (wr[0] + wb[-1])
        kb = wb < cut; kr = wr >= cut
        wb, fb, vb, bb = wb[kb], fb[kb], vb[kb], bb[kb]
        wr, fr, vr, br = wr[kr], fr[kr], vr[kr], br[kr]
        logging.info('blue and red overlap; spliced at %.1f A' % cut)

    ny, nx = fb.shape[1:]
    s = (None, ny, nx)
    fb = fb.reshape(len(wb), -1); vb = vb.reshape(len(wb), -1); bb = bb.reshape(len(wb), -1)
    fr = fr.reshape(len(wr), -1); vr = vr.reshape(len(wr), -1); br = br.reshape(len(wr), -1)

    good_var = np.concatenate([vb[~bb], vr[~br]])
    big_var = BIG_VAR_FACTOR * np.median(good_var[good_var > 0])
    mfb = _repair_masked(fb, vb, bb, big_var)
    mfr = _repair_masked(fr, vr, br, big_var)

    # one uniform grid from blue start to red end (both channels sample 0.5 A on
    # grids that line up; otherwise red is resampled onto the blue grid)
    dlam = float(np.median(np.diff(wb)))
    wave = wb[0] + dlam * np.arange(int(round((wr[-1] - wb[0]) / dlam)) + 1)
    ib = np.rint((wb - wave[0]) / dlam).astype(int)
    ir = np.rint((wr - wave[0]) / dlam).astype(int)
    if not np.allclose(wave[ir], wr, atol=1e-3 * dlam) or not np.allclose(np.diff(wr), dlam, rtol=1e-6):
        logging.warning('red grid does not line up with the blue one: resampling red onto it')
        ir = np.where((wave >= wr[0]) & (wave <= wr[-1]) & (np.arange(wave.size) > ib[-1]))[0]
        fr = np.array([np.interp(wave[ir], wr, fr[:, j]) for j in range(fr.shape[1])]).T
        vr = np.array([np.interp(wave[ir], wr, vr[:, j]) for j in range(vr.shape[1])]).T
    nspax = fb.shape[1]
    spec = np.empty((wave.size, nspax)); espec = np.empty((wave.size, nspax))
    spec[ib], espec[ib] = fb, vb
    spec[ir], espec[ir] = fr, vr
    gap = np.ones(wave.size, bool); gap[ib] = False; gap[ir] = False
    if gap.any():
        g0, g1 = ib[-1], ir[0]                         # last blue, first red pixel
        t = (np.arange(g0 + 1, g1) - g0) / float(g1 - g0)
        spec[g0 + 1:g1] = spec[g0] + t[:, None] * (spec[g1] - spec[g0])
        espec[g0 + 1:g1] = big_var
    z = config['GENERAL']['REDSHIFT']
    msg = ('dichroic gap %.1f-%.1f A observed = %.1f-%.1f A rest: it carries no data. Add it '
           'to SPEC_MASK for KIN, GAS and SFH, e.g. "%.1f %.1f gap"' % (
               wave[gap][0], wave[gap][-1], wave[gap][0] / (1 + z), wave[gap][-1] / (1 + z),
               0.5 * (wave[gap][0] + wave[gap][-1]) / (1 + z), (wave[gap][-1] - wave[gap][0]) / (1 + z) + 20.)) \
        if gap.any() else 'no gap between the channels'
    logging.info(msg); printStatus.running(msg)

    # a spaxel too heavily masked in either channel cannot give a spectrum that
    # is continuous across the dichroic: zero it, so spatialMasking drops it
    unusable = (mfb > MAX_MASKED_FRAC) | (mfr > MAX_MASKED_FRAC)
    footprint = (mfb < 1) | (mfr < 1)
    fp2 = footprint.reshape(ny, nx)
    edge = int(fp2[0].sum() + fp2[-1].sum() + fp2[:, 0].sum() + fp2[:, -1].sum())
    if footprint.mean() < 0.15:
        msg = ('only %.1f%% of the cube (%d of %d spaxels) has data, and %d of its edge spaxels do: the '
               'field may be cut off by the stack grid. If the frames should fill it, stack_ad in the .par '
               'and the absolute offset in the shift lists disagree (kcwi_xalign --ref or --astrom).'
               % (100 * footprint.mean(), int(footprint.sum()), footprint.size, edge))
        logging.warning(msg); printStatus.warning(msg)
    spec[:, unusable] = 0.
    espec[:, unusable] = big_var                    # finite SNR (= 0) for them
    logging.info('%d of %d footprint spaxels dropped: more than %.0f%% masked in one channel' % (
        int((unusable & footprint).sum()), int(footprint.sum()), 100 * MAX_MASKED_FRAC))

    #smooth=True
    if False:#config['READ_DATA']['RSMOOTH'] > 0:
        pass  # spatial smoothing experiment, kept disabled as before

    # Getting the spatial coordinates
    origin = [ float(config['READ_DATA']['ORIGIN'].split(',')[0].strip()), float(config['READ_DATA']['ORIGIN'].split(',')[1].strip()) ]
    pixelsize = hdrb['CD2_2']*3600
    xaxis = (np.arange(nx) - origin[0]) * pixelsize
    yaxis = (np.arange(ny) - origin[1]) * pixelsize
    x, y  = np.meshgrid(xaxis,yaxis)
    x     = np.reshape(x,[ny*nx])
    y     = np.reshape(y,[ny*nx])
    logging.info("Extracting spatial information:\n"\
            +loggingBlanks+"* Spatial coordinates are centred to "+str(origin)+"\n"\
            +loggingBlanks+"* Spatial pixelsize is "+str(pixelsize))

    # Galactic foreground extinction (READ_DATA EBV, E(B-V) [mag]; gist_prep fills it from
    # the SFD maps x 0.86): de-redden at the OBSERVED wavelengths with the Fitzpatrick
    # (1999) R_V = 3.1 law (READ_DATA EXT_LAW: f99 default, ccm89, od94), before anything else sees the spectra -- every module (KIN,
    # SFH and its gas lines, GAS, LS) then works on corrected spectra. Do not also give
    # GAS EBmV (GandALF would correct a second time).
    ebv = _galactic_ebv(config)
    if ebv:
        from gistPipeline.auxiliary import galactic as _galactic
        law = config['READ_DATA'].get('EXT_LAW', None) or 'f99'
        fac = _galactic.deredden_factor(wave, ebv, law=law)
        spec = spec * fac[:, None]
        espec = espec * (fac ** 2)[:, None]              # variances
        msg = ("Galactic extinction corrected: E(B-V) = %.4f (%s, R_V = 3.1): x%.2f at %.0f A to x%.2f at %.0f A "
               "observed" % (ebv, law, fac[0], wave[0], fac[-1], wave[-1]))
        logging.info(msg); printStatus.running(msg)
    else:
        logging.info("No Galactic extinction correction (READ_DATA EBV not set)")

    # De-redshift spectra
    wave = wave / (1+z)
    logging.info("Shifting spectra to rest-frame, assuming a redshift of "+str(z))

    # Shorten spectra to required wavelength range
    lmin  = config['READ_DATA']['LMIN_TOT']
    lmax  = config['READ_DATA']['LMAX_TOT']
    idx   = np.where( np.logical_and( wave >= lmin, wave <= lmax ) )[0]
    spec  = spec[idx,:]
    espec = espec[idx,:]
    wave  = wave[idx]
    gap   = gap[idx]
    logging.info("Shortening spectra to the wavelength range from "+str(lmin)+"A to "+str(lmax)+"A.")

    # Error spectra are variances (the _vcubes), as GIST expects

    # Computing the SNR per spaxel, on real pixels only
    use = np.logical_and( wave >= config['READ_DATA']['LMIN_SNR'], wave <= config['READ_DATA']['LMAX_SNR'] ) & ~gap
    if not use.any():
        raise ValueError('the SNR window LMIN_SNR-LMAX_SNR falls entirely in the dichroic gap')
    # Optional READ_DATA SNR_MASK: a specMask-format file in CONFIG_DIR whose intervals
    # (emission lines, sky, telluric) are left out of the S/N, so that line emission
    # does not count as continuum signal
    snr_mask = config['READ_DATA'].get('SNR_MASK', None)
    if snr_mask:
        n0 = int(use.sum())
        use &= ~_snr_mask(os.path.join(config['GENERAL']['CONFIG_DIR'], str(snr_mask)), wave, z)
        n1 = int(use.sum())
        msg = ("SNR window %g-%g A: %d of %d pixels left after SNR_MASK %s (%.0f%%)"
               % (config['READ_DATA']['LMIN_SNR'], config['READ_DATA']['LMAX_SNR'], n1, n0, snr_mask, 100. * n1 / n0))
        logging.info(msg)
        if n1 < 20:
            raise ValueError(msg + ': too few to measure the S/N; widen LMIN_SNR-LMAX_SNR or choose a cleaner window')
        if n1 < 0.5 * n0:
            printStatus.warning(msg + ': consider a cleaner window (gist_prep.py suggests some)')
            logging.warning(msg + ': consider a cleaner window')
    idx_snr = np.where(use)[0]
    signal = np.nanmedian(spec[idx_snr,:],axis=0)
    noise  = np.abs(np.nanmedian(np.sqrt(espec[idx_snr,:]),axis=0))
    snr    = signal / noise
    logging.info("Computing the signal-to-noise ratio in the wavelength range from "+str(config['READ_DATA']['LMIN_SNR'])+"A to "+str(config['READ_DATA']['LMAX_SNR'])+"A.")

    # Storing everything into a structure
    cube = {'x':x, 'y':y, 'wave':wave, 'spec':spec, 'error':espec, 'snr':snr, 'signal':signal, 'noise':noise, 'pixelsize':pixelsize}

    # Constrain cube to one central row if switch DEBUG is set
    if config['READ_DATA']['DEBUG'] == True: cube = set_debug(cube, nx, ny)

    printStatus.updateDone("Reading the KECK-KCWI cube")
    print("             Read "+str(len(cube['x']))+" spectra!")
    logging.info("Finished reading the KECK cube! Read a total of "+str(len(cube['x']))+" spectra!")

    return(cube)
