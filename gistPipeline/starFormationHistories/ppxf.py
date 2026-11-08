import numpy    as np
from astropy.io import fits
from multiprocessing import Queue, Process

import copy
import time
import os
import glob
import logging

from printStatus import printStatus 

from gistPipeline.prepareTemplates import _prepareTemplates
from gistPipeline.auxiliary import _auxiliary
from gistPipeline.auxiliary import noisetools as _noise
from gistPipeline.auxiliary import gaslines as _gaslines
from scipy.special import erf
from scipy.interpolate import interp1d

# Rescale the noise so that the unregularised fit has chi2/DOF = 1 before the
# regularised fit (set from SFH|NOISE_RESCALE, default True; see run_ppxf)
RESCALE_NOISE = True

# Then use system installed version instead
from ppxf.ppxf      import ppxf
try:
    from ppxf.ppxf  import attenuation as _attenuation     # pPXF's Calzetti (2000) curve
except ImportError:                                        # older pPXF: GAS_REDDENING unavailable
    _attenuation = None

# Physical constants
C = 299792.458 # speed of light in km/s


"""
PURPOSE: 
  This module performs the extraction of non-parametric star-formation histories
  by full-spectral fitting.  Basically, it acts as an interface between pipeline
  and the pPXF routine from Cappellari & Emsellem 2004
  (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
  ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C). 
"""


def workerPPXF(inQueue, outQueue):
    """
    Defines the worker process of the parallelisation with multiprocessing.Queue
    and multiprocessing.Process. 
    """
    for templates, galaxy, noise, velscale, start, goodPixels_sfh, mom, dv,\
        mdeg, regul_err, fixed, velscale_ratio, npix, ncomb, nbins, i, gas, opts\
        in iter(inQueue.get, 'STOP'): 

        sol, w_row, bestfit, formal_error, gas_out, regul_used, boot = run_ppxf(templates, galaxy, noise, velscale,
                                            start, goodPixels_sfh, mom, dv, mdeg, regul_err, fixed, velscale_ratio,
                                            npix, ncomb, nbins, i, gas, opts)

        outQueue.put(( i, sol, w_row, bestfit, formal_error, gas_out, regul_used, boot ))


def _fit_setup(templates, start, nmom, fixed, ncomb, gas):
    """
    Templates and kinematic set-up of one bin for pPXF: (tpl, start, kw).
    gas = None: the stellar grid as it is (pPXF infers the regularisation grid from it).
    gas = dict: stellar grid flattened, gas templates appended as kinematic component 1.
    """
    if gas is None:
        return templates, start, dict(moments=nmom, fixed=fixed)
    ngas = gas['ngas']
    # Stellar grid (npix, nAge, nMetal, nAlpha) -> 2-D, gas appended as extra columns.
    # pPXF then needs reg_dim to know which columns form the regularised stellar grid
    # (only component 0 is regularised).
    reg_dim = templates.shape[1:]
    tpl = np.column_stack([templates.reshape(templates.shape[0], -1), gas['templates']])
    component = np.array([0] * ncomb + [1] * ngas)
    # Kinematics: stars as before (fixed to the KIN results, or free); gas: one free
    # component, starting at the stellar velocity.
    v0 = float(start[0])
    start_star = list(np.asarray(start, float))
    start_i = [start_star, [v0, gas['sigma0']]]
    fixed_i = None if fixed is None else [list(fixed), [False, False]]
    # Bounds: the gas may not wander off to fit noise or sky residuals:
    # |V_gas - V_star| <= vmax, sigma_gas <= sigmax. Stellar bounds are wide and always
    # contain the (possibly fixed) start values.
    def _b(lo, hi, x):
        return [min(lo, x), max(hi, x)]
    bstar = [_b(v0 - 2000., v0 + 2000., v0), _b(1., 1000., start_star[1] if len(start_star) > 1 else 100.)]
    bstar += [_b(-0.3, 0.3, h) for h in start_star[2:nmom]]
    while len(bstar) < nmom:
        bstar.append([-0.3, 0.3])
    if len(start_star) < nmom:          # free kinematics: start = [V, sigma]
        start_i[0] = start_star[:2]
    bounds = [bstar, [[v0 - gas['vmax'], v0 + gas['vmax']], [1., gas['sigmax']]]]
    kw = dict(moments=[nmom, 2], fixed=fixed_i, component=component, gas_component=component > 0,
              gas_names=gas['names'], reg_dim=reg_dim, bounds=bounds)
    # Gas attenuation (SFH GAS_REDDENING, with the Case-B 'Balmer' template): A_V of the
    # gas templates fitted with pPXF's Calzetti curve; pPXF needs the rest wavelengths
    if gas.get('reddening') is not None:
        kw['gas_reddening'] = float(gas['reddening'])
        kw['lam'] = gas['lam']
    return tpl, start_i, kw


def _normalise(galaxy_i, noise_i, goodPixels):
    """
    Divide the spectrum and its noise by the median flux of the good pixels. The
    templates are normalised to mean 1, so the fitted weights then sum to ~1 whatever the
    flux units or the brightness of the bin; the regularisation penalises differences
    between weights in absolute terms, so without this the same REGUL_ERR smooths a faint
    bin much more strongly than a bright one. Kinematics, chi2 and the (normalised)
    light weights do not change; best fit and gas fluxes are scaled back afterwards.
    """
    g = galaxy_i[goodPixels]
    scale = np.nanmedian(g)
    if not (np.isfinite(scale) and scale > 0):
        scale = np.nanmean(np.abs(g))
    if not (np.isfinite(scale) and scale > 0):
        scale = 1.
    return galaxy_i / scale, noise_i / scale, scale


def find_regul(templates, galaxy_i, noise_i, velscale, start, goodPixels, nmom, dv, mdeg, fixed,
               velscale_ratio, ncomb, gas=None, log_lo=-1., log_hi=5., rtol=0.1, maxfit=25, ncorr=1.):
    """
    The largest regularisation consistent with the data (Press et al. 1992; Cappellari
    2017, sect. 3.5; pPXF's population examples): with the noise rescaled so that the
    UNregularised fit has chi2/DOF = 1, increase regul until the total chi2 has risen
    by Delta = sqrt(2 N_good) above it. Delta chi2 grows monotonically with regul, so a
    bisection in log10(regul) finds it in ~10 fits.
    ncorr: the noise of neighbouring pixels is correlated (e.g. after MATCH_RES smoothed
    the data): the scatter of chi2 is then sqrt(2 N_good ncorr), ncorr = sum of the
    squared pixel-to-pixel correlation coefficients (1 for independent pixels).

    Returns (regul_err = 1/regul, info). info: target and reached Delta chi2, the number
    of fits, and 'flag': 'ok', 'at_min' (even regul = 10**log_lo exceeds the target:
    the data constrain the weights strongly -- use the smallest value) or 'at_max'
    (the data cannot tell even regul = 10**log_hi from the best fit: weights
    unconstrained, the smoothest value is returned).
    """
    goodPixels = _noise.usable_pixels(goodPixels, noise_i)
    galaxy_i, noise_i, _ = _normalise(galaxy_i, noise_i, goodPixels)
    tpl, start_i, kw = _fit_setup(templates, start, nmom, fixed, ncomb, gas)
    common = dict(goodpixels=goodPixels, plot=False, quiet=True, degree=-1, vsyst=dv, mdegree=mdeg,
                  velscale_ratio=velscale_ratio)
    pp0 = ppxf(tpl, galaxy_i, noise_i, velscale, start_i, **common, **kw)
    noise_r = noise_i * np.sqrt(pp0.chi2)                      # unregularised chi2/DOF = 1
    pp1 = ppxf(tpl, galaxy_i, noise_r, velscale, start_i, **common, **kw)
    chi0 = pp1.chi2 * pp1.dof
    target = np.sqrt(2. * goodPixels.size * max(float(ncorr), 1.))
    nfit = [2]

    def dchi2(logr):
        pp = ppxf(tpl, galaxy_i, noise_r, velscale, start_i, regul=10. ** logr, **common, **kw)
        nfit[0] += 1
        return pp.chi2 * pp.dof - chi0

    lo, hi = log_lo, log_hi
    d_lo, d_hi = dchi2(lo), dchi2(hi)
    if d_lo >= target:
        return 10. ** -lo, dict(target=target, dchi2=d_lo, nfit=nfit[0], flag='at_min')
    if d_hi <= target:
        return 10. ** -hi, dict(target=target, dchi2=d_hi, nfit=nfit[0], flag='at_max')
    best, d_best = None, None
    while nfit[0] < maxfit:
        mid = 0.5 * (lo + hi)
        d = dchi2(mid)
        if best is None or abs(d - target) < abs(d_best - target):
            best, d_best = mid, d
        if abs(d / target - 1.) < rtol or hi - lo < 0.02:
            break
        if d < target:
            lo = mid
        else:
            hi = mid
    return 10. ** -best, dict(target=target, dchi2=d_best, nfit=nfit[0], flag='ok')


def _light_means(w, grids):
    """Weighted means of (log age, age [Gyr], [M/H], [alpha/Fe]) for the weight vector w."""
    w = np.asarray(w, float)
    t = w.sum()
    if not (np.isfinite(t) and t > 0):
        return np.full(4, np.nan)
    lg, me, al = grids
    return np.array([np.sum(w * lg), np.sum(w * 10 ** lg), np.sum(w * me), np.sum(w * al)]) / t


def _rademacher(rng, n, block):
    """Random signs +-1, constant over blocks of `block` pixels (block wild bootstrap:
    keeps the pixel-to-pixel correlation of the residuals, e.g. after MATCH_RES)."""
    s = rng.randint(0, 2, size=int(np.ceil(n / float(block)))) * 2 - 1
    return np.repeat(s, block)[:n].astype(float)


def run_ppxf(templates, galaxy_i, noise_i, velscale, start, goodPixels, nmom, dv, mdeg,\
             regul_err, fixed, velscale_ratio, npix, ncomb, nbins, i, gas=None, opts=None):
    """
    Calls the penalised Pixel-Fitting routine from Cappellari & Emsellem 2004 
    (ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C; 
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C), in order to determine the
    non-parametric star-formation histories. 

    gas = None: stars only, exactly as before.
    gas = dict(templates, sigma0, vmax, sigmax, ngas, ...): the gas templates are appended
          after the stellar grid as kinematic component 1 (one V, sigma for all lines)
          and fitted together with the stars; the regularisation acts on the stars
          only (reg_dim). Returns, in addition, the gas kinematics, fluxes and best fit.
    regul_err = 'auto': found in this bin with find_regul (Delta chi2 = sqrt(2 N_good ncorr)).
    opts: dict(nboot, ncorr, grids, seed). nboot > 0: wild bootstrap of the residuals
          (Davidson & Flachaire 2008; Cappellari 2023, pPXF population-bootstrap example): nboot UNregularised
          refits of bestfit + residual x random signs; the scatter of the light-weighted
          <log age>, <age>, [M/H], [alpha/Fe] are the errors.
    Returns (sol, w_row, bestfit, formal_error, gas_out, regul_err used, bootstrap errors (4,)).
    """ 
    printStatus.progressBar(i, nbins, barLength = 50)
    ngas = 0 if gas is None else gas['ngas']
    opts = opts or {}
    nboot = int(opts.get('nboot', 0) or 0)
    ncorr = float(opts.get('ncorr', 1.) or 1.)

    try:
        if fixed is not None and not np.all(np.isfinite(start)):
            raise ValueError("the fixed kinematics of this bin are not finite (its KIN fit failed)")

        # Pixels without data in this bin (gap, long masked runs) leave the fit
        goodPixels = _noise.usable_pixels(goodPixels, noise_i)

        if isinstance(regul_err, str):            # 'auto': the Delta chi2 criterion, per bin
            regul_err, _ = find_regul(templates, galaxy_i, noise_i, velscale, start, goodPixels, nmom, dv, mdeg,
                                      fixed, velscale_ratio, ncomb, gas, ncorr=ncorr)

        # Weights of order unity in every bin, so REGUL_ERR means the same everywhere
        galaxy_i, noise_i, scale = _normalise(galaxy_i, noise_i, goodPixels)

        tpl, start_i, kw = _fit_setup(templates, start, nmom, fixed, ncomb, gas)
        common = dict(goodpixels=goodPixels, plot=False, quiet=True, degree=-1, vsyst=dv, mdegree=mdeg,
                      velscale_ratio=velscale_ratio)

        # REGUL_ERR only has its intended meaning if chi2/DOF = 1 for the
        # unregularised fit (pPXF documentation). With the variance-cube noise
        # that is not guaranteed, so rescale the noise first.
        if RESCALE_NOISE:
            pp0 = ppxf(tpl, galaxy_i, noise_i, velscale, start_i, **common, **kw)
            noise_i = noise_i * np.sqrt(pp0.chi2)

        pp = ppxf(tpl, galaxy_i, noise_i, velscale, start_i, regul=1./regul_err, **common, **kw)

        if gas is None:
            weights = pp.weights.reshape(templates.shape[1:])/pp.weights.sum()
            sol, err = pp.sol, pp.error
            gas_out = None
        else:
            # Stellar weights normalised by the STELLAR sum (the gas weights are line
            # fluxes, in other units: they must not enter the population weights)
            ws = pp.weights[:ncomb]
            weights = ws.reshape(templates.shape[1:]) / ws.sum()
            sol, err = pp.sol[0], pp.error[0]
            k = np.sqrt(pp.chi2)
            # basis weights and their formal errors (pPXF gives the error of
            # gas_flux = weight x pixel sum of the fitted template column)
            wg = np.array(pp.weights[ncomb:ncomb + ngas], float)
            integ = np.abs(pp.matrix[:, ncomb:ncomb + ngas].sum(0))
            with np.errstate(all='ignore'):
                ew = np.where(integ > 0, np.array(pp.gas_flux_error, float) / integ, np.nan) * k
            M = gas['M']
            lflux = M @ wg                                   # intrinsic (before GAS_REDDENING)
            lerr = np.sqrt(np.sum(np.where(M > 0, M ** 2 * ew[None, :] ** 2, 0.), axis=1))   # NaN only via a used template
            av = getattr(pp, 'gas_reddening', None) if gas.get('reddening') is not None else None
            if av is not None:
                # pPXF reddens the gas columns by attenuation(lam, A_V) (its default
                # gas_reddening_func): the weights are intrinsic, observed = x attenuation
                att = _attenuation(gas['lam1'], float(av))
                lflux, lerr = lflux * att, lerr * att
            gas_out = dict(sol=np.array(pp.sol[1], float), err=np.array(pp.error[1], float) * k,
                           flux=lflux * scale, flux_err=lerr * scale,
                           bestfit=np.array(pp.gas_bestfit, float) * scale,
                           av=float(av) if av is not None else np.nan)
        w_row   = np.reshape(weights, ncomb)

        # Correct the formal errors assuming that the fit is good
        formal_error = err * np.sqrt(pp.chi2)

        # Wild bootstrap of the residuals (unregularised refits, as in pPXF's
        # population-bootstrap example): errors of the light-weighted means
        boot = np.full(4, np.nan)
        if nboot > 0 and opts.get('grids') is not None:
            rng = np.random.RandomState((int(opts.get('seed', 0)) * 100003 + int(i)) % (2**31 - 1))
            resid = galaxy_i - pp.bestfit
            block = max(1, int(round(ncorr)))
            kw_b = dict(kw)
            # free kinematics start from the best fit (faster, same minimum)
            st_b = start_i if fixed is not None else (pp.sol.tolist() if gas is None else
                                                      [list(pp.sol[0]), list(pp.sol[1])])
            vals = []
            for _ in range(nboot):
                gal_b = pp.bestfit + resid * _rademacher(rng, resid.size, block)
                ppb = ppxf(tpl, gal_b, noise_i, velscale, st_b, regul=0, **common, **kw_b)
                vals.append(_light_means(ppb.weights[:ncomb], opts['grids']))
            vals = np.array(vals)
            ok = np.all(np.isfinite(vals), axis=1)
            if ok.sum() > 2:
                boot = np.std(vals[ok], axis=0, ddof=1)

        return(sol, w_row, pp.bestfit * scale, formal_error, gas_out, float(regul_err), boot)

    except Exception:
        # Report it, and return NaNs so the bin is listed at the end; letting
        # the exception escape kills the worker and the main process then
        # waits forever for its result
        _noise.log_failure('SFH', i)
        nl = 0 if gas is None else len(gas['lines'])
        gas_out = None if gas is None else dict(sol=np.full(2, np.nan), err=np.full(2, np.nan),
                                                flux=np.full(nl, np.nan), flux_err=np.full(nl, np.nan),
                                                bestfit=np.full(npix, np.nan), av=np.nan)
        return(np.nan, np.full(ncomb, np.nan), np.full(npix, np.nan), np.nan, gas_out, np.nan, np.full(4, np.nan))


def gas_templates(names, logLam_template, LSF_Data):
    """
    Unit-flux line templates on the stellar-template grid (ln lambda, rest frame,
    velscale/velscale_ratio), one column per GAS_LINES entry. Each line is a Gaussian
    with the INSTRUMENTAL width of the data at that wavelength (LSF_Data: rest-frame FWHM,
    as used to broaden the stellar templates; with MATCH_RES the matched LSF), integrated
    over each pixel (lines are only ~2 template pixels wide), so the fitted gas sigma is
    the intrinsic one, like the stellar sigma.
    Normalisation: in a spectrum per unit wavelength sampled in ln lambda a line's pixel
    sum is flux / (pixel width in A), i.e. proportional to flux / lambda. Column j is
    sum_k (rel_k / rel_1) (REF_LAM / lambda_k) G_k (G_k: unit-sum Gaussians), so that for
    every line, multiplets included, weight x REF_LAM x (1+z) velscale / c = flux of the
    named (first) line, in the units of the galaxy spectrum x A. A weight is then a line
    flux in the same units for all lines, which is what the ratio-limited combinations of
    gaslines.build_basis need.
    """
    ln = np.asarray(logLam_template, float)
    dln = ln[1] - ln[0]
    edges = np.append(ln - dln / 2., ln[-1] + dln / 2.)
    out = np.zeros((ln.size, len(names)))
    for j, nm in enumerate(names):
        lines = _gaslines.GAS_LINES[nm]
        rel1 = lines[0][1]
        for lam, rel in lines:
            s = float(LSF_Data(lam)) / 2.3548 / lam          # sigma in ln(lambda)
            c = np.log(lam)
            cdf = 0.5 * (1. + erf((edges - c) / (np.sqrt(2.) * s)))
            out[:, j] += (rel / rel1) * (_gaslines.REF_LAM / lam) * np.diff(cdf)
    return out


def mean_agemetalalpha(w_row, ageGrid, metalGrid, alphaGrid, nbins):
    """
    Calculate the mean age, metallicity and alpha enhancement in each bin. 
    """
    mean = np.zeros( (nbins,3) ); mean[:,:] = np.nan

    for i in range( nbins ):
        mean[i,0] = np.sum(w_row[i] * ageGrid.ravel())   / np.sum(w_row[i])
        mean[i,1] = np.sum(w_row[i] * metalGrid.ravel()) / np.sum(w_row[i])
        mean[i,2] = np.sum(w_row[i] * alphaGrid.ravel()) / np.sum(w_row[i])
    
    return(mean)


def save_sfh(mean_result, kin, formal_error, w_row, logAge_grid, metal_grid, alpha_grid, bestfit, logLam_galaxy, goodPixels,\
             velscale, logLam1, ncomb, nAges, nMetal, nAlpha, npix, config, gas_res=None, regul_used=None,
             logage=None, boot=None):
    """ Save all results to disk. """
    # ========================
    # SAVE KINEMATICS
    outfits_sfh = os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_sfh.fits'
    printStatus.running("Writing: "+config['GENERAL']['RUN_ID']+'_sfh.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with stellar kinematics
    cols = []
    cols.append(fits.Column(name='AGE',   format='D', array=mean_result[:,0]))
    cols.append(fits.Column(name='METAL', format='D', array=mean_result[:,1]))
    cols.append(fits.Column(name='ALPHA', format='D', array=mean_result[:,2]))
    if logage is not None:
        # light-weighted <log10 age [yr]> (the usual SFH average; AGE is <age> in Gyr)
        cols.append(fits.Column(name='LOGAGE', format='D', array=logage))
    if boot is not None and np.any(np.isfinite(boot)):
        # wild-bootstrap errors (SFH BOOTSTRAP): scatter of the unregularised refits
        for j, nm in enumerate(['LOGAGE', 'AGE', 'METAL', 'ALPHA']):
            cols.append(fits.Column(name='ERR_' + nm, format='D', array=boot[:, j]))
    if regul_used is not None:
        cols.append(fits.Column(name='REGUL_ERR', format='D', array=regul_used))   # value used per bin

    if config['SFH']['FIXED'] == False:
        cols.append(fits.Column(name='V',     format='D', array=kin[:,0]))
        cols.append(fits.Column(name='SIGMA', format='D', array=kin[:,1]))
        if np.any(kin[:,2]) != 0: cols.append(fits.Column(name='H3', format='D', array=kin[:,2]))
        if np.any(kin[:,3]) != 0: cols.append(fits.Column(name='H4', format='D', array=kin[:,3]))
        if np.any(kin[:,4]) != 0: cols.append(fits.Column(name='H5', format='D', array=kin[:,4]))
        if np.any(kin[:,5]) != 0: cols.append(fits.Column(name='H6', format='D', array=kin[:,5]))
    
        cols.append(fits.Column(name='FORM_ERR_V',     format='D', array=formal_error[:,0]))
        cols.append(fits.Column(name='FORM_ERR_SIGMA', format='D', array=formal_error[:,1]))
        if np.any(formal_error[:,2]) != 0: cols.append(fits.Column(name='FORM_ERR_H3', format='D', array=formal_error[:,2]))
        if np.any(formal_error[:,3]) != 0: cols.append(fits.Column(name='FORM_ERR_H4', format='D', array=formal_error[:,3]))
        if np.any(formal_error[:,4]) != 0: cols.append(fits.Column(name='FORM_ERR_H5', format='D', array=formal_error[:,4]))
        if np.any(formal_error[:,5]) != 0: cols.append(fits.Column(name='FORM_ERR_H6', format='D', array=formal_error[:,5]))

    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'SFH'

    # Create HDU list and write to file
    priHDU  = _auxiliary.saveConfigToHeader(priHDU, config['SFH'])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config['SFH'])
    HDUList = fits.HDUList([priHDU, dataHDU])
    HDUList.writeto(outfits_sfh, overwrite=True)

    printStatus.updateDone("Writing: "+config['GENERAL']['RUN_ID']+'_sfh.fits')
    logging.info("Wrote: "+outfits_sfh)


    # ========================
    # SAVE WEIGHTS AND GRID
    outfits_sfh = os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_sfh-weights.fits'
    printStatus.running("Writing: "+config['GENERAL']['RUN_ID']+'_sfh-weights.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with weights
    cols = []
    cols.append( fits.Column(name='WEIGHTS', format=str(w_row.shape[1])+'D', array=w_row ))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'WEIGHTS'

    logAge_row = np.reshape(logAge_grid, ncomb)
    metal_row  = np.reshape(metal_grid,  ncomb)
    alpha_row  = np.reshape(alpha_grid,  ncomb)

    # Table HDU with grids
    cols = []
    cols.append( fits.Column(name='LOGAGE',  format='D',           array=logAge_row  ))
    cols.append( fits.Column(name='METAL',   format='D',           array=metal_row   ))
    cols.append( fits.Column(name='ALPHA',   format='D',           array=alpha_row   ))
    gridHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    gridHDU.name = 'GRID'

    # Create HDU list and write to file
    priHDU  = _auxiliary.saveConfigToHeader(priHDU, config['SFH'])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config['SFH'])
    gridHDU = _auxiliary.saveConfigToHeader(gridHDU, config['SFH'])
    HDUList = fits.HDUList([priHDU, dataHDU, gridHDU])
    HDUList.writeto(outfits_sfh, overwrite=True)

    fits.setval(outfits_sfh,'NAGES',  value=nAges)
    fits.setval(outfits_sfh,'NMETAL', value=nMetal)
    fits.setval(outfits_sfh,'NALPHA', value=nAlpha)

    printStatus.updateDone("Writing: "+config['GENERAL']['RUN_ID']+'_sfh-weights.fits')
    logging.info("Wrote: "+outfits_sfh)


    # ========================
    # SAVE BESTFIT
    outfits_sfh = os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_sfh-bestfit.fits'
    printStatus.running("Writing: "+config['GENERAL']['RUN_ID']+'_sfh-bestfit.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU with SFH bestfit
    cols = []
    cols.append( fits.Column(name='BESTFIT', format=str(npix)+'D', array=bestfit ))
    if gas_res is not None:
        # stars alone = BESTFIT - GAS_BESTFIT
        cols.append( fits.Column(name='GAS_BESTFIT', format=str(npix)+'D', array=gas_res['bestfit'] ))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'BESTFIT'

    # Table HDU with SFH logLam
    cols = []
    cols.append( fits.Column(name='LOGLAM', format='D', array=logLam_galaxy ))
    logLamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    logLamHDU.name = 'LOGLAM'

    # Table HDU with SFH goodpixels
    cols = []
    cols.append( fits.Column(name='GOODPIX', format='J', array=goodPixels ))
    goodpixHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    goodpixHDU.name = 'GOODPIX'

    # Create HDU list and write to file
    priHDU     = _auxiliary.saveConfigToHeader(priHDU, config['SFH'])
    dataHDU    = _auxiliary.saveConfigToHeader(dataHDU, config['SFH'])
    logLamHDU  = _auxiliary.saveConfigToHeader(logLamHDU, config['SFH'])
    goodpixHDU = _auxiliary.saveConfigToHeader(goodpixHDU, config['SFH'])
    HDUList = fits.HDUList([priHDU, dataHDU, logLamHDU, goodpixHDU])
    HDUList.writeto(outfits_sfh, overwrite=True)

    fits.setval(outfits_sfh,'VELSCALE',value=velscale)
    fits.setval(outfits_sfh,'CRPIX1',  value=1.0)
    fits.setval(outfits_sfh,'CRVAL1',  value=logLam1[0])
    fits.setval(outfits_sfh,'CDELT1',  value=logLam1[1]-logLam1[0])

    printStatus.updateDone("Writing: "+config['GENERAL']['RUN_ID']+'_sfh-bestfit.fits')
    logging.info("Wrote: "+outfits_sfh)

    if gas_res is not None:
        save_sfh_gas(gas_res, velscale, config)


def save_sfh_gas(gas_res, velscale, config):
    """
    _sfh-gas.fits: kinematics and fluxes of the emission lines fitted together with the
    stars. LFLUX_<line> is the flux of the NAMED line (e.g. 5007 without 4959; Hbeta for
    the Case-B Balmer template), observed (not corrected for AV_GAS), in the cube's
    flux-density units x Angstrom (the cube being per observed A, as KCWI's). With
    GAS_RATIOS a line flux is a sum over the ratio-limited templates; its formal error
    adds theirs in quadrature. Errors are formal, scaled by sqrt(chi2/DOF); a line with no
    data in the fit (outside LMIN-LMAX, or masked) gives NaN.
    """
    outfits = os.path.join(config['GENERAL']['OUTPUT'], config['GENERAL']['RUN_ID']) + '_sfh-gas.fits'
    printStatus.running("Writing: " + config['GENERAL']['RUN_ID'] + '_sfh-gas.fits')
    z = config['GENERAL']['REDSHIFT']
    cols = [fits.Column(name='V_GAS', format='D', array=gas_res['sol'][:, 0]),
            fits.Column(name='SIGMA_GAS', format='D', array=gas_res['sol'][:, 1]),
            fits.Column(name='FORM_ERR_V_GAS', format='D', array=gas_res['err'][:, 0]),
            fits.Column(name='FORM_ERR_SIGMA_GAS', format='D', array=gas_res['err'][:, 1])]
    if np.any(np.isfinite(gas_res.get('av', np.nan))):
        # A_V of the gas templates (SFH GAS_REDDENING; Calzetti 2000 curve)
        cols.append(fits.Column(name='AV_GAS', format='D', array=gas_res['av']))
    # weight -> flux (see gas_templates); pPXF averages velscale_ratio template pixels
    # into one galaxy pixel, so a unit weight has pixel sum REF_LAM/lambda/velscale_ratio
    k = _gaslines.REF_LAM * (1 + z) * velscale / gas_res.get('vratio', 2) / C
    for j, nm in enumerate(gas_res['names']):
        tag = nm.replace('[', '').replace(']', '')
        cols += [fits.Column(name='LFLUX_' + tag, format='D', array=gas_res['flux'][:, j] * k),
                 fits.Column(name='LFLUXERR_' + tag, format='D', array=gas_res['flux_err'][:, j] * k)]
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'GAS'
    priHDU = _auxiliary.saveConfigToHeader(fits.PrimaryHDU(), config['SFH'])
    dataHDU = _auxiliary.saveConfigToHeader(dataHDU, config['SFH'])
    fits.HDUList([priHDU, dataHDU]).writeto(outfits, overwrite=True)
    fits.setval(outfits, 'GASLINES', value=','.join(gas_res['names']))
    printStatus.updateDone("Writing: " + config['GENERAL']['RUN_ID'] + '_sfh-gas.fits')
    logging.info("Wrote: " + outfits)


def _set(v):
    """A MasterConfig value that is actually set (not None / empty / 'None' / False)."""
    return v is not None and v is not False and str(v).strip().lower() not in ('', 'none', 'false')


def _lsf_file(config, name):
    """Rest-frame FWHM [A] of a template LSF file in configFiles/ (as getLSF reads LSF_TEMP)."""
    lsf = np.genfromtxt(os.path.join(config['GENERAL']['CONFIG_DIR'], name), comments='#')
    return interp1d(lsf[:, 0], lsf[:, 1], 'linear', fill_value='extrapolate')


def sfh_template_config(config, LSF_Templates):
    """
    The template library of the SFH fit. By default the one of KIN (PREPARE_TEMPLATES
    METHOD / LIBRARY / NORM_TEMP, GENERAL LSF_TEMP); SFH TEMPLATES, LIBRARY, NORM_TEMP and
    LSF_TEMP replace them for the SFH fit only -- e.g. KIN with XSL (R ~ 10000, exact
    kinematics) and SFH with E-MILES (METHOD miles, 2.51 A). Returns (config copy to
    give prepareTemplates_Module, template LSF, description).
    """
    cfg = dict(config)
    cfg['PREPARE_TEMPLATES'] = dict(config['PREPARE_TEMPLATES'])
    own = []
    for k_sfh, k_pt in (('TEMPLATES', 'METHOD'), ('LIBRARY', 'LIBRARY'), ('NORM_TEMP', 'NORM_TEMP')):
        v = config['SFH'].get(k_sfh)
        if _set(v):
            cfg['PREPARE_TEMPLATES'][k_pt] = v
            own.append(k_sfh)
    lsf_name = config['GENERAL']['LSF_TEMP']
    if _set(config['SFH'].get('LSF_TEMP')):
        lsf_name = config['SFH']['LSF_TEMP']
        LSF_Templates = _lsf_file(config, lsf_name)
        own.append('LSF_TEMP')
    elif own:
        logging.warning("SFH: own template library (%s) but no SFH LSF_TEMP: the template LSF is "
                        "GENERAL LSF_TEMP = %s -- check that it describes that library" % (
                            ', '.join(own), lsf_name))
    desc = "%s templates, %s (LSF %s)%s" % (cfg['PREPARE_TEMPLATES']['METHOD'], cfg['PREPARE_TEMPLATES']['LIBRARY'],
                                           lsf_name, ' -- SFH only' if own else '')
    return cfg, LSF_Templates, desc


def _smooth_rows(y, sig, power=1):
    """
    Convolve every row of y (nbins, npix) with a Gaussian whose sigma [pixels] varies
    along the spectrum (sig: (npix,), the sigma at each OUTPUT pixel, as pPXF's
    gaussian_filter1d), with the ends extended by their edge value. power=2 applies
    the squared kernel (for a variance: var_out = sum K^2 var_in).
    """
    y = np.atleast_2d(np.asarray(y, float))
    sig = np.clip(np.asarray(sig, float), 1e-3, None)
    p = int(np.ceil(4 * sig.max()))
    if p < 1 or sig.max() < 0.05:
        return y.copy()
    n = y.shape[1]
    d = np.arange(-p, p + 1)
    k = np.exp(-0.5 * (d[:, None] / sig[None, :]) ** 2)
    k /= k.sum(0)
    if power == 2:
        k = k ** 2
    ypad = np.concatenate([np.repeat(y[:, :1], p, 1), y, np.repeat(y[:, -1:], p, 1)], axis=1)
    out = np.zeros_like(y)
    for j, dj in enumerate(d):
        out += k[j] * ypad[:, p + dj: p + dj + n]
    return out


def match_resolution(galaxy, noise, logLam_galaxy, velscale, LSF_Data, LSF_fit):
    """
    MATCH_RES: degrade the binned spectra from the data LSF to LSF_fit (both rest-frame
    FWHM [A]) where LSF_fit is broader -- the templates are then broadened to LSF_fit
    too, so data and templates share one resolution (Zhuang et al. 2024 do the same with
    KCWI and the 2.5 A FSPS-MILES models). The kinematics stay physical: a FIXED sigma
    from KIN (measured at full resolution) is still the right one.
      galaxy, noise: (nbins, npix), as in prepare_sfh.
    The noise becomes sqrt(K^2 * var) per pixel and neighbouring pixels are correlated;
    pixels without data (and up to 4 sigma from them, and from the ends) are flagged
    with a huge noise so that the fits drop them. Returns (galaxy, noise, info) with
    info: kernel sigma [pixels] along the spectrum, ncorr = sum of squared correlation
    coefficients of the smoothed noise (scales the chi2 scatter; 1 = independent pixels).
    """
    lam = np.exp(logLam_galaxy)
    dfw2 = LSF_fit(lam) ** 2 - LSF_Data(lam) ** 2
    sig = np.sqrt(np.clip(dfw2, 0, None)) / 2.3548 / lam / (velscale / C)      # [pixels]
    if sig.max() < 0.05:
        return galaxy, noise, dict(sig=sig, ncorr=1.)
    p = int(np.ceil(4 * sig.max()))
    g = np.array(galaxy, float)
    v = np.array(noise, float) ** 2                                          # (nbins, npix)
    with np.errstate(all='ignore'):
        ref = np.nanmedian(np.where(np.isfinite(v) & (v > 0), v, np.nan), axis=1)
    bad = ~np.isfinite(g) | ~np.isfinite(v) | (v <= 0) | (v > (_noise.NOISE_BAD_FACTOR ** 2) * ref[:, None])
    # fill bad pixels (by interpolation) so they do not leak into their neighbours,
    # and flag their neighbourhood afterwards
    x = np.arange(g.shape[1])
    for i in np.flatnonzero(bad.any(axis=1)):
        ok = ~bad[i]
        if ok.sum() < 2:
            continue
        g[i, ~ok] = np.interp(x[~ok], x[ok], g[i, ok])
        v[i, ~ok] = ref[i]
    gs = _smooth_rows(g, sig)
    vs = _smooth_rows(v, sig, power=2)
    grow = np.zeros_like(bad)
    from scipy.ndimage import binary_dilation
    if bad.any():
        grow = binary_dilation(bad, structure=np.ones((1, 2 * p + 1), bool))
    grow[:, :p] = True
    grow[:, -p:] = True
    ns = np.sqrt(vs)
    ns[grow] = _noise.BIG_NOISE_FACTOR * np.sqrt(ref)[:, None].repeat(g.shape[1], 1)[grow]
    # correlation of the smoothed noise at the median kernel width
    s0 = float(np.median(sig[sig > 0])) if np.any(sig > 0) else 0.
    if s0 > 0.05:
        dd = np.arange(-int(np.ceil(4 * s0)), int(np.ceil(4 * s0)) + 1)
        kk = np.exp(-0.5 * (dd / s0) ** 2)
        ac = np.correlate(kk, kk, 'full') / np.sum(kk ** 2)
        ncorr = float(np.sum(ac ** 2))
    else:
        ncorr = 1.
    return gs, ns, dict(sig=sig, ncorr=ncorr)


def prepare_sfh(config):
    """
    Everything the SFH fit needs, read from a run that has finished the spectra (and,
    with FIXED, the KIN) steps: templates (+ gas templates), binned spectra, noise,
    starting/fixed kinematics, good pixels. Shared by extractStarFormationHistories and
    tools/gist_regul.py, so that both fit exactly the same thing.
    """
    # Noise rescaling before the regularised fit (inherited by the workers)
    global RESCALE_NOISE
    RESCALE_NOISE = _noise.as_bool(config['SFH'].get('NOISE_RESCALE'), default=True)
    logging.info("SFH: noise rescaled to chi2/DOF=1 before the regularised fit: "+str(RESCALE_NOISE))

    # Read LSF information; the SFH fit may use its own template library
    LSF_Data, LSF_Templates = _auxiliary.getLSF(config)
    cfg_t, LSF_Templates, tdesc = sfh_template_config(config, LSF_Templates)
    logging.info("SFH: " + tdesc)
    printStatus.done("SFH: " + tdesc)

    # Resolution: templates broader than the data anywhere in the fit range (e.g.
    # E-MILES 2.51 A vs KCWI-blue ~1.0-1.5 A) -> MATCH_RES degrades the data to the
    # broader of the two LSFs (MATCH_RES = auto: only if needed; True; False: stop).
    lmin, lmax = float(config['SFH']['LMIN']), float(config['SFH']['LMAX'])
    lam_chk = np.linspace(lmin - 150., lmax + 150., 400)
    coarser = LSF_Templates(lam_chk) > LSF_Data(lam_chk) * 1.001
    mres = str(config['SFH'].get('MATCH_RES', 'auto')).strip().lower()
    if mres in ('none', '', 'auto'):
        do_match = bool(np.any(coarser))
    else:
        do_match = _noise.as_bool(mres)
    if np.any(coarser) and not do_match:
        raise ValueError("SFH: the templates are broader than the data at %.0f-%.0f A (template FWHM up to %.2f A, "
                         "data %.2f A) and MATCH_RES = False: set MATCH_RES = auto to degrade the data" % (
                             lam_chk[coarser].min(), lam_chk[coarser].max(), LSF_Templates(lam_chk[coarser]).max(),
                             LSF_Data(lam_chk[coarser]).min()))
    if do_match:
        _ld, _lt = LSF_Data, LSF_Templates
        LSF_fit = lambda l: np.maximum(_ld(l), _lt(l))
    else:
        LSF_fit = LSF_Data

    # Prepare template library (broadened to LSF_fit)
    velscale = fits.open(os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+"_BinSpectra.fits")[0].header['VELSCALE']
    velscale_ratio = 2
    templates, lamRange_temp, logLam_template, ntemplates, logAge_grid, metal_grid, alpha_grid, ncomb, nAges, nMetal, nAlpha = \
            _prepareTemplates.prepareTemplates_Module(cfg_t, lmin, lmax, velscale/velscale_ratio, LSF_fit, LSF_Templates, sortInGrid=True)

    # Joint stars + gas fit (SFH GAS_LINES): emission lines fitted with the stars
    # instead of masked. See gistPipeline/auxiliary/gaslines.py for the lines.
    gas_names = _gaslines.parse_gas_lines(config['SFH'].get('GAS_LINES', None))
    if gas_names:
        inside = [n for n in gas_names if any(lmin < lam < lmax for lam, _ in _gaslines.GAS_LINES[n])]
        dropped = [n for n in gas_names if n not in inside]
        if dropped:
            logging.info("SFH gas: outside LMIN-LMAX, not fitted: " + ", ".join(dropped))
        gas_names = inside

    # Read spectra
    if not gas_names and os.path.isfile(os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_gas-cleaned_BIN.fits') == True:
        logging.info('Using emission-subtracted spectra at '+os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_gas-cleaned_BIN.fits')
        printStatus.done("Using emission-subtracted spectra")
        hdu = fits.open(os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_gas-cleaned_BIN.fits')
    else:
        if gas_names and os.path.isfile(os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_gas-cleaned_BIN.fits'):
            logging.warning("SFH GAS_LINES is set: the GAS module's emission-subtracted spectra are NOT used "
                            "(the gas is fitted here); _BinSpectra.fits is fitted instead")
        logging.info('Using regular spectra without any emission-correction at '+os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_BinSpectra.fits')
        printStatus.done("Using regular spectra without any emission-correction")
        hdu = fits.open(os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_BinSpectra.fits')

    galaxy        = np.array( hdu[1].data.SPEC )
    logLam_galaxy = hdu[2].data.LOGLAM
    idx_lam       = np.where( np.logical_and( np.exp(logLam_galaxy) > lmin, np.exp(logLam_galaxy) < lmax ) )[0]
    galaxy        = galaxy[:,idx_lam]
    logLam_galaxy = logLam_galaxy[idx_lam]
    nbins         = galaxy.shape[0]
    npix          = galaxy.shape[1]
    ubins         = np.arange(0, nbins)
    noise         = _noise.sanitize( _noise.read_bin_espec(config, logLam_galaxy).T ).T
    # pPXF with velscale_ratio: the reference is the centre of the first *combined*
    # template pixel, mean(lam_temp[:velscale_ratio]) (what pPXF itself uses when given
    # lam and lam_temp). logLam_template[0] alone biased every velocity by
    # +velscale/(2*velscale_ratio) = +10 km/s for VELSCALE 40 (same fix in the KIN module,
    # so FIXED kinematics stay consistent).
    dv            = C * (np.log(np.mean(np.exp(logLam_template[:velscale_ratio]))) - logLam_galaxy[0])

    ncorr = 1.
    if do_match:
        galaxy, noise, minfo = match_resolution(galaxy, noise, logLam_galaxy, velscale, LSF_Data, LSF_fit)
        ncorr = minfo['ncorr']
        lam_g = np.exp(logLam_galaxy)
        sg = minfo['sig'] * (velscale / C) * lam_g * 2.3548
        msg = ("SFH: MATCH_RES -- data degraded to the template resolution: FWHM %.2f-%.2f A -> %.2f-%.2f A "
               "(extra %.2f-%.2f A); noise correlated over ~%.1f pixels (ncorr, used by REGUL_ERR = auto, gist_regul "
               "and the bootstrap)" % (LSF_Data(lam_g).min(), LSF_Data(lam_g).max(), LSF_fit(lam_g).min(),
                                       LSF_fit(lam_g).max(), sg.min(), sg.max(), ncorr))
        logging.info(msg)
        printStatus.done(msg)

    gas = None
    if gas_names:
        velscale_gal = velscale
        U = gas_templates(gas_names, logLam_template, LSF_fit)          # unit-flux line templates
        ratios = _noise.as_bool(config['SFH'].get('GAS_RATIOS'), default=True)
        labels, M = _gaslines.build_basis(gas_names, ratios=ratios)
        gas = dict(templates=U @ M, names=labels, ngas=len(labels), lines=gas_names, M=M,
                   lam1=np.array([_gaslines.GAS_LINES[n][0][0] for n in gas_names]),
                   sigma0=float(config['SFH'].get('GAS_SIGMA0', None) or max(100., 3 * velscale_gal)),
                   vmax=float(config['SFH'].get('GAS_VMAX', None) or 1000.),
                   sigmax=float(config['SFH'].get('GAS_SIGMAX', None) or 1000.))
        nlim = len(labels) - len(gas_names)
        # gas attenuation fitted (needs the tied Case-B Balmer template to mean anything)
        gred = config['SFH'].get('GAS_REDDENING')
        if _set(gred):
            if 'Balmer' not in gas_names:
                raise ValueError("SFH GAS_REDDENING needs the Case-B Balmer template (GAS_LINES ...,balmer_tied): "
                                 "without it nothing in the fit constrains the gas attenuation")
            if _attenuation is None:
                raise ValueError("SFH GAS_REDDENING needs a pPXF with gas_reddening (>= 8)")
            gas['reddening'] = float(gred)
            gas['lam'] = np.exp(logLam_galaxy)
        msg = ("SFH: stars and gas fitted together; %d lines in %d gas templates (one V, sigma): %s; "
               "regularisation on the stars only" % (len(gas_names), len(labels), ", ".join(gas_names)))
        if ratios and nlim:
            msg += "; line ratios limited to their physical range (GAS_RATIOS): " + ", ".join(
                l for l in labels if '/' in l or l.startswith('Balmer>='))
        if gas.get('reddening') is not None:
            msg += "; gas A_V fitted (start %g)" % gas['reddening']
        if set(gas_names) & _gaslines.BALMER:
            msg += ("; Balmer lines fitted as NARROW gas -- only valid without broad-line emission")
        logging.info(msg)
        printStatus.done(msg)

    # Implementation of switch FIXED
    # Do fix kinematics to those obtained previously
    if config['SFH']['FIXED'] == True:
        logging.info('Stellar kinematics are FIXED to the results obtained before.')
        # Set fixed option to True
        # Columns are read by name: _kin.fits omits all-zero H3..H6 columns,
        # so "the first MOM columns" could be LAMBDA_R or ERR_V
        nmom = config['SFH']['MOM']
        if nmom > config['KIN']['MOM']:
            raise ValueError("SFH|MOM (%d) > KIN|MOM (%d): cannot fix moments that were not fitted" % (nmom, config['KIN']['MOM']))
        fixed = [True]*nmom

        # Read PPXF results
        ppxf_data = fits.open(os.path.join(config['GENERAL']['OUTPUT'],config['GENERAL']['RUN_ID'])+'_kin.fits')[1].data
        start = np.zeros((nbins, nmom))
        for j, name in enumerate(['V', 'SIGMA', 'H3', 'H4', 'H5', 'H6'][:nmom]):
            if name in ppxf_data.columns.names:
                start[:,j] = ppxf_data[name]

    # Do *NOT* fix kinematics to those obtained previously
    elif config['SFH']['FIXED'] == False:
        logging.info('Stellar kinematics are NOT FIXED to the results obtained before but extracted simultaneously with the stellar population properties.')
        # Set fixed option to False and use initial guess from Config-file
        fixed = None
        start = np.zeros((nbins, 2))
        for i in range(nbins):
            start[i,:] = np.array( [0.0, config['SFH']['SIGMA']] )

    # Define goodpixels
    goodPixels_sfh = _auxiliary.spectralMasking(config, config['SFH']['SPEC_MASK'], logLam_galaxy)

    # Options of every bin's fit: bootstrap, noise correlation
    nboot = config['SFH'].get('BOOTSTRAP', 0)
    nboot = int(float(nboot)) if _set(nboot) else 0
    grids = (np.ravel(logAge_grid), np.ravel(metal_grid), np.ravel(alpha_grid))
    opts = dict(nboot=nboot, ncorr=ncorr, grids=grids, seed=0)
    if nboot:
        logging.info("SFH: wild bootstrap with %d unregularised refits per bin (block %d pixels)" % (nboot, max(1, int(round(ncorr)))))

    return dict(templates=templates, logLam_template=logLam_template, logAge_grid=logAge_grid,
                metal_grid=metal_grid, alpha_grid=alpha_grid, ncomb=ncomb, nAges=nAges, nMetal=nMetal,
                nAlpha=nAlpha, gas=gas, galaxy=galaxy, logLam_galaxy=logLam_galaxy, nbins=nbins, npix=npix,
                noise=noise, dv=dv, velscale=velscale, velscale_ratio=velscale_ratio, fixed=fixed, start=start,
                goodPixels=goodPixels_sfh, opts=opts, LSF_fit=LSF_fit, match_res=do_match)


def extractStarFormationHistories(config):
    """ 
    Starts the computation of non-parametric star-formation histories with
    pPXF.  A spectral template library sorted in a three-dimensional grid of
    age, metallicity, and alpha-enhancement is loaded.  Emission-subtracted
    spectra are used for the fit. An according emission-line mask is
    constructed. The stellar kinematics can or cannot be fixed to those obtained
    with a run of unregularized pPXF and the analysis started.  Results are
    saved to disk and the plotting routines called.
    """

    d = prepare_sfh(config)
    templates, logLam_template, logAge_grid, metal_grid, alpha_grid = (d[k] for k in
        ('templates', 'logLam_template', 'logAge_grid', 'metal_grid', 'alpha_grid'))
    ncomb, nAges, nMetal, nAlpha, gas = (d[k] for k in ('ncomb', 'nAges', 'nMetal', 'nAlpha', 'gas'))
    galaxy, logLam_galaxy, nbins, npix, noise, dv = (d[k] for k in
        ('galaxy', 'logLam_galaxy', 'nbins', 'npix', 'noise', 'dv'))
    velscale, velscale_ratio, fixed, start, goodPixels_sfh = (d[k] for k in
        ('velscale', 'velscale_ratio', 'fixed', 'start', 'goodPixels'))
    opts = d['opts']

    # REGUL_ERR: a number, or 'auto' = the Delta chi2 = sqrt(2 N_good) criterion in every bin
    regul_err = config['SFH']['REGUL_ERR']
    if isinstance(regul_err, str) and regul_err.strip().lower() == 'auto':
        regul_err = 'auto'
        logging.info("SFH: REGUL_ERR = auto: regularisation set per bin by Delta chi2 = sqrt(2 N_good) "
                     "(about 10x the fits of a fixed value)")
        printStatus.done("REGUL_ERR = auto: regularisation chosen per bin (Delta chi2 = sqrt(2 N_good))")
    else:
        regul_err = float(regul_err)
    regul_used = np.full(nbins, np.nan)

    # Define output arrays
    kin          = np.zeros((nbins,6    ))
    w_row        = np.zeros((nbins,ncomb))
    bestfit      = np.zeros((nbins,npix ))
    formal_error = np.zeros((nbins,6    ))
    if gas is not None:
        nl = len(gas['lines'])
        gas_res = dict(names=gas['lines'], sol=np.full((nbins, 2), np.nan), err=np.full((nbins, 2), np.nan),
                       flux=np.full((nbins, nl), np.nan), flux_err=np.full((nbins, nl), np.nan),
                       bestfit=np.zeros((nbins, npix)), av=np.full(nbins, np.nan), vratio=velscale_ratio)
    else:
        gas_res = None
    boot = np.full((nbins, 4), np.nan)

    def _store_gas(i, g):
        if gas_res is not None and g is not None:
            gas_res['sol'][i] = g['sol']; gas_res['err'][i] = g['err']
            gas_res['flux'][i] = g['flux']; gas_res['flux_err'][i] = g['flux_err']
            gas_res['bestfit'][i] = g['bestfit']
            gas_res['av'][i] = g.get('av', np.nan)

    # ====================
    # Run PPXF
    start_time = time.time()
    if config['GENERAL']['PARALLEL'] == True:
        printStatus.running("Running PPXF in parallel mode")            
        logging.info("Running PPXF in parallel mode")

        # Create Queues
        inQueue  = Queue()
        outQueue = Queue()
    
        # Create worker processes
        ps = [Process(target=workerPPXF, args=(inQueue, outQueue))
              for _ in range(config['GENERAL']['NCPU'])]
    
        # Start worker processes
        for p in ps: p.start()
    
        # Fill the queue
        for i in range(nbins):
            # use noise from var cube
            inQueue.put( ( templates, galaxy[i,:], noise.T[:,i], velscale, start[i,:], goodPixels_sfh, config['SFH']['MOM'], dv,\
                           config['SFH']['MDEG'], regul_err, fixed, velscale_ratio, npix,\
                           ncomb, nbins, i, gas, opts ) )
    
        # now get the results with indices
        ppxf_tmp = [outQueue.get() for _ in range(nbins)]
    
        # send stop signal to stop iteration
        for _ in range(config['GENERAL']['NCPU']): inQueue.put('STOP')

        # stop processes
        for p in ps: p.join()
    
        # Get output
        index = np.zeros(nbins)
        for i in range(0, nbins):
            index[i]                        = ppxf_tmp[i][0]
            kin[i,:config['SFH']['MOM']]    = ppxf_tmp[i][1]
            w_row[i,:]                      = ppxf_tmp[i][2]
            bestfit[i,:]                    = ppxf_tmp[i][3]
            formal_error[i,:config['SFH']['MOM']] = ppxf_tmp[i][4]
        # Sort output
        argidx = np.argsort( index )
        for i in range(0, nbins):
            _store_gas(int(ppxf_tmp[argidx[i]][0]), ppxf_tmp[argidx[i]][5])
            regul_used[int(ppxf_tmp[argidx[i]][0])] = ppxf_tmp[argidx[i]][6]
            boot[int(ppxf_tmp[argidx[i]][0])] = ppxf_tmp[argidx[i]][7]
        kin          = kin[argidx,:]
        w_row        = w_row[argidx,:]
        bestfit      = bestfit[argidx,:]
        formal_error = formal_error[argidx,:]

        printStatus.updateDone("Running PPXF in parallel mode", progressbar=True)

    if config['GENERAL']['PARALLEL'] == False:
        printStatus.running("Running PPXF in serial mode")            
        logging.info("Running PPXF in serial mode")
        for i in range(nbins):
            kin[i,:config['SFH']['MOM']], w_row[i,:], bestfit[i,:], formal_error[i,:config['SFH']['MOM']], g, regul_used[i], boot[i] = run_ppxf\
                (templates, galaxy[i,:], noise.T[:,i], velscale, start[i,:], goodPixels_sfh, config['SFH']['MOM'], dv, \
                config['SFH']['MDEG'], regul_err, fixed, velscale_ratio, npix, ncomb, nbins, i, gas, opts)
            _store_gas(i, g)
        printStatus.updateDone("Running PPXF in serial mode", progressbar=True)

    print("             Running PPXF on %s spectra took %.2fs using %i cores" % (nbins, time.time() - start_time, config['GENERAL']['NCPU']))
    logging.info("Running PPXF on %s spectra took %.2fs using %i cores" % (nbins, time.time() - start_time, config['GENERAL']['NCPU']))

    # Check for exceptions which occurred during the analysis
    idx_error = np.where( np.isnan( kin[:,0] ) == True )[0]
    if len(idx_error) != 0:
        printStatus.warning("There was a problem in the analysis of the spectra with the following BINID's: ")
        print("             "+str(idx_error))
        logging.warning("There was a problem in the analysis of the spectra with the following BINID's: "+str(idx_error))
    else:
        print("             "+"There were no problems in the analysis.")
        logging.info("There were no problems in the analysis.")
    print("")

    # Calculate mean age, metallicity and alpha
    mean_results = mean_agemetalalpha(w_row, 10**logAge_grid, metal_grid, alpha_grid, nbins)
    logage = np.array([_light_means(w_row[i], opts['grids'])[0] for i in range(nbins)])

    # Save to file
    save_sfh(mean_results, kin, formal_error, w_row, logAge_grid, metal_grid, alpha_grid, bestfit, logLam_galaxy, goodPixels_sfh, \
            velscale, logLam_galaxy, ncomb, nAges, nMetal, nAlpha, npix, config, gas_res, regul_used,
            logage=logage, boot=boot if opts['nboot'] else None)

    # Return
    return(None)


