import glob
import os
import re
import numpy as np
from astropy.io import fits
import logging

from printStatus import printStatus

from ppxf.ppxf_util import log_rebin, gaussian_filter1d


_NAME_Z = re.compile(r'Z([mp])(\d+\.\d+)')
_NAME_T = re.compile(r'T(\d+\.\d+)')
_NAME_A = re.compile(r'_E([mp])(\d\.\d\d)')


def parse_name(fn):
    """
    (age [Gyr], [M/H], [alpha/Fe], alpha token) from a (E-)MILES-style file name, e.g.
        Eun1.30Zp0.06T01.0000_iTp0.00_baseFe.fits        E-MILES, MILES-star abundances
        Eun1.30Zp0.06T01.0000_iTp0.00_Ep0.40.fits        E-MILES / MILES alpha-variable
        Mun1.30Zp0.06T01.0000_iPp0.00_Ep0.00.fits        MILES alpha-variable
        Eun1.30Zp0.06T00.0010_iTp0.00_baseFe_FSPS.fits   make_fsps_ssp.py
    baseFe (the abundance pattern of the MILES stars: ~solar at solar [M/H], alpha-
    enhanced at low [M/H]) is listed as alpha = 0.0 with token 'baseFe'.
    (The old parser took the leading 'E' of E-MILES names for the alpha tag.)
    """
    s = os.path.basename(fn)
    z, t = _NAME_Z.search(s), _NAME_T.search(s[_NAME_Z.search(s).end():] if _NAME_Z.search(s) else s)
    if z is None or t is None:
        raise ValueError("not a (E-)MILES file name (Z[mp]x.xxTxx.xxxx): %s" % s)
    metal = (-1 if z.group(1) == 'm' else 1) * float(z.group(2))
    age = float(t.group(1))
    a = _NAME_A.search(s)
    if a is not None:
        alpha = (-1 if a.group(1) == 'm' else 1) * float(a.group(2))
        tok = 'E%s%s' % (a.group(1), a.group(2))
    elif 'baseFe' in s:
        alpha, tok = 0.0, 'baseFe'
    else:
        raise ValueError("no alpha tag (_Ep0.00 / _Ep0.40 / baseFe) in the template name %s" % s)
    return age, metal, alpha, tok


def age_metal_alpha(passedFiles):
    """
    Ages, metallicities and alpha-enhancements of the template files (see parse_name):
    returns (log10 ages, metals, alphas, per-file (iAge, iMetal, iAlpha) indices, nAges,
    nMetal, nAlpha, ncomb). The library must be a COMPLETE grid -- every age at every
    [M/H] and [alpha/Fe], each once -- and may not mix baseFe with Ep/Em files (baseFe
    is not [alpha/Fe] = 0; mixing would mislabel the grid).
    """
    par = [parse_name(f) for f in passedFiles]
    toks = set(p[3] for p in par)
    if 'baseFe' in toks and len(toks) > 1:
        raise ValueError("the template folder mixes baseFe and alpha-variable (%s) files: use one kind"
                         % ', '.join(sorted(toks - {'baseFe'})))
    Age = np.unique([p[0] for p in par])
    Metal = np.unique([p[1] for p in par])
    Alpha = np.unique([p[2] for p in par])
    nAges, nMetal, nAlpha = len(Age), len(Metal), len(Alpha)
    idx = [(int(np.searchsorted(Age, p[0])), int(np.searchsorted(Metal, p[1])), int(np.searchsorted(Alpha, p[2])))
           for p in par]
    seen = {}
    for f, ijk in zip(passedFiles, idx):
        if ijk in seen:
            raise ValueError("two templates for the same age/[M/H]/[alpha/Fe]: %s and %s"
                             % (os.path.basename(seen[ijk]), os.path.basename(f)))
        seen[ijk] = f
    ncomb = nAges * nMetal * nAlpha
    if len(seen) != ncomb:
        miss = [(Age[i], Metal[k], Alpha[j]) for i in range(nAges) for k in range(nMetal) for j in range(nAlpha)
                if (i, k, j) not in seen]
        raise ValueError("incomplete template grid: %d of %d age x [M/H] x [alpha/Fe] combinations missing, "
                         "e.g. %s" % (len(miss), ncomb, ', '.join('%g Gyr/%+.2f/%+.2f' % m for m in miss[:5])))
    return np.log10(Age), Metal, Alpha, idx, nAges, nMetal, nAlpha, ncomb


def prepareSpectralTemplateLibrary(config, lmin, lmax, velscale, LSF_Data, LSF_Templates, sortInGrid):
    """
    Prepares the spectral template library. The templates are loaded from disk,
    shortened to meet the spectral range in consideration, convolved to meet the
    resolution of the observed spectra (according to the LSF), log-rebinned, and
    normalised. In addition, they are sorted in a three-dimensional array
    sampling the parameter space in age, metallicity and alpha-enhancement. 
    """
    printStatus.running("Preparing the stellar population templates")
    cvel  = 299792.458

    # SSP model library
    sp_models = glob.glob(os.path.join(config['GENERAL']['TEMPLATE_DIR'],config['PREPARE_TEMPLATES']['LIBRARY'])+'*.fits')
    sp_models.sort()
    ntemplates = len(sp_models)

    # Read data
    hdu_spmod      = fits.open(sp_models[0])
    ssp_data       = np.squeeze(hdu_spmod[0].data)
    ssp_head       = hdu_spmod[0].header
    lamRange_spmod = ssp_head['CRVAL1'] + np.array([0., ssp_head['CDELT1']*(ssp_head['NAXIS1'] - 1)])

    # Determine length of templates
    template_overhead = np.zeros(2)
    if lmin - lamRange_spmod[0] > 150.:
        template_overhead[0] = 150.
    else: 
        template_overhead[0] = lmin - lamRange_spmod[0] - 5
    if lamRange_spmod[1] - lmax > 150.:
        template_overhead[1] = 150.
    else: 
        template_overhead[1] = lamRange_spmod[1] - lmax - 5

    # Shorten templates to size of data
    # Reconstruct full original lamRange
    lamRange_lin = np.arange( lamRange_spmod[0], lamRange_spmod[-1]+ssp_head['CDELT1'], ssp_head['CDELT1'] )
    # Create new lamRange according to the provided LMIN and LMAX values, according to the module which calls
    constr = np.array([ lmin - template_overhead[0], lmax + template_overhead[1] ])
    idx_lam = np.where( np.logical_and(lamRange_lin > constr[0], lamRange_lin < constr[1] ) )[0]
    lamRange_spmod = np.array([ lamRange_lin[idx_lam[0]], lamRange_lin[idx_lam[-1]] ])
    # Shorten data to size of new lamRange
    ssp_data = ssp_data[idx_lam]

    # Convolve templates to same resolution as data
    if len( np.where( LSF_Data(lamRange_lin[idx_lam]) - LSF_Templates(lamRange_lin[idx_lam]) < 0. )[0] ) != 0:
        message = "According to the specified LSF's, the resolution of the "+\
                  "templates is lower than the resolution of the data. Exit!"
        printStatus.updateFailed("Preparing the stellar population templates")
        print("             "+message)
        logging.critical(message)
        exit(1)
    else:
        FWHM_dif = np.sqrt( LSF_Data(lamRange_lin[idx_lam])**2 - LSF_Templates(lamRange_lin[idx_lam])**2 )
        sigma = FWHM_dif/2.355/ssp_head['CDELT1']

    # Create an array to store the templates
    sspNew, _, _ = log_rebin(lamRange_spmod, ssp_data, velscale=velscale)


    # Do NOT sort the templates in any way
    if sortInGrid == False:

        # Load templates, convolve and log-rebin them
        templates = np.empty((sspNew.size, ntemplates))
        for j, file in enumerate(sp_models):
            hdu      = fits.open(file)
            ssp_data = np.squeeze(hdu[0].data)[idx_lam]
            ssp_data = gaussian_filter1d(ssp_data, sigma)
            templates[:, j], logLam_spmod, _ = log_rebin(lamRange_spmod, ssp_data, velscale=velscale)
   
        # Normalise templates in such a way to get mass-weighted results
        if config['PREPARE_TEMPLATES']['NORM_TEMP'] == 'MASS':
            templates = templates / np.mean( templates )
    
        # Normalise templates in such a way to get light-weighted results
        if config['PREPARE_TEMPLATES']['NORM_TEMP'] == 'LIGHT':
            for i in range( templates.shape[1] ):
                templates[:,i] = templates[:,i] / np.mean(templates[:,i], axis=0)
    
        printStatus.updateDone("Preparing the stellar population templates")
        logging.info("Prepared the stellar population templates")
    
        return( templates, [lamRange_spmod[0],lamRange_spmod[1]], logLam_spmod, ntemplates, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan )

    
    # Sort the templates in a cube of age, metal, alpha for the SFH module
    elif sortInGrid == True:

        # Extract ages, metallicities and alpha from the templates
        logAge, metal, alpha, idx, nAges, nMetal, nAlpha, ncomb = age_metal_alpha(sp_models)

        templates          = np.zeros((sspNew.size, nAges, nMetal, nAlpha))
        templates[:,:,:,:] = np.nan
    
        # Arrays to store properties of the models
        logAge_grid = np.empty((nAges, nMetal, nAlpha))
        metal_grid  = np.empty((nAges, nMetal, nAlpha))
        alpha_grid  = np.empty((nAges, nMetal, nAlpha))
    
        # Sort the templates in the cube of age, metal, alpha (by the values in the
        # file names, not by the order of the files)
        for filename, (j, k, i) in zip(sp_models, idx):
            hdu = fits.open(filename)
            h = hdu[0].header
            if (h['CRVAL1'], h['CDELT1'], h['NAXIS1']) != (ssp_head['CRVAL1'], ssp_head['CDELT1'], ssp_head['NAXIS1']):
                raise ValueError("%s is not on the wavelength grid of %s (CRVAL1/CDELT1/NAXIS1): all templates "
                                 "of a library must share one grid" % (os.path.basename(filename),
                                                                       os.path.basename(sp_models[0])))
            ssp = np.squeeze(hdu[0].data)[idx_lam]
            ssp = gaussian_filter1d(ssp, sigma)
            sspNew, logLam2, _ = log_rebin(lamRange_spmod, ssp, velscale=velscale)

            logAge_grid[j, k, i] = logAge[j]
            metal_grid[j, k, i]  = metal[k]
            alpha_grid[j, k, i]  = alpha[i]

            # Normalise templates for light-weighted results
            if config['PREPARE_TEMPLATES']['NORM_TEMP'] == 'LIGHT':
                templates[:, j, k, i] = sspNew / np.mean(sspNew)
            else:
                templates[:, j, k, i] = sspNew 
    
        # Normalise templates for mass-weighted results
        if config['PREPARE_TEMPLATES']['NORM_TEMP'] == 'MASS':
            templates = templates / np.mean( templates )

        printStatus.updateDone("Preparing the stellar population templates")
        logging.info("Prepared the stellar population templates")
    
        return(templates, [lamRange_spmod[0],lamRange_spmod[1]], logLam2, ntemplates, logAge_grid, metal_grid, alpha_grid, ncomb, nAges, nMetal, nAlpha)


