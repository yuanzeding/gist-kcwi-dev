from astropy.io import fits
import numpy as np

import os
import logging

from printStatus import printStatus

from ppxf.ppxf_util import log_rebin
from scipy import sparse

# Spaxels are log-rebinned in blocks of this many, as one 2-D call to
# log_rebin (it accepts an array of spectra arranged as columns).
LOGREBIN_CHUNK = 1000


def prepSpectra(config, cube):
    """
    This function performs the following tasks:
     * Apply spatial bins to linear spectra; Save these spectra to disk
     * Log-rebin all spectra, regardless of whether the spaxels are masked or not; Save all spectra to disk
     * Apply spatial bins to log-rebinned spectra; Save these spectra to disk
    """

    # Read maskfile
    maskfile = os.path.join(config['GENERAL']['OUTPUT'], config['GENERAL']['RUN_ID']) + "_mask.fits"
    mask = fits.open(maskfile)[1].data.MASK
    idxUnmasked = np.where( mask == 0 )[0]
    idxMasked   = np.where( mask == 1 )[0]

    # Read binning pattern
    tablefile = os.path.join(config['GENERAL']['OUTPUT'], config['GENERAL']['RUN_ID']) + "_table.fits"
    binNum = fits.open(tablefile)[1].data.BIN_ID[idxUnmasked]

    # Apply spatial bins to linear spectra
    bin_data, bin_error, bin_flux = applySpatialBins(binNum, cube['spec'][:,idxUnmasked], cube['error'][:,idxUnmasked], config['PREPARE_SPECTRA']['VELSCALE'], "lin" )
    # Save spatially binned spectra
    saveBinSpectra(config, bin_data, bin_error, config['PREPARE_SPECTRA']['VELSCALE'], cube['wave'], "lin")

    # Log-rebin spectra
    log_spec, log_error, logLam = log_rebinning(config, cube)
    # Save all log-rebinned spectra
    saveAllSpectra(config, log_spec, log_error, config['PREPARE_SPECTRA']['VELSCALE'], logLam)

    # Apply bins to log spectra
    bin_data, bin_error, bin_flux = applySpatialBins(binNum, log_spec[:,idxUnmasked], log_error[:,idxUnmasked], config['PREPARE_SPECTRA']['VELSCALE'], "log" )
    # Save spatially binned spectra
    saveBinSpectra(config, bin_data, bin_error, config['PREPARE_SPECTRA']['VELSCALE'], logLam, "log")

    return(None)


def _check_wave(wave):
    """
    log_rebin needs to know the wavelength of every pixel. A [min, max] pair
    is only correct for a uniformly sampled spectrum, which is what this module
    used to pass regardless; for a spectrum with a gap (e.g. two concatenated
    channels) that silently shifted every pixel after the gap. Pass the full
    vector instead, and say so if the grid is not uniform.
    """
    wave = np.asarray(wave, dtype=float)
    dw = np.diff(wave)
    if np.any(dw <= 0):
        raise ValueError("cube['wave'] must increase monotonically")
    if not np.allclose(dw, np.median(dw), rtol=1e-4):
        jump = np.argmax(np.abs(dw - np.median(dw)))
        msg = ("The input wavelength grid is not uniform (largest step %.2f A at %.1f A, median "
               "%.3f A). Log-rebinning with the full wavelength vector; pixels in any gap will be "
               "empty -- make sure they are masked." % (dw[jump], wave[jump], np.median(dw)))
        printStatus.warning(msg)
        logging.warning(msg)
    return wave


def log_rebinning(config, cube):
    """
    Logarithmically rebin spectra and error spectra.

    cube['error'] holds VARIANCES. log_rebin returns flux density, i.e. a
    weighted average f_out = sum_i w_i f_i with sum_i w_i = 1 (w_i = overlap of
    input pixel i / width of the output pixel). Applying it to the variance
    gives sum_i w_i var_i, but the variance of f_out is sum_i w_i**2 var_i,
    which is smaller: by ~2.5x at 5000 A and ~4x at 8900 A for 0.5 A input
    pixels and VELSCALE = 60 km/s. The errors handed to pPXF were inflated by
    that factor's square root, increasingly to the red. The variance is now
    propagated with the squared weights (rebin_variance). Neighbouring output
    pixels share input pixels and stay correlated; a per-pixel error cannot
    represent that, as before.
    """
    velscale = config['PREPARE_SPECTRA']['VELSCALE']
    wave = _check_wave(cube['wave'])

    # Log-rebin the spectra
    printStatus.running("Log-rebinning the spectra")
    log_spec, logLam = run_logrebinning(cube['spec'], velscale, len(cube['x']), wave)
    printStatus.updateDone("Log-rebinning the spectra", progressbar=True)
    logging.info("Log-rebinned the spectra")

    # Log-rebin the error spectra (variances)
    printStatus.running("Log-rebinning the error spectra")
    log_error = rebin_variance(cube['error'], wave, velscale, logLam)
    printStatus.updateDone("Log-rebinning the error spectra", progressbar=True)
    logging.info("Log-rebinned the error spectra")

    return(log_spec, log_error, logLam)


def logrebin_weights(wave, velscale):
    """
    Sparse matrix W (n_out x n_in) with f_out = W @ f_in reproducing
    ppxf_util.log_rebin(wave, f_in, velscale=velscale) (flux=False) when
    `wave` is the full vector of pixel centres: same input borders (midpoints),
    same output borders.
    """
    c = 299792.458
    wave = np.asarray(wave, dtype=float)
    lim = 1.5*wave[[0, -1]] - 0.5*wave[[1, -2]]
    borders = np.hstack([lim[0], 0.5*(wave[1:] + wave[:-1]), lim[1]])
    ln_lim = np.log(lim)
    m = int(((ln_lim[1] - ln_lim[0])/(velscale/c)))
    newBorders = np.exp(ln_lim[0] + velscale/c*np.arange(m + 1))
    # every interval between consecutive borders of either grid lies in exactly
    # one input pixel and (if inside the output range) one output pixel
    edges = np.unique(np.hstack([borders, newBorders]))
    edges = edges[(edges >= newBorders[0]) & (edges <= newBorders[-1])]
    mid = 0.5*(edges[1:] + edges[:-1])
    i_in = np.clip(np.searchsorted(borders, mid) - 1, 0, wave.size - 1)
    i_out = np.clip(np.searchsorted(newBorders, mid) - 1, 0, m - 1)
    w = np.diff(edges) / np.diff(newBorders)[i_out]
    return sparse.csr_matrix((w, (i_out, i_in)), shape=(m, wave.size))


def rebin_variance(var, wave, velscale, logLam):
    """ Variance of the log-rebinned flux: (W**2) @ var. `var` is (n_in, nspec). """
    W = logrebin_weights(wave, velscale)
    if W.shape[0] != len(logLam):
        raise ValueError("log-rebinning weights do not match the log_rebin output (%d vs %d pixels)" % (W.shape[0], len(logLam)))
    W2 = W.multiply(W).tocsr()
    out = np.zeros((W.shape[0], var.shape[1]))
    for i0 in range(0, var.shape[1], LOGREBIN_CHUNK):
        i1 = min(var.shape[1], i0 + LOGREBIN_CHUNK)
        out[:, i0:i1] = W2 @ np.asarray(var[:, i0:i1], dtype=float)
    return out


def run_logrebinning( bin_data, velscale, nbins, wave ):
    """
    Calls the log-rebinning routine of pPXF (see Cappellari & Emsellem 2004;
    ui.adsabs.harvard.edu/?#abs/2004PASP..116..138C;
    ui.adsabs.harvard.edu/?#abs/2017MNRAS.466..798C).

    `wave` is the wavelength of every input pixel (not just [min, max]).
    """
    _, logLam, _ = log_rebin(wave, bin_data[:, 0], velscale=velscale)
    log_bin_data = np.zeros([len(logLam), nbins])

    for i0 in range(0, nbins, LOGREBIN_CHUNK):
        i1 = min(nbins, i0 + LOGREBIN_CHUNK)
        log_bin_data[:, i0:i1] = log_rebin(wave, bin_data[:, i0:i1], velscale=velscale)[0]
        printStatus.progressBar(i1, nbins, barLength = 50)

    # log_rebin is local (np.add.reduceat), so a NaN only affects the output
    # pixels that overlap it; report them instead of hiding them
    nbad = np.count_nonzero(~np.isfinite(log_bin_data).all(axis=0))
    if nbad > 0:
        logging.warning("%d of %d spectra contain non-finite values after log-rebinning" % (nbad, nbins))

    return(log_bin_data, logLam)


def saveAllSpectra(config, log_spec, log_error, velscale, logLam):
    """
    Save all logarithmically rebinned spectra to file.
    NOTE: here ESPEC holds the VARIANCE of each spaxel; in _BinSpectra.fits it
    holds the standard deviation.
    """
    outfits_spectra = os.path.join(config['GENERAL']['OUTPUT'], config['GENERAL']['RUN_ID']) + '_AllSpectra.fits'
    printStatus.running("Writing: "+config['GENERAL']['RUN_ID']+'_AllSpectra.fits')

    # Primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU for spectra
    cols = []
    cols.append( fits.Column(name='SPEC',   format=str(len(log_spec))+'D', array=log_spec.T  ))
    cols.append( fits.Column(name='ESPEC',  format=str(len(log_spec))+'D', array=log_error.T ))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'SPECTRA'

    # Table HDU for LOGLAM
    cols = []
    cols.append( fits.Column(name='LOGLAM', format='D', array=logLam ))
    loglamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    loglamHDU.name = 'LOGLAM'

    # Create HDU List and save to file
    HDUList = fits.HDUList([priHDU, dataHDU, loglamHDU])
    HDUList.writeto(outfits_spectra, overwrite=True)

    # Set header keywords
    fits.setval(outfits_spectra,'VELSCALE', value=velscale)
    fits.setval(outfits_spectra,'CRPIX1',   value=1.0)
    fits.setval(outfits_spectra,'CRVAL1',   value=logLam[0])
    fits.setval(outfits_spectra,'CDELT1',   value=logLam[1]-logLam[0])

    printStatus.updateDone("Writing: "+config['GENERAL']['RUN_ID']+'_AllSpectra.fits')
    logging.info("Wrote: "+outfits_spectra)


def saveBinSpectra(config, log_spec, log_error, velscale, logLam, flag):
    """ Save spatially binned spectra and error spectra are saved to disk. """
    outfile = os.path.join(config['GENERAL']['OUTPUT'], config['GENERAL']['RUN_ID'])

    if flag == 'log':
        outfits_spectra = outfile+'_BinSpectra.fits'
        printStatus.running("Writing: "+config['GENERAL']['RUN_ID']+'_BinSpectra.fits')
    elif flag == 'lin':
        outfits_spectra  = outfile+'_BinSpectra_linear.fits'
        printStatus.running("Writing: "+config['GENERAL']['RUN_ID']+'_BinSpectra_linear.fits')

    npix = len(log_spec)

    # Create primary HDU
    priHDU = fits.PrimaryHDU()

    # Table HDU for spectra
    cols = []
    cols.append( fits.Column(name='SPEC',  format=str(npix)+'D', array=log_spec.T  ))
    cols.append( fits.Column(name='ESPEC', format=str(npix)+'D', array=log_error.T ))
    dataHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    dataHDU.name = 'BIN_SPECTRA'

    # Table HDU for LOGLAM
    cols = []
    cols.append( fits.Column(name='LOGLAM', format='D', array=logLam ))
    loglamHDU = fits.BinTableHDU.from_columns(fits.ColDefs(cols))
    loglamHDU.name = 'LOGLAM'

    # Create HDU list and save to file
    HDUList = fits.HDUList([priHDU, dataHDU, loglamHDU])
    HDUList.writeto(outfits_spectra, overwrite=True)

    # Set header values
    fits.setval(outfits_spectra,'VELSCALE',value=velscale)
    fits.setval(outfits_spectra,'CRPIX1',  value=1.0)
    fits.setval(outfits_spectra,'CRVAL1',  value=logLam[0])
    fits.setval(outfits_spectra,'CDELT1',  value=logLam[1]-logLam[0])

    if flag == 'log':
        printStatus.updateDone("Writing: "+config['GENERAL']['RUN_ID']+'_BinSpectra.fits')
    elif flag == 'lin':
        printStatus.updateDone("Writing: "+config['GENERAL']['RUN_ID']+'_BinSpectra_linear.fits')
    logging.info("Wrote: "+outfits_spectra)


def applySpatialBins(binNum, spec, espec, velscale, flag):
    """
    The constructed spatial binning scheme is applied to the spectra.
    """
    printStatus.running("Applying the spatial bins to "+flag+"-data")
    bin_data, bin_error, bin_flux = spatialBinning( binNum, spec, espec )
    printStatus.updateDone("Applying the spatial bins to "+flag+"-data", progressbar=True)
    logging.info("Applied spatial bins to "+flag+"-data")

    return(bin_data, bin_error, bin_flux)


def spatialBinning( binNum, spec, error ):
    """
    Spectra belonging to the same spatial bin are added. `error` holds
    variances; the returned bin_error is the standard deviation.
    Flux and variance are summed the same way (np.sum): the old nansum on the
    flux and sum on the variance treated a NaN differently in the two.
    """
    ubins     = np.unique(binNum)
    nbins     = len(ubins)
    npix      = spec.shape[0]
    bin_data  = np.zeros([npix,nbins])
    bin_error = np.zeros([npix,nbins])
    bin_flux  = np.zeros(nbins)

    for i in range(nbins):
        k = np.where( binNum == ubins[i] )[0]
        av_spec     = np.sum(spec[:,k],axis=1)
        av_err_spec = np.sqrt(np.sum(error[:,k],axis=1))

        bin_data[:,i]  = np.ravel(av_spec)
        bin_error[:,i] = np.ravel(av_err_spec)
        bin_flux[i]    = np.mean(av_spec,axis=0)
        printStatus.progressBar(i+1, nbins, barLength = 50)

    return(bin_data, bin_error, bin_flux)


def corefunc_logrebin(lamRange, bin_data, velscale, npix, iterate, nbins):
    """
    Kept for backward compatibility. `lamRange` must be the wavelength of
    every pixel unless the spectrum is uniformly sampled. Errors are no longer
    swallowed (the old bare except returned an all-NaN spectrum silently).
    """
    sspNew, logLam, _ = log_rebin(lamRange, bin_data, velscale=velscale)
    printStatus.progressBar(iterate+1, nbins, barLength = 50)
    return(sspNew)
