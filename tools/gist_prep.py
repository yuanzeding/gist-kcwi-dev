#!/usr/bin/env python
"""
gist_prep.py -- build the spectral masks and a MasterConfig row for a KCWI
blue+red GIST run (readData method KECK_KCWIbpr).

What it writes (in --configdir, names end in _<tag>):
  specMask_KIN_<tag>         rest-frame masks: emission lines, dichroic gap, sky, telluric
  specMask_SFH_<tag>         same content (separate file so the two can be edited apart)
  emissionLines_<tag>.config copy of --emi with the gap / sky / telluric added as
                             'm' (mask) entries, and fitted lines that fall in the gap or
                             outside the data switched to 'i' (ignore)
  MasterConfig_<tag>         --template with one run row filled in (or appended, --append)
  masks_<tag>.png            optional (--plot): integrated spectrum with the masks shaded

FRAMES. GIST masks are REST-FRAME, with one exception: a specMask entry whose
comment is exactly 'sky' (and an emissionLines entry named 'sky') is taken as
OBSERVED and its centre -- but not its width -- is divided by (1+z). To avoid
that half-conversion, this script converts every observed-frame feature (sky
lines, telluric bands, the dichroic gap) to the rest frame itself and labels it
e.g. 'sky5577' / 'gap' / 'telluricA', which GIST treats as rest-frame.

THE GAP. Taken from the same code the reader uses (gistPipeline.readData.
KECK_KCWIbpr._read_channel, so plane trimming from the mask cube is identical)
when that is importable; otherwise from the WAVGOOD header keywords (--header-only
forces this; the reader logs the exact gap, compare). The gap is masked because
it carries no data: KIN/SFH would drop it anyway (infinite variance), but GAS
(gandalf, noise = 1) would fit the interpolated flux.

PARAMETER FILE. Everything can go in one file instead of on the command line:
  python gist_prep.py --example > prep_v3.in     # commented template; edit it
  python gist_prep.py prep_v3.in
  python gist_prep.py prep_v3.in --tag v3b --set KIN.MOM=2   # command line overrides the file
[prep] holds this script's options (names as below, '_' or '-'); every other
[SECTION] is a MasterConfig module, its KEY = VALUE lines go into the new run
row -- e.g. [SPATIAL_BINNING] METHOD = hierarchical. The settings actually used
are saved as <configdir>/gist_prep_<tag>.in, which reproduces the run.

Examples
  # from the target directory (the one holding configFiles/, inputData/, results/)
  python gist_prep.py --input inputData/IRAS03582+6012 --z 0.030 --tag v2 \\
      --emi configFiles/emissionLines.config --template configFiles/MasterConfig \\
      --run-id IRAS03582bin35_v2 --set SPATIAL_BINNING.TARGET_SNR=35 --plot

  # type-1 nucleus: broad Balmer lines masked at +-3000 km/s
  python gist_prep.py ... --broad Hb,Ha,Hg,Hd --broad-hw 3000

  # extra hand-picked rest-frame intervals, and drop NaD from the defaults
  python gist_prep.py ... --extra 8432-8459,8567-8631 --drop NaD
"""
from __future__ import print_function

import argparse
import configparser
import copy
import math
import os
import re
import sys
import warnings

import numpy as np
from astropy.io import fits

C = 299792.458

# Rest-frame AIR wavelengths [A] of lines masked by default (name, lambda)
LINES = [
    ('HeII', 3203.10), ('[NeV]', 3345.82), ('[NeV]', 3425.88), ('[OII]', 3726.03), ('[OII]', 3728.82),
    ('H9', 3835.39), ('[NeIII]', 3868.76), ('H8', 3889.05), ('[NeIII]', 3967.47), ('He', 3970.07),
    ('[SII]', 4068.60), ('[SII]', 4076.35), ('Hd', 4101.74), ('Hg', 4340.47), ('[OIII]', 4363.21),
    ('HeI', 4471.48), ('HeII', 4685.71), ('[ArIV]', 4711.37), ('[ArIV]', 4740.20), ('Hb', 4861.33),
    ('[OIII]', 4958.91), ('[OIII]', 5006.84), ('[NI]', 5197.90), ('[NI]', 5200.26), ('[FeVII]', 5720.70),
    ('[NII]', 5754.64), ('HeI', 5875.62), ('NaD', 5889.95), ('NaD', 5895.92), ('[FeVII]', 6087.00),
    ('[OI]', 6300.30), ('[SIII]', 6312.06), ('[OI]', 6363.78), ('[FeX]', 6374.51), ('[NII]', 6548.05),
    ('Ha', 6562.80), ('[NII]', 6583.45), ('HeI', 6678.15), ('[SII]', 6716.44), ('[SII]', 6730.82),
    ('HeI', 7065.19), ('[ArIII]', 7135.80), ('[OII]', 7319.90), ('[OII]', 7330.20), ('[ArIII]', 7751.06),
    ('[SIII]', 9068.60), ('[SIII]', 9530.60),
]

# OBSERVED-frame sky emission lines [A] that leave residuals in KCWI cubes:
# [OI], Na I (lamps/airglow) and the strongest OH lines. Extend with --sky-file.
SKY = [5577.34, 5889.95, 5895.92, 6300.30, 6363.78, 6498.7, 6863.96, 7276.4, 7316.3, 7340.9,
       7750.6, 7794.1, 7821.5, 7913.7, 7993.3, 8344.6, 8399.2, 8430.2, 8465.4, 8827.1,
       8885.8, 8919.6, 8943.4, 8958.1, 9001.4, 9375.9, 9439.7]

# OBSERVED-frame telluric absorption bands [A] (not corrected by the KCWI DRP)
TELLURIC = [('telluricB', 6865., 6930.), ('telluricA', 7590., 7710.)]


# ---------------------------------------------------------------------------
# Built-in MasterConfig: every GIST key, its default and what it does. This is the
# single source for the generated MasterConfig and for --example. 'auto' values are
# filled in by gist_prep (from the data or the [prep] options); 'None' = unset (the
# module's own default). Values must be one word (GIST's MasterConfig format).
# ---------------------------------------------------------------------------
MODULES = [
    ('GENERAL', [
        ('RUN_ID', 'auto', 'run name: results/ subfolder and file prefix; auto = <input name>_<tag>'),
        ('INPUT', 'auto', 'cube name in inputData/; auto = from [prep] input'),
        ('OUTPUT', '.', 'output folder, relative to outputDir in configFiles/defaultDir'),
        ('REDSHIFT', 'auto', 'auto = [prep] z'),
        ('PARALLEL', 'True', 'multiprocessing'),
        ('NCPU', '24', 'worker processes'),
        ('LSF_DATA', 'lsf_KCWI_BR', 'line-spread function of the data (in configFiles/)'),
        ('LSF_TEMP', 'lsf_XSL', 'line-spread function of the templates'),
        ('OW_CONFIG', 'True', 'ignore the CONFIG saved by an earlier run of this RUN_ID'),
        ('OW_OUTPUT', 'True', 'overwrite outputs (partial re-runs: run_gist.py --redo)'),
    ]),
    ('READ_DATA', [
        ('METHOD', 'KECK_KCWIbpr', 'reader in gistPipeline/readData/'),
        ('DEBUG', 'False', 'True: one central row of spaxels only'),
        ('ORIGIN', 'auto', 'centre x,y [pixels, 0-based]; auto = continuum peak in the S/N window; '
                           'radec:RA,DEC [deg] = that position via the WCS'),
        ('SUFFIX_BLUE', 'auto', 'auto = [prep] suffix_blue'),
        ('SUFFIX_RED', 'auto', 'auto = [prep] suffix_red'),
        ('LMIN_TOT', 'auto', 'rest-frame range kept [A]; auto = all the data'),
        ('LMAX_TOT', 'auto', ''),
        ('LMIN_SNR', 'auto', 'rest-frame continuum window for the S/N [A]; auto = the cleanest '
                             '(channel: [prep] snr_auto)'),
        ('LMAX_SNR', 'auto', ''),
        ('SNR_MASK', 'auto', 'lines/sky left out of the S/N; auto = specMask_SNR_<tag> (None with no_snr_mask)'),
        ('EBV', 'auto', 'Galactic E(B-V) [mag] removed by the reader (Fitzpatrick 1999, R_V 3.1) before every '
                        'module; auto = SFD x 0.86 at the cube centre ([prep] ebv_dir, ebv_scale); None = none'),
        ('EXT_LAW', 'f99', 'extinction curve for EBV: f99 (Fitzpatrick 1999, the law the SFD x 0.86 '
                           'recalibration assumes), ccm89 (Cardelli+ 1989) or od94 (O\'Donnell 1994)'),
    ]),
    ('SPATIAL_MASKING', [
        ('METHOD', 'default', ''),
        ('MIN_SNR', '0.5', 'spaxels outside the isophote of this mean S/N are masked'),
        ('MASK', 'False', 'file with a spatial mask, or False'),
    ]),
    ('SPATIAL_BINNING', [
        ('METHOD', 'hierarchical', 'hierarchical or voronoi (the keys below SB_TOL: hierarchical only)'),
        ('TARGET_SNR', '35', ''),
        ('COVARIANCE', '0.5', 'S/N /= COV_C (1 + COVARIANCE log10 N), N capped at COV_NMAX; measure with '
                              'kcwi_covar (mock-noise stacks) or gist_snrcheck (also used by the spectral test)'),
        ('COV_C', '1.0', 'noise of ONE spaxel / the variance cube (kcwi_covar measures it); 1 = trust the cube'),
        ('COV_NMAX', 'None', 'N beyond which the noise stops growing (correlation length reached; kcwi_covar); '
                             'None = never'),
        ('COV_FILE', 'None', 'measured curve N, ratio (kcwi_covar measure writes covar_<name>.txt; copy it to '
                             'configFiles/): used INSTEAD of COVARIANCE/COV_C/COV_NMAX'),
        ('SB_TOL', '0.1', 'SB difference allowed in a merge [dex]; smaller = closer to isophotes'),
        ('SB_RMS', 'None', 'max rms of log SB inside a bin [dex]; None = SB_TOL/2, 0 = off'),
        ('COMPACT', '0.3', 'weight of the compactness penalty'),
        ('MAX_ELONG', '2.5', 'max rms radius / that of a disc of the same area'),
        ('MAX_DPA', '0', 'max position-angle span seen from ORIGIN [deg]; 0 = off (45-60 for KIN)'),
        ('CORE_RADIUS', '1.0', '[arcsec] MAX_DPA not applied inside this radius'),
        ('SB_FWHM', '3', 'smoothing of the SB map [spaxels]'),
        ('MIN_SNR_FRAC', '0.5', 'leftovers below this x TARGET_SNR are absorbed by a neighbour'),
        ('SPEC_Z', '0', '> 0 (e.g. 3): refuse merges whose spectra differ beyond the noise'),
        ('SPEC_WEIGHT', '0', '> 0: spectral difference also added to the merge cost'),
        ('SPEC_LMIN', 'None', 'rest-frame range of the spectral term; None = whole spectrum'),
        ('SPEC_LMAX', 'None', ''),
        ('SPEC_FEATURES', 'None', 'stellar features compared: stellar (= CaHK,Hd,Gband,Fe4383,Hb,Mgb,'
                                  'Fe52,CaT) or a list of those (+NaD); None = SPEC_WINDOWS / whole range'),
        ('SPEC_MAXMASK', '0.5', 'a feature/window more than this fraction masked (sky, lines) or '
                                'without data is left out'),
        ('SPEC_WINDOWS', 'None', 'extra windows, e.g. 3920-3990,4080-4120 (one word); with neither '
                                 'this nor SPEC_FEATURES: the range in SPEC_NCHUNK chunks'),
        ('SPEC_DLAM', '10', 'chunk width inside SPEC_WINDOWS [A]'),
        ('SPEC_NCHUNK', '48', 'chunks when SPEC_WINDOWS is None'),
        ('SPEC_NORM', 'None', 'local = each feature by its own total (default with SPEC_FEATURES); '
                              'None = the S/N window with SNR_MASK; total; or windows'),
        ('SPEC_MASK', 'None', 'specMask left out of the spectral term; None = the KIN mask; False = no mask'),
    ]),
    ('PREPARE_SPECTRA', [
        ('METHOD', 'default', ''),
        ('VELSCALE', '40', 'sampling of the log-rebinned spectra [km/s]'),
    ]),
    ('PREPARE_TEMPLATES', [
        ('METHOD', 'xsl_lin', 'template reader in prepareTemplates/'),
        ('LIBRARY', 'XSL/', 'template folder, relative to templateDir'),
        ('NORM_TEMP', 'LIGHT', 'LIGHT or MASS weighted results'),
    ]),
    ('KIN', [
        ('METHOD', 'ppxf', 'False = off'),
        ('SPEC_MASK', 'auto', 'auto = specMask_KIN_<tag> (written by gist_prep)'),
        ('LMIN', '8400', 'rest-frame fit range [A] (clipped to the data)'),
        ('LMAX', '8600', ''),
        ('SIGMA', '100', 'initial velocity dispersion [km/s]'),
        ('MOM', '4', 'kinematic moments'),
        ('ADEG', '-1', 'additive Legendre degree (-1 = none)'),
        ('MDEG', '4', 'multiplicative Legendre degree (0 = none)'),
        ('REDDENING', '0.0878', 'initial E(B-V) to fit the stellar reddening, or None'),
        ('MC_PPXF', '0', 'Monte-Carlo realisations for the errors'),
    ]),
    ('GAS', [
        ('METHOD', 'False', 'gandalf, or False = off'),
        ('LEVEL', 'BIN', 'BIN, SPAXEL or BOTH'),
        ('LMIN', '4500', 'rest-frame range [A] (clipped to the data)'),
        ('LMAX', '8674', ''),
        ('ERRORS', '0', ''),
        ('REDDENING', '0.1,0.1', 'dust in the GandALF fit (no spaces), or False'),
        ('EBmV', 'None', 'Galactic E(B-V) for GandALF only -- leave None when READ_DATA EBV is set (no double '
                         'correction)'),
        ('EMI_FILE', 'auto', 'auto = emissionLines_<tag>.config from [prep] emi (None without it)'),
    ]),
    ('SFH', [
        ('METHOD', 'ppxf', 'False = off'),
        ('LMIN', '3650', 'rest-frame fit range [A] (clipped to the data)'),
        ('LMAX', '8674', ''),
        ('SPEC_MASK', 'auto', 'auto = specMask_SFH_<tag>'),
        ('MOM', '4', ''),
        ('MDEG', '4', ''),
        ('REGUL_ERR', '0.01', 'regularisation error; auto = chosen per bin by Delta chi2 = sqrt(2 N_good); '
                              'tools/gist_regul.py finds a value on a finished run'),
        ('NOISE', '1.', ''),
        ('FIXED', 'True', 'kinematics fixed to the KIN results'),
        ('NOISE_RESCALE', 'True', 'rescale the noise to chi2/DOF = 1 before the regularised fit'),
        ('GAS_LINES', 'None', 'emission lines fitted WITH the stars (unmasked in specMask_SFH): forbidden '
                              '([OII],[NeIII],[SII],[OIII],[FeIII],[NI],[OI],[NII],[ArIII]); +balmer_tied (Case-B '
                              'series, one flux) or +balmer (free lines) and +helium -- narrow lines only; '
                              'None = all masked'),
        ('GAS_VMAX', '1000', '|V_gas - V_star| limit [km/s]'),
        ('GAS_SIGMAX', '1000', 'sigma_gas upper limit [km/s]'),
        ('GAS_SIGMA0', 'None', 'starting sigma_gas [km/s]; None = max(100, 3 VELSCALE)'),
        ('GAS_RATIOS', 'True', 'keep [OII], [SII], [NI] doublet ratios and a free Balmer decrement inside their '
                               'physical range'),
        ('GAS_REDDENING', 'None', 'start A_V of the gas (fitted, Calzetti curve) -- only with balmer_tied, '
                                  'whose Case-B series constrains it; None = not fitted'),
        ('TEMPLATES', 'None', 'template reader for the SFH fit only (e.g. miles); None = PREPARE_TEMPLATES METHOD'),
        ('LIBRARY', 'None', 'template folder for the SFH fit only (e.g. EMILES_BASTI_UN/); None = '
                            'PREPARE_TEMPLATES LIBRARY'),
        ('LSF_TEMP', 'None', 'LSF file of that library (e.g. lsf_MILES); None = GENERAL LSF_TEMP'),
        ('MATCH_RES', 'auto', 'templates broader than the data (E-MILES 2.51 A vs KCWI blue): degrade the data '
                              'to their LSF; auto = when needed, False = stop instead'),
        ('BOOTSTRAP', '0', 'wild-bootstrap refits per bin for the errors of <log age>, [M/H], [alpha/Fe] '
                           '(e.g. 50; costs that many unregularised fits)'),
    ]),
    ('LS', [
        ('METHOD', 'False', 'default, or False = off'),
        ('TYPE', 'SPP', 'LS or SPP'),
        ('LS_FILE', 'lsBands.config', ''),
        ('CONV_COR', '8.4', ''),
        ('SPP_FILE', 'MILES_KB_LIS8.4.fits', ''),
        ('MC_LS', '30', ''),
        ('NWALKER', '10', ''),
        ('NCHAIN', '100', ''),
    ]),
]
MODULE_KEYS = dict((m, [k for k, _, _ in items]) for m, items in MODULES)
# 'auto' values that GIST itself resolves at run time (gist_prep passes them through)
RUNTIME_AUTO = {('SFH', 'REGUL_ERR'), ('SFH', 'MATCH_RES')}

# keys of SPATIAL_BINNING METHOD hierarchical (gistPipeline/spatialBinning/hierarchical.py)
HIER_KEYS = MODULE_KEYS['SPATIAL_BINNING'][MODULE_KEYS['SPATIAL_BINNING'].index('SB_TOL'):]
PREP_BOOL = ['header_only', 'lines_from_emi', 'no_sky', 'no_telluric', 'append', 'plot', 'no_snr_mask',
             'no_config']

EXAMPLE_PREP = """\
# gist_prep parameter file:      python gist_prep.py this_file.in
#
# This file is the whole set-up of a run: gist_prep writes the masks AND the
# MasterConfig (configFiles/MasterConfig_<tag>) from it -- do not edit that file,
# edit this one and rerun gist_prep. Then:  python run_gist.py configFiles/MasterConfig_<tag>
#
# [prep]   gist_prep's own options (the command-line names, '_' or '-').
# [MODULE] the GIST MasterConfig values, one KEY = VALUE per line (one word each).
#          A key left out, or left empty, takes the default shown here;
#          'auto' = filled in by gist_prep (from the cube or the [prep] options).
# Command-line options (and --set MODULE.KEY=VALUE) override this file.
# Wavelengths are REST-FRAME [A] unless marked observed.

[prep]
input       = inputData/IRAS03582+6012    # cube base name (red channel = <input>-red)
suffix_blue =            # blue stack = <input><suffix_blue>, e.g. _skl (kcwi_skylevel); empty = none
suffix_red  =            # red stack  = <input><suffix_red>-red
z           = 0.030
tag         = v4                          # suffix of every file written
configdir   = configFiles
run_id      =            # GENERAL RUN_ID; empty = <input name>_<tag>
emi         = configFiles/emissionLines.config    # GandALF file for GAS (omit if GAS is off)
plot        = true       # masks_<tag>.png and origin_<tag>.png
# template  = configFiles/MasterConfig    # optional: start from a row of an existing MasterConfig
# from_row  = -1                          #   (that row instead of the built-in defaults below)
# append    = false                       #   true: keep the template's rows, add this one
# --- masks
line_hw     = 500        # emission-line half-width [km/s]
broad       =            # lines masked with broad_hw, e.g. Hb,Ha,Hg,Hd (type-1 nucleus)
broad_hw    = 3000       # [km/s]
drop        =            # lines NOT to mask, e.g. NaD
extra       =            # extra rest-frame intervals, e.g. 8432-8459,8567-8631
sky         = variance   # variance: measured in the cube -- mask where sky lines raise the variance of
                         #   the faint spaxels by more than sky_excess (needs the reader, not header_only);
                         # model: sky_lines_paranal.dat (2098 lines) above sky_thresh x [OI]5577;
                         # list: old short list +- sky_hw; none
sky_excess  = 0.25       # variance: R = V/V_cont - 1 above which a sky line is masked (1 = doubles it)
sky_thresh  = 0.02       # model: smaller = more masked (KCWI red, FWHM ~4 A: 0.02 masks ~35-45% of 7500-9000 A)
sky_fwhm    = auto       # auto = the LSF_DATA file; or a number [A], or e.g. R1800
sky_hw      = 4          # [A, observed] half-width for sky = list and for sky_file lines
# sky_file  =            # more observed-frame sky lines, one per line
no_sky      = false
no_telluric = false
gap_pad     = 5          # [A] added on each side of the dichroic gap
gas_protect = 500        # [km/s] no sky mask this close to a fitted GAS line
header_only = false      # gap from the WAVGOOD headers instead of the reader
# --- S/N window (READ_DATA LMIN_SNR/LMAX_SNR = auto picks one; SNR_MASK keeps lines out of it)
snr_line_hw = 1000       # [km/s] emission lines kept out of the S/N (broad lines: broad_hw)
no_snr_mask = false      # true: no specMask_SNR_<tag>
snr_auto    = any        # channel for LMIN_SNR = auto: blue, red or any
snr_width   = 300        # [A] width of the S/N windows considered
# --- Galactic extinction (READ_DATA EBV = auto)
ebv_dir     =            # folder with SFD_dust_4096_ngp/sgp.fits; empty = $SFD_DIR, ~/software/dustmaps/sfd,
                         #   or software/dustmaps/sfd next to gist-development
ebv_scale   = 0.86       # x SFD (Schlafly & Finkbeiner 2011); 1 = original SFD
"""


def example_par():
    out = [EXAMPLE_PREP]
    for mod, items in MODULES:
        out.append('\n[%s]\n' % mod)
        for k, v, c in items:
            line = '%-13s = %s' % (k, v)
            out.append((line.ljust(30) + ('# ' + c if c else '')).rstrip() + '\n')
    return ''.join(out)


def _bool(v):
    s = str(v).strip().lower()
    if s in ('1', 'true', 'yes', 'y', 'on'):
        return True
    if s in ('0', 'false', 'no', 'n', 'off', ''):
        return False
    raise SystemExit('not a true/false value: %r' % v)


def read_par(fn, ap):
    """[prep] -> {dest: value}; every other section -> ['MODULE.KEY=VALUE', ...]."""
    cp = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#', ';'),
                                   comment_prefixes=('#', ';'), strict=True)
    cp.optionxform = str                                   # MasterConfig keys are case-sensitive
    with open(fn) as f:
        cp.read_file(f)
    dests = {x.dest for x in ap._actions} - {'help', 'parfile', 'set', 'example'}
    prep, sets = {}, []
    for sec in cp.sections():
        if sec.lower() == 'prep':
            for k, v in cp.items(sec):
                d = k.strip().lower().replace('-', '_')
                if d not in dests:
                    raise SystemExit('%s [prep]: unknown option %r\n  valid: %s' % (fn, k, ', '.join(sorted(dests))))
                v = v.strip()
                if d in PREP_BOOL:
                    prep[d] = _bool(v)
                elif v:                                     # empty = keep the default
                    prep[d] = v.replace(' ', '') if d in ('broad', 'drop', 'extra') else v
        else:
            for k, v in cp.items(sec):
                v = v.strip()
                if not v:
                    continue                                # empty = the default
                if any(c.isspace() for c in v):
                    raise SystemExit('%s [%s] %s = %r: a MasterConfig value must be one word' % (fn, sec, k, v))
                if sec not in MODULE_KEYS:
                    raise SystemExit('%s: unknown section [%s]; GIST modules are %s' % (fn, sec, ', '.join(MODULE_KEYS)))
                if k.strip() not in MODULE_KEYS[sec]:
                    print('NOTE: %s [%s] %s is not a known GIST key; it is passed on as given' % (fn, sec, k.strip()))
                sets.append('%s.%s=%s' % (sec, k.strip(), v))
    return prep, sets


def write_used_par(fn, a, ap, par_sets, cli_sets):
    """The settings this run used, as a parameter file that reproduces it."""
    with open(fn, 'w') as f:
        f.write('# written by gist_prep.py: the settings of this run (rerun: python gist_prep.py %s)\n\n[prep]\n'
                % os.path.basename(fn))
        for d in sorted(vars(a)):
            if d in ('set', 'parfile', 'example', 'help') or d.startswith('_'):
                continue
            v = getattr(a, d)
            if v is None or v == '':
                continue
            f.write('%-12s = %s\n' % (d, str(v).lower() if isinstance(v, bool) else v))
        secs = {}
        for kv in par_sets + cli_sets:                     # later wins, as in the MasterConfig row
            k, v = kv.split('=', 1)
            mod, key = k.split('.', 1)
            secs.setdefault(mod, {})[key] = v
        for mod, kv in secs.items():
            f.write('\n[%s]\n' % mod)
            for k, v in kv.items():
                f.write('%-12s = %s\n' % (k, v))


# ---------------------------------------------------------------------------
# data range and dichroic gap
# ---------------------------------------------------------------------------
def _wave(h):
    return h['CRVAL3'] + (np.arange(h['NAXIS3']) + 1 - h['CRPIX3']) * h['CD3_3']


def channel_ranges(base, header_only=False, pad=10., suffix_blue='', suffix_red=''):
    """Observed (first, last) usable wavelength of each channel, plus a reader
    handle for plotting. Uses the reader's own trimming when available."""
    out = {}
    reader = None
    if not header_only:
        try:
            from gistPipeline.readData import KECK_KCWIbpr as reader
            if not hasattr(reader, '_read_channel'):
                print('NOTE: the installed KECK_KCWIbpr has no _read_channel (old reader?); using WAVGOOD headers')
                reader = None
        except Exception as e:
            print('NOTE: could not import the GIST reader (%s); using WAVGOOD headers' % e)
            reader = None
    for label, b in (('blue', base + (suffix_blue or '')), ('red', base + (suffix_red or '') + '-red')):
        if reader is not None:
            h, w, f, v, bad = reader._read_channel(b, label)
            spec = np.where(bad, np.nan, f).reshape(len(w), -1)
            out[label] = dict(w0=w[0], w1=w[-1], wave=w, flux=spec, shape=f.shape[1:], hdr=h, file=b)
            # median variance spectrum of the FAINT half of the footprint (where the sky,
            # not the galaxy, sets the noise): the input of sky = variance
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                vv = np.where(bad, np.nan, v).reshape(len(w), -1)
                wl = np.nanmedian(spec, axis=0)
                ok = np.isfinite(wl) & (np.mean(np.isfinite(vv), axis=0) > 0.8)
                if ok.sum() >= 10:
                    faint = ok & (wl <= np.nanmedian(wl[ok]))
                    out[label]['varspec'] = np.nanmedian(vv[:, faint], axis=1)
                    out[label]['nfaint'] = int(faint.sum())
            del vv
        else:
            h = fits.getheader(b + '_icubes_wcs.fits')
            w = _wave(h)
            k = (w > h['WAVGOOD0'] + pad) & (w < h['WAVGOOD1'] - pad)
            out[label] = dict(w0=w[k][0], w1=w[k][-1], wave=None, flux=None, hdr=h, file=b)
    return out, reader is not None


def gap_rest(ch, z, gap_pad):
    """Rest-frame (lo, hi) of the region without data between the channels, or None."""
    b1, r0 = ch['blue']['w1'], ch['red']['w0']
    if b1 >= r0:
        return None
    return ((b1) / (1 + z) - gap_pad, (r0) / (1 + z) + gap_pad)


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------
def read_emi(fn):
    """Rows of a GandALF emissionLines.config: list of token lists (comments kept apart)."""
    rows = []
    for line in open(fn):
        s = line.split('#')[0].split()
        if len(s) >= 10:
            rows.append(s[:10])
    return rows


SKY_CATALOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sky_lines_paranal.dat')


def _vac2air(v):
    s2 = (1e4 / v) ** 2
    return v / (1. + 0.0000834254 + 0.02406147 / (130. - s2) + 0.00015998 / (38.9 - s2))


def _instrument_fwhm(a):
    """Observed-frame FWHM [A] as a function of wavelength, and a description: sky_fwhm
    = a number [A], R<number> (resolving power), or auto = the LSF_DATA file of the run
    (configdir/<GENERAL LSF_DATA>, columns wavelength and FWHM [A])."""
    v = str(a.sky_fwhm).strip()
    if v.lower() != 'auto':
        if v.upper().startswith('R'):
            R = float(v[1:])
            return (lambda lam: lam / R), 'R = %g' % R
        f = float(v)
        return (lambda lam: f + 0. * lam), '%g A' % f
    lsf = None
    for kv in a.set:
        if kv.split('=', 1)[0] == 'GENERAL.LSF_DATA':
            lsf = kv.split('=', 1)[1]
    if lsf is None:
        lsf = dict(MODULES)['GENERAL'][[k for k, _, _ in dict(MODULES)['GENERAL']].index('LSF_DATA')][1]
    fn = os.path.join(a.configdir, lsf)
    if os.path.isfile(fn):
        t = np.array([[float(x) for x in l.split()[:2]] for l in open(fn)
                      if l.split() and not l.lstrip().startswith('#')])
        o = np.argsort(t[:, 0])
        return (lambda lam: np.interp(lam, t[o, 0], t[o, 1])), 'LSF file %s' % fn
    print('NOTE: sky model: %s not found; using R = 1800' % fn)
    return (lambda lam: lam / 1800.), 'R = 1800 (no LSF file)'


def _sky_catalog(ch):
    """(wavelengths in the cube's frame, flux / F(5577)) of the sky-line catalogue."""
    cat = np.loadtxt(SKY_CATALOG)
    air = True
    for label in ('blue', 'red'):
        h = ch[label].get('hdr')
        if h is not None and str(h.get('CTYPE3', 'AWAV')).upper().startswith('WAVE'):
            air = False
    return (cat[:, 1] if air else cat[:, 0]), cat[:, 2], air


def sky_variance_intervals(a, ch, z):
    """
    Sky-line masks measured in the data: R(lambda) = V(lambda) / V_cont(lambda) - 1, with V
    the median variance spectrum of the faint half of the footprint and V_cont its lower
    envelope (20th percentile over +-max(10 A, 4 FWHM), then smoothed). Where the background
    dominates the noise, V is proportional to the photons collected, so R is the sky line's
    brightness relative to everything else under it (sky continuum, galaxy, read noise),
    i.e. its share of the noise budget: R = 1 means the line doubles the variance. Masked
    where R > sky_excess. A peak that matches no catalogued sky line but sits on a known
    GALAXY emission line is left alone (extended line emission also raises the variance).
    Returns observed-frame (lo, hi, label) intervals, or None without variance spectra.
    """
    if getattr(a, '_sky_iv', None) is not None:
        return a._sky_iv
    if any('varspec' not in ch[l] for l in ('blue', 'red')):
        return None
    from scipy import ndimage
    lam_cat, rel_cat, air = _sky_catalog(ch)
    strong = rel_cat >= 0.002
    fwhm, how = _instrument_fwhm(a)
    gal_lines = [lam * (1 + z) for _, lam in LINES]
    out, a._sky_R = [], {}
    for label in ('blue', 'red'):
        w, V = ch[label]['wave'], np.array(ch[label]['varspec'], float)
        good = np.isfinite(V) & (V > 0)
        if good.sum() < 50:
            continue
        V = np.interp(w, w[good], V[good])
        dw = np.median(np.diff(w))
        half = max(10., 4. * float(np.median(fwhm(w))))
        n = max(int(2 * half / dw) | 1, 5)
        cont = ndimage.uniform_filter1d(ndimage.percentile_filter(V, 20, size=n, mode='nearest'), n, mode='nearest')
        R = V / cont - 1.
        a._sky_R[label] = (w, R)
        cut = R > a.sky_excess
        edges = np.flatnonzero(np.diff(np.r_[0, cut.astype(int), 0]))
        kept = skipped = 0
        for i0, i1 in zip(edges[::2], edges[1::2]):
            lo, hi = w[i0] - dw / 2, w[i1 - 1] + dw / 2
            pk = w[i0 + int(np.argmax(R[i0:i1]))]
            f = float(fwhm(pk))
            is_sky = np.any(strong & (lam_cat > lo - f) & (lam_cat < hi + f))
            on_gal = any(lo - g * a.line_hw / C < g < hi + g * a.line_hw / C for g in gal_lines)
            if on_gal and not is_sky:
                skipped += 1
                continue
            out.append((lo, hi, 'sky%d' % round(pk)))
            kept += 1
        frac = np.mean(cut)
        print('sky variance (%s, faint %d spaxels): %d intervals with R > %g (sky lines adding > %.0f%% to the '
              'variance), %.0f%% of %.0f-%.0f A masked%s' % (label, ch[label]['nfaint'], kept, a.sky_excess,
              100 * a.sky_excess, 100 * frac, w[0], w[-1],
              '; %d peaks on galaxy emission lines left alone' % skipped if skipped else ''))
    a._sky_iv = out
    return out


def sky_model_intervals(a, ch):
    """
    Observed-frame (lo, hi, label) intervals where the night-sky line spectrum, convolved
    to the instrument resolution, exceeds a.sky_thresh x the peak of [OI] 5577 at the same
    resolution. Blends add up, and a strong line gets a wider mask than a weak one, so the
    mask follows where sky residuals can actually be large.
    """
    if getattr(a, '_sky_iv', None) is not None:          # computed once per run
        return a._sky_iv
    if not os.path.isfile(SKY_CATALOG):
        raise SystemExit('sky = model needs %s (shipped with gist_prep.py)' % SKY_CATALOG)
    lam, rel, air = _sky_catalog(ch)
    fwhm, how = _instrument_fwhm(a)
    out = []
    for label in ('blue', 'red'):
        w0, w1 = ch[label]['w0'], ch[label]['w1']
        g = np.arange(w0, w1, 0.1)
        sig = fwhm(g) / 2.3548
        mod = np.zeros_like(g)
        use = (lam > w0 - 20) & (lam < w1 + 20)
        for l, r in zip(lam[use], rel[use]):
            s = float(np.interp(l, g, sig))
            k = np.abs(g - l) < 5 * s
            mod[k] += r * np.exp(-0.5 * ((g[k] - l) / s) ** 2)     # peak = flux / F(5577)
        cut = mod > a.sky_thresh
        edges = np.flatnonzero(np.diff(np.r_[0, cut.astype(int), 0]))
        for i0, i1 in zip(edges[::2], edges[1::2]):
            lo, hi = g[i0], g[min(i1, g.size) - 1]
            pk = g[i0 + np.argmax(mod[i0:i1])]
            out.append((lo - 0.05, hi + 0.05, 'sky%d' % round(pk)))
        frac = cut.mean()
        print('sky model (%s, %s wavelengths, FWHM from %s, threshold %g x [OI]5577): %d intervals, '
              '%.0f%% of %.0f-%.0f A masked' % (label, 'air' if air else 'vacuum', how, a.sky_thresh,
                                              len(edges) // 2, 100 * frac, w0, w1))
    a._sky_iv = out
    return out


def _load_aux(name):
    """gistPipeline/auxiliary/<name>.py: the installed package, or next to this script."""
    try:
        import importlib
        return importlib.import_module('gistPipeline.auxiliary.' + name)
    except ImportError:
        import importlib.util
        fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gistPipeline', 'auxiliary', name + '.py')
        spec = importlib.util.spec_from_file_location(name, fn)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod


def _load_gaslines():
    return _load_aux('gaslines')


def galactic_ebv(a, ch, law='f99'):
    """READ_DATA EBV = auto: SFD x ebv_scale at the centre of the blue cube (WCS)."""
    gal = _load_aux('galactic')
    c = ch['blue']
    h = c.get('hdr')
    if h is None:
        h = fits.getheader(c['file'] + '_icubes_wcs.fits')
    from astropy.wcs import WCS
    w = WCS(h).celestial
    ra, dec = w.all_pix2world([[(h['NAXIS1'] - 1) / 2., (h['NAXIS2'] - 1) / 2.]], 0)[0]
    try:
        ebv, info = gal.sfd_ebv(ra, dec, a.ebv_dir or None, scale=a.ebv_scale)
    except IOError as e:
        raise SystemExit('READ_DATA EBV = auto: %s. Give --ebv-dir (or ebv_dir in [prep]), or set READ_DATA '
                         'EBV to a number / None.' % e)
    k = gal.curve(law)
    print('Galactic extinction: E(B-V) = %.4f (SFD %.4f x %g) at RA,Dec %.5f,%.5f (l,b = %.2f,%+.2f); A_V = %.2f, '
          'A(3700 A) - A(5300 A) = %.2f mag (%s)' % (ebv, info['sfd'], a.ebv_scale, ra, dec, info['l'], info['b'],
                                                    3.1 * ebv, ebv * (k(3700.) - k(5300.)), law))
    if str(law).lower() != 'f99' and abs(a.ebv_scale - 0.86) < 1e-6:
        print('NOTE: ebv_scale 0.86 is the Schlafly & Finkbeiner (2011) recalibration FOR F99; with %s it is '
              'not self-consistent (their fits favour F99 over CCM89/O\'Donnell 1994)' % law)
    if abs(info['b']) < 10:
        print('NOTE: |b| = %.1f deg: SFD is less reliable this close to the plane (unresolved dust structure); '
              'compare with an independent estimate (e.g. Planck, or the stellar Balmer decrement of the '
              'foreground-free gas) if the line ratios matter' % abs(info['b']))
    return ebv


def sfh_gas_lines(a):
    """SFH GAS_LINES from the parameter file / --set: (template names, gaslines module)."""
    val = None
    for kv in a.set:
        k, v = kv.split('=', 1)
        if k == 'SFH.GAS_LINES':
            val = v
    gl = _load_gaslines()
    return gl.parse_gas_lines(val), gl


def build_intervals(a, z, ch, gap):
    """List of (lo, hi, label, kind) rest-frame intervals; kind = line / sky / gap / extra."""
    iv = []
    # emission lines
    if a.lines_from_emi and a.emi:
        lines = [(r[1], float(r[2])) for r in read_emi(a.emi) if r[3] == 'f']
    else:
        lines = list(LINES)
    drop = set(x.strip() for x in a.drop.split(',') if x.strip())
    broad = set(x.strip() for x in a.broad.split(',') if x.strip())
    for name, lam in lines:
        if name in drop:
            continue
        hw = a.broad_hw if name in broad else a.line_hw
        d = lam * hw / C
        iv.append((lam - d, lam + d, re.sub(r'[\[\]]', '', name), 'line'))
    # sky lines (observed -> rest): 'model' = every line of sky_lines_paranal.dat that rises
    # above sky_thresh x the peak of [OI] 5577, both at the instrument's resolution;
    # 'list' = the short SKY list +- sky_hw
    mode = 'none' if a.no_sky else a.sky
    if mode == 'variance':
        sv = sky_variance_intervals(a, ch, z)
        if sv is None:
            if not getattr(a, '_sky_warned', False):
                print('NOTE: sky = variance needs the variance cubes through the GIST reader (not with '
                      'header_only); using sky = model instead')
                a._sky_warned = True
            mode = 'model'
        else:
            for lo, hi, lab in sv:
                iv.append((lo / (1 + z), hi / (1 + z), lab, 'sky'))
    if mode == 'model':
        for lo, hi, lab in sky_model_intervals(a, ch):
            iv.append((lo / (1 + z), hi / (1 + z), lab, 'sky'))
    sky = list(SKY) if mode == 'list' else []
    if a.sky_file:
        sky += [float(l.split()[0]) for l in open(a.sky_file) if l.strip() and not l.startswith('#')]
    for s in sky:
        iv.append(((s - a.sky_hw) / (1 + z), (s + a.sky_hw) / (1 + z), 'sky%d' % round(s), 'sky'))
    if not a.no_telluric:
        for name, lo, hi in TELLURIC:
            iv.append((lo / (1 + z), hi / (1 + z), name, 'sky'))
    if gap is not None:
        iv.append((gap[0], gap[1], 'gap', 'gap'))
    for tok in [t for t in a.extra.split(',') if t.strip()]:
        lo, hi = [float(x) for x in tok.split('-')]
        iv.append((lo, hi, 'extra', 'extra'))
    # keep what overlaps the data
    lo_d, hi_d = ch['blue']['w0'] / (1 + z), ch['red']['w1'] / (1 + z)
    return [x for x in iv if x[1] > lo_d and x[0] < hi_d]


def merge(iv):
    """Merge overlapping intervals; labels joined with '+'."""
    iv = sorted(x[:3] for x in iv)
    out = []
    for lo, hi, lab in iv:
        if out and lo <= out[-1][1]:
            plo, phi, plab = out[-1]
            labs = plab.split('+')
            if lab not in labs:
                labs.append(lab)
            out[-1] = (plo, max(phi, hi), '+'.join(labs))
        else:
            out.append((lo, hi, lab))
    return out


def gas_masks(a, raw):
    """
    Masks for GandALF: the gap and --extra always; sky/telluric only where they
    do not touch a FITTED emission line (+-a.gas_protect km/s) -- masking there
    would cut into the line being measured. Line masks are not added: GAS fits them.
    """
    fitted = [float(r[2]) for r in read_emi(a.emi) if r[3] == 'f']
    keep, dropped = [], []
    for lo, hi, lab, kind in raw:
        if kind == 'line':
            continue
        if kind == 'sky' and any(lo - l * a.gas_protect / C < l < hi + l * a.gas_protect / C for l in fitted):
            dropped.append(lab)
            continue
        keep.append((lo, hi, lab, kind))
    if dropped:
        print('GAS: not masking %s (on top of a fitted line; check those lines by eye)' % ', '.join(dropped))
    return merge(keep)


def suggest_snr_windows(snr_iv, z, ch, gap, width=300., step=10., edge=150., oh=7200.):
    """
    Candidate S/N windows of `width` A (rest), scanned every `step` A inside each
    channel (not within `edge` A of a channel end, where throughput drops, and not
    beyond `oh` A observed, the OH forest). Each is scored by the fraction of its
    pixels the S/N mask leaves free. Returns {channel: [(lo, hi, free), ...]},
    best first, non-overlapping.
    """
    out = {}
    for label in ('blue', 'red'):
        c0, c1 = ch[label]['w0'] / (1 + z) + edge, ch[label]['w1'] / (1 + z) - edge
        c1 = min(c1, oh / (1 + z))
        cand = []
        lo = c0
        while lo + width <= c1:
            hi = lo + width
            if gap is None or hi <= gap[0] or lo >= gap[1]:
                g = np.linspace(lo, hi, 301)
                cut = np.zeros(g.size, bool)
                for a, b, _ in snr_iv:
                    cut |= (g >= a) & (g <= b)
                cand.append((lo, hi, 1. - cut.mean()))
            lo += step
        mid = 0.5 * (c0 + c1)          # ties: prefer the middle of the channel (throughput)
        cand.sort(key=lambda t: (-round(t[2] / 0.02), abs(0.5 * (t[0] + t[1]) - mid)))
        best = []
        for t in cand:
            if all(t[1] <= b[0] or t[0] >= b[1] for b in best):
                best.append(t)
            if len(best) == 3:
                break
        out[label] = best
    return out


def write_specmask(fn, iv, z, module):
    with open(fn, 'w') as f:
        f.write('# Spectral masks for the %s module, written by gist_prep.py (z = %.5f).\n' % (module, z))
        f.write('# ALL entries are REST-FRAME. Observed-frame features (sky, telluric, the\n')
        f.write('# dichroic gap) were converted here, so none is labelled "sky" (GIST would\n')
        f.write('# divide its centre by 1+z a second time).\n#\n')
        f.write('# Lambda [Angst.]    Width [Angst.]    Comment\n')
        for lo, hi, lab in iv:
            f.write('%-20.2f %-17.2f %s\n' % (0.5 * (lo + hi), hi - lo, lab))


def write_emi(fn_in, fn_out, iv_obs, z, ch, gap):
    """Copy the GandALF file; add 'm' entries; ignore fitted lines without data."""
    rows = read_emi(fn_in)
    used = [int(r[0]) for r in rows]
    nxt = max(used + [0]) + 1
    lo_d, hi_d = ch['blue']['w0'] / (1 + z), ch['red']['w1'] / (1 + z)
    ignored = []
    lines = open(fn_in).read().rstrip('\n').split('\n')
    out = []
    for line in lines:
        s = line.split('#')[0].split()
        if len(s) >= 10 and s[3] == 'f':
            lam = float(s[2])
            nogap = gap is None or not (gap[0] <= lam <= gap[1])
            if not (lo_d < lam < hi_d and nogap):
                line = re.sub(r'(^\s*\S+\s+\S+\s+\S+\s+)f(\s)', r'\1i\2', line, count=1)
                ignored.append((s[0], s[1], lam))
        out.append(line)
    # lines referring to an ignored one (d<i> / t<i>) would break the fit
    ign_idx = set(i for i, _, _ in ignored)
    for r in rows:
        refs = re.findall(r'[dt](\d+)', r[4] + ' ' + r[8])
        if r[3] == 'f' and any(x in ign_idx for x in refs) and r[0] not in ign_idx:
            print('WARNING: GAS line %s %s is tied to an ignored line (%s); edit %s' % (r[0], r[1], r[4] + '/' + r[8], fn_out))
    out.append('')
    out.append('# Added by gist_prep.py: regions without usable data or with sky residuals, as')
    out.append('# rest-frame masks (half-width = 3 x sigma). Not named "sky" on purpose: they are')
    out.append('# already converted to the rest frame.')
    for lo, hi, lab in iv_obs:
        lam = 0.5 * (lo + hi)
        sig = int(math.ceil((hi - lo) / 2. / lam * C / 3.))
        out.append('%4d     %-8s %8.2f m      l       1.000      0  %5d     f      4' % (nxt, lab.split('+')[0][:8], lam, sig))
        nxt += 1
    with open(fn_out, 'w') as f:
        f.write('\n'.join(out) + '\n')
    return ignored


# ---------------------------------------------------------------------------
# MasterConfig
# ---------------------------------------------------------------------------
def read_masterconfig(fn):
    head, body = [], []
    for line in open(fn):
        line = line.rstrip('\n')
        if line.startswith('#'):
            head.append(line)
        elif line.strip():
            body.append(line)
    mods = [m.strip() for m in body[0].split('|') if m.strip()]
    keys = [k.split() for k in body[1].split('|')]
    rows = []
    for line in body[2:]:
        vals = [v.split() for v in line.split('|')]
        rows.append([(mods[i], list(zip(keys[i], vals[i]))) for i in range(len(mods))])
    return head, mods, keys, rows


def write_masterconfig(fn, head, mods, keys, rows, summary=()):
    # column widths per (module, key)
    cells = []
    for i, m in enumerate(mods):
        cols = []
        for j, k in enumerate(keys[i]):
            w = max([len(k)] + [len(r[i][1][j][1]) for r in rows])
            cols.append(w)
        cells.append(cols)
    def line(items):
        return ' | '.join('  '.join(v.ljust(cells[i][j]) for j, v in enumerate(items[i])) for i in range(len(mods)))
    width = [len('  '.join(k.ljust(cells[i][j]) for j, k in enumerate(keys[i]))) for i in range(len(mods))]
    with open(fn, 'w') as f:
        for h in list(summary) + list(head):
            f.write(h + '\n')
        f.write(' | '.join(m.ljust(width[i]) for i, m in enumerate(mods)).rstrip() + '\n')
        f.write(line(keys).rstrip() + '\n')
        for r in rows:
            f.write(line([[v for _, v in r[i][1]] for i in range(len(mods))]).rstrip() + '\n')
    # no trailing blank line: GIST counts every non-comment line as a run


def default_table():
    """The built-in MasterConfig (MODULES) in read_masterconfig's structure."""
    mods = [m for m, _ in MODULES]
    keys = [[k for k, _, _ in items] for _, items in MODULES]
    row = [(m, [(k, v) for k, v, _ in items]) for m, items in MODULES]
    return [], mods, keys, [row]


def set_value(row, keys, mods, mod, key, val):
    i = mods.index(mod)
    items = row[i][1]
    for j, (k, _) in enumerate(items):
        if k == key:
            items[j] = (k, val)
            return False
    items.append((key, val))           # new column
    keys[i].append(key)
    return True


def _is_auto(v):
    return v is None or str(v).strip().lower() == 'auto'


def _unset(v):
    return v is None or str(v).strip().lower() in ('none', 'false', '', 'auto')


def find_origin(a, ch, z, s0, s1, spec):
    """
    READ_DATA ORIGIN (x, y in 0-based pixels, as the reader uses it).
      auto         the continuum peak: white-light image in the S/N window (rest s0-s1),
                   smoothed (sigma 1.5 px) inside the footprint shrunk by 3 px, then the
                   flux-weighted centre within 3 px of the peak
      radec:RA,DEC that sky position [deg] through the cube's WCS
    """
    lo, hi = s0 * (1 + z), s1 * (1 + z)
    over = dict((l, min(hi, ch[l]['w1']) - max(lo, ch[l]['w0'])) for l in ('blue', 'red'))
    label = max(over, key=over.get)
    c = ch[label]
    hdr = c.get('hdr')
    if str(spec).lower().startswith('radec:'):
        from astropy.wcs import WCS
        ra, dec = [float(t) for t in str(spec).split(':', 1)[1].split(',')]
        h = hdr if hdr is not None else fits.getheader(c['file'] + '_icubes_wcs.fits')
        x, y = WCS(h).celestial.all_world2pix([[ra, dec]], 0)[0]
        return float(x), float(y), 'RA,Dec %.6f,%.6f through the %s cube WCS' % (ra, dec, label)
    if c.get('flux') is not None:
        wave = c['wave']; cube = c['flux'].reshape((len(wave),) + tuple(c['shape']))
    else:
        fn = c['file'] + '_icubes_wcs.fits'
        cube = fits.getdata(fn).astype(float)
        wave = _wave(fits.getheader(fn))
    k = (wave >= lo) & (wave <= hi)
    if k.sum() < 5:
        raise SystemExit('ORIGIN auto: the S/N window %g-%g A rest has no data in either cube' % (s0, s1))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        img = np.nanmedian(cube[k], axis=0)
    from scipy import ndimage
    good = np.isfinite(img)
    foot = ndimage.binary_erosion(good, iterations=3)
    if not foot.any():
        foot = good
    # only the footprint shrunk by 3 px: the cube edges are often hot or half-covered;
    # a 3x3 median removes hot pixels / cosmic-ray residuals before smoothing
    im0 = np.where(foot, ndimage.median_filter(np.where(good, img, 0.), size=3), 0.)
    sm = ndimage.gaussian_filter(im0, 1.5) / np.maximum(ndimage.gaussian_filter(foot.astype(float), 1.5), 1e-6)
    sm[~foot] = -np.inf
    py, px = np.unravel_index(np.argmax(sm), sm.shape)
    yy, xx = np.indices(img.shape)
    wgt = np.where((np.hypot(yy - py, xx - px) <= 3) & foot, im0 - np.median(img[foot]), 0.)
    wgt[wgt < 0] = 0.
    if wgt.sum() > 0:
        cy, cx = (wgt * yy).sum() / wgt.sum(), (wgt * xx).sum() / wgt.sum()
    else:
        cy, cx = float(py), float(px)
    if a.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 6))
        v = img[good]
        ax.imshow(np.arcsinh(np.where(good, img, np.nan) / max(np.percentile(v, 99) / 50., 1e-30)),
                  origin='lower', cmap='Greys_r', aspect='auto')
        ax.plot(cx, cy, '+', color='red', ms=18, mew=1.5)
        ax.set_title('ORIGIN %.2f,%.2f (%g-%g A rest, %s)' % (cx, cy, s0, s1, label), fontsize=9)
        fig.tight_layout()
        fn = os.path.join(a.configdir, 'origin_%s.png' % a.tag)
        fig.savefig(fn, dpi=110)
        plt.close(fig)
        print('wrote', fn)
    return float(cx), float(cy), 'continuum peak, %g-%g A rest, %s cube' % (s0, s1, label)


def make_config(a, z, ch, gap, names, iv=()):
    if a.template:
        head, mods, keys, rows = read_masterconfig(a.template)
        src = rows[a.from_row]
        origin_of_row = 'row %d of %s' % (a.from_row, a.template)
    else:
        head, mods, keys, rows = default_table()
        src = rows[0]
        origin_of_row = 'built-in defaults'
    row = [(m, list(items)) for m, items in src]
    lo_d, hi_d = ch['blue']['w0'] / (1 + z), ch['red']['w1'] / (1 + z)
    def get(mod, key):
        return dict(row[mods.index(mod)][1]).get(key) if mod in mods else None
    neutral = {('READ_DATA', 'SUFFIX_BLUE'): 'None', ('READ_DATA', 'SUFFIX_RED'): 'None'}
    def put(mod, key, v):
        if (mod, key) in neutral and v == neutral[(mod, key)] and get(mod, key) is None:
            return                        # no suffix and no such column: leave the table as it is
        if set_value(row, keys, mods, mod, key, v):
            for r in rows:                # give the other rows the new column too
                if r is not row:          # (a suffix column: 'None' there, so old rows read as before)
                    set_value(r, [list(x) for x in keys], mods, mod, key, neutral.get((mod, key), v))

    # 1. the user's values ([MODULE] sections of the parameter file, then --set)
    user = {}
    for kv in a.set:
        k, v = kv.split('=', 1)
        mod, key = k.split('.', 1)
        if mod not in mods:
            raise SystemExit('module %s is not in the MasterConfig (%s has %s)' % (mod, origin_of_row, ', '.join(mods)))
        user[(mod, key)] = v
        put(mod, key, v)

    # 2. values that follow from the [prep] options and the files written here; a value
    #    the user gave wins, unless it is 'auto'
    derived = {('GENERAL', 'REDSHIFT'): '%.5f' % z,
               ('GENERAL', 'INPUT'): os.path.basename(a.input),
               ('READ_DATA', 'SUFFIX_BLUE'): a.suffix_blue or 'None',
               ('READ_DATA', 'SUFFIX_RED'): a.suffix_red or 'None',
               ('KIN', 'SPEC_MASK'): names['kin'], ('SFH', 'SPEC_MASK'): names['sfh'],
               ('GAS', 'EMI_FILE'): names.get('emi', 'None'),
               ('READ_DATA', 'SNR_MASK'): names.get('snr', 'None'),
               ('GENERAL', 'RUN_ID'): a.run_id or '%s_%s' % (os.path.basename(a.input), a.tag)}
    for (mod, key), v in derived.items():
        if mod not in mods:
            continue
        if (mod, key) in user and not _is_auto(user[(mod, key)]):
            continue
        if (mod, key) == ('GENERAL', 'RUN_ID') and not a.run_id and not _is_auto(get(mod, key)):
            continue                      # a template's RUN_ID is kept unless run_id is given
        if (mod, key) in (('GAS', 'EMI_FILE'), ('READ_DATA', 'SNR_MASK')) and v == 'None' \
                and not _is_auto(get(mod, key)):
            continue                      # nothing written here: keep what the row has
        put(mod, key, v)

    # 3. S/N window: auto -> the cleanest window (the S/N mask leaves most of it)
    sugg = suggest_snr_windows(names['snr_iv'], z, ch, gap, a.snr_width) if names.get('snr_iv') else {}
    want = _is_auto(get('READ_DATA', 'LMIN_SNR')) or _is_auto(get('READ_DATA', 'LMAX_SNR'))
    # (snr_auto picks the channel; with a template it also replaces the row's window, as before)
    if (want or (a.template and a.snr_auto and ('READ_DATA', 'LMIN_SNR') not in user)) and sugg:
        which = a.snr_auto or 'any'
        pool = [t for k, v in sugg.items() if which in (k, 'any') for t in v]
        if pool:
            lo, hi, fr = max(pool, key=lambda t: t[2])
            put('READ_DATA', 'LMIN_SNR', '%g' % round(lo)); put('READ_DATA', 'LMAX_SNR', '%g' % round(hi))
            print('S/N window set to %g-%g A rest (%.0f%% free of masks; channel %s)' % (round(lo), round(hi), 100 * fr, which))
    if _is_auto(get('READ_DATA', 'LMIN_SNR')) or _is_auto(get('READ_DATA', 'LMAX_SNR')):
        raise SystemExit('READ_DATA LMIN_SNR/LMAX_SNR = auto needs the S/N mask intervals; give the window explicitly')

    # 4. wavelength windows inside the data (rest frame); auto = all the data
    for mod, k0, k1 in (('READ_DATA', 'LMIN_TOT', 'LMAX_TOT'), ('KIN', 'LMIN', 'LMAX'),
                        ('GAS', 'LMIN', 'LMAX'), ('SFH', 'LMIN', 'LMAX')):
        if mod in mods and get(mod, k0) is not None:
            if _is_auto(get(mod, k0)) or _is_auto(get(mod, k1)):
                put(mod, k0, '%g' % math.ceil(lo_d)); put(mod, k1, '%g' % math.floor(hi_d))
                continue
            l0, l1 = float(get(mod, k0)), float(get(mod, k1))
            n0, n1 = max(l0, math.ceil(lo_d)), min(l1, math.floor(hi_d))
            if (n0, n1) != (l0, l1):
                print('%s %s/%s: %g-%g -> %g-%g (data cover %.1f-%.1f A rest)' % (mod, k0, k1, l0, l1, n0, n1, lo_d, hi_d))
            put(mod, k0, '%g' % n0); put(mod, k1, '%g' % n1)
    s0, s1 = float(get('READ_DATA', 'LMIN_SNR')), float(get('READ_DATA', 'LMAX_SNR'))

    # 5. ORIGIN
    org = get('READ_DATA', 'ORIGIN')
    if 'READ_DATA' in mods and (_is_auto(org) or str(org).lower().startswith('radec:')):
        x0, y0, how = find_origin(a, ch, z, s0, s1, 'auto' if _is_auto(org) else org)
        put('READ_DATA', 'ORIGIN', '%.2f,%.2f' % (x0, y0))
        print('READ_DATA ORIGIN = %.2f,%.2f (%s); set ORIGIN = x,y to override' % (x0, y0, how))

    # 6. Galactic extinction
    if 'READ_DATA' in mods and _is_auto(get('READ_DATA', 'EBV')) and get('READ_DATA', 'EBV') is not None:
        put('READ_DATA', 'EBV', '%.4f' % galactic_ebv(a, ch, get('READ_DATA', 'EXT_LAW') or 'f99'))
    ebv = get('READ_DATA', 'EBV')
    if not _unset(ebv) and float(ebv) > 0 and 'GAS' in mods and not _unset(get('GAS', 'EBmV')):
        raise SystemExit('READ_DATA EBV = %s and GAS EBmV = %s: the reader already removes the Galactic '
                         'extinction; set GAS EBmV = None' % (ebv, get('GAS', 'EBmV')))

    left = ['%s %s' % (m, k) for m, items in row for k, v in items
            if _is_auto(v) and v is not None and (m, k) not in RUNTIME_AUTO]
    if left:
        raise SystemExit('no value could be derived for: %s' % ', '.join(left))

    # sanity: SNR window vs gap
    if gap is not None and s0 >= gap[0] and s1 <= gap[1]:
        print('ERROR: READ_DATA LMIN_SNR-LMAX_SNR (%g-%g) lies inside the gap; the reader will stop' % (s0, s1))
    elif gap is not None and s0 < gap[1] and s1 > gap[0]:
        print('NOTE: the SNR window overlaps the gap; the reader excludes gap pixels from the SNR')
    if gap is None or not (s0 >= gap[0] and s1 <= gap[1]):
        osky = [s for s in SKY if s0 * (1 + z) <= s <= s1 * (1 + z)]
        if s1 * (1 + z) > 7200.:
            print('NOTE: the SNR window %g-%g A rest is %.0f-%.0f A observed, in the OH forest (%d of the '
                  'masked sky lines). Sky residuals are not in the variance and add up over a bin; a '
                  'clean continuum window gives a truer S/N (check with gist_snrcheck.py).'
                  % (s0, s1, s0 * (1 + z), s1 * (1 + z), len(osky)))
    if names.get('snr_iv'):
        free, hit = _free_fraction(s0, s1, names['snr_iv'])
        print('S/N window %g-%g A rest: %.0f%% of it is free of line/sky masks%s%s' % (
            s0, s1, 100 * free, ' (masked: %s)' % ', '.join(hit) if hit else '',
            '; SNR_MASK leaves the rest out' if not _unset(get('READ_DATA', 'SNR_MASK')) else
            ' -- WARNING: no SNR_MASK, so those lines count as continuum signal'))
        if free < 0.8:
            print('NOTE: a cleaner window would need less masking:')
        for k, v in sugg.items():
            print('  cleanest %s windows (%g A): %s' % (k, a.snr_width, ', '.join(
                '%.0f-%.0f (%.0f%% free)' % (t[0], t[1], 100 * t[2]) for t in v) or 'none'))
    a_configdir[0] = a.configdir
    check_binning(row, mods, get, iv, names.get('snr_iv', []), lo_d, hi_d)
    check_sfh(row, mods, get, a, z)
    rows = rows + [row] if (a.template and a.append) else [row]
    fn = os.path.join(a.configdir, 'MasterConfig_' + a.tag)
    summary = ['# MasterConfig written by gist_prep.py (tag %s, from %s).' % (a.tag, origin_of_row),
               '# Do not edit it: change the gist_prep parameter file (configFiles/gist_prep_%s.in' % a.tag,
               '# reproduces this run) and rerun gist_prep. GIST reads only the two header lines and',
               '# the run row at the bottom; the listing below is the same row, for reading.', '#']
    for m, items in row:
        summary.append('# [%s]' % m)
        for k, v in items:
            summary.append('#   %-14s %s' % (k, v))
    summary.append('#')
    write_masterconfig(fn, head, mods, keys, rows, summary)
    return fn


def _lsf(configdir, name):
    fn = os.path.join(configdir, str(name))
    if not os.path.isfile(fn):
        return None
    t = np.genfromtxt(fn, comments='#')
    return t[:, 0], t[:, 1]


def check_sfh(row, mods, get, a, z):
    """Report the SFH set-up: template library, resolution matching, gas lines."""
    if 'SFH' not in mods or _unset(get('SFH', 'METHOD')):
        return
    lib = get('SFH', 'LIBRARY'); meth = get('SFH', 'TEMPLATES'); ltemp = get('SFH', 'LSF_TEMP')
    lib_k = get('PREPARE_TEMPLATES', 'LIBRARY') if _unset(lib) else lib
    meth_k = get('PREPARE_TEMPLATES', 'METHOD') if _unset(meth) else meth
    lt = get('GENERAL', 'LSF_TEMP') if _unset(ltemp) else ltemp
    own = not (_unset(lib) and _unset(meth))
    print('SFH templates: %s %s, LSF %s%s' % (meth_k, lib_k, lt, ' (SFH only; KIN keeps %s %s)' % (
        get('PREPARE_TEMPLATES', 'METHOD'), get('PREPARE_TEMPLATES', 'LIBRARY')) if own else ''))
    if own and _unset(ltemp):
        print('WARNING: SFH LIBRARY/TEMPLATES set but SFH LSF_TEMP is not: the SFH templates are taken to have '
              'the KIN library LSF (%s)' % get('GENERAL', 'LSF_TEMP'))
    d = _lsf(a.configdir, get('GENERAL', 'LSF_DATA')); t = _lsf(a.configdir, lt)
    for nm, v in ((get('GENERAL', 'LSF_DATA'), d), (lt, t)):
        if v is None:
            print('WARNING: LSF file %s not found in %s' % (nm, a.configdir))
    try:
        l0, l1 = float(get('SFH', 'LMIN')), float(get('SFH', 'LMAX'))
    except (TypeError, ValueError):
        l0 = l1 = None
    if d is not None and t is not None and l0 is not None:
        g = np.linspace(l0, l1, 200)
        fd = np.interp(g * (1 + z), d[0], d[1]) / (1 + z)        # data LSF, rest frame (as GIST's getLSF)
        ft = np.interp(g, t[0], t[1])
        mres = str(get('SFH', 'MATCH_RES')).strip().lower()
        if np.any(ft > fd * 1.001):
            if mres == 'false':
                print('ERROR: the SFH templates (FWHM %.2f-%.2f A) are broader than the data (%.2f-%.2f A rest) '
                      'and MATCH_RES = False: the SFH module will stop' % (ft.min(), ft.max(), fd.min(), fd.max()))
            else:
                print('SFH MATCH_RES: data degraded from FWHM %.2f-%.2f A to the template %.2f-%.2f A (rest, '
                      '%g-%g A); fixed KIN kinematics stay valid (physical sigma). Correlated noise after '
                      'smoothing is allowed for in REGUL_ERR = auto / gist_regul / BOOTSTRAP'
                      % (fd.min(), fd.max(), max(ft.min(), fd.min()), max(ft.max(), fd.max()), l0, l1))
    gl = _load_gaslines()
    try:
        names = gl.parse_gas_lines(get('SFH', 'GAS_LINES'))
    except ValueError as e:
        raise SystemExit(str(e))
    if not _unset(get('SFH', 'GAS_REDDENING')) and 'Balmer' not in names:
        raise SystemExit('SFH GAS_REDDENING = %s needs GAS_LINES ...,balmer_tied (the Case-B Balmer series is '
                         'what constrains the gas A_V)' % get('SFH', 'GAS_REDDENING'))
    if names and l0 is not None:
        lab, M = gl.build_basis([n for n in names if any(l0 < lam < l1 for lam, _ in gl.GAS_LINES[n])],
                                ratios=_bool(get('SFH', 'GAS_RATIOS') or 'True'))
        lim = [x for x in lab if '/' in x or x.startswith('Balmer>=')]
        if lim:
            print('SFH gas: %d templates; ratio-limited: %s' % (len(lab), ', '.join(lim)))
    try:
        nb = int(float(get('SFH', 'BOOTSTRAP') or 0))
    except ValueError:
        nb = 0
    if nb:
        print('SFH BOOTSTRAP %d: %d extra unregularised fits per bin (errors ERR_LOGAGE/ERR_METAL/ERR_ALPHA in '
              '_sfh.fits)' % (nb, nb))


def _free_fraction(lo, hi, ivs):
    g = np.linspace(lo, hi, 801)
    cut = np.zeros(g.size, bool)
    hit = []
    for a, b, lab in ivs:
        m = (g >= a) & (g <= b)
        if m.any():
            hit.append(lab)
        cut |= m
    return 1. - cut.mean(), hit


def _win_list(v):
    if _unset(v):
        return None
    out = []
    for tok in str(v).split(','):
        if tok.strip():
            lo, hi = tok.split('-')
            out.append((float(lo), float(hi)))
    return out


a_configdir = ['configFiles']


def check_binning(row, mods, get, iv=(), snr_iv=(), lo_d=None, hi_d=None):
    """Report the binning set-up and catch the mistakes that only show at run time."""
    if 'SPATIAL_BINNING' not in mods:
        return
    meth = get('SPATIAL_BINNING', 'METHOD')
    if meth is None or str(meth).lower() == 'false':
        print('binning: SPATIAL_BINNING METHOD is %s (every spaxel is its own bin)' % meth)
        return
    present = dict((k, get('SPATIAL_BINNING', k)) for k in HIER_KEYS if not _unset(get('SPATIAL_BINNING', k)))
    print('binning: METHOD %s, TARGET_SNR %s, COVARIANCE %s%s' % (
        meth, get('SPATIAL_BINNING', 'TARGET_SNR'), get('SPATIAL_BINNING', 'COVARIANCE'),
        ''.join(', %s %s' % kv for kv in present.items()) if meth == 'hierarchical' else ''))
    try:
        import gistPipeline
        f = os.path.join(os.path.dirname(gistPipeline.__file__), 'spatialBinning', str(meth) + '.py')
        if not os.path.isfile(f):
            print('WARNING: the installed GIST has no %s: METHOD %s will fail' % (f, meth))
    except Exception:
        pass
    defaults = dict((k, v) for k, v, _ in dict(MODULES)['SPATIAL_BINNING'])
    changed = [k for k in present if k in defaults and str(present[k]) != defaults[k]]
    if meth != 'hierarchical' and changed:
        print('NOTE: %s: only read by METHOD hierarchical; %s ignores them' % (', '.join(changed), meth))
    try:
        cov = float(get('SPATIAL_BINNING', 'COVARIANCE') or 0)
    except ValueError:
        cov = None
    cfile = get('SPATIAL_BINNING', 'COV_FILE')
    if not _unset(cfile):
        fn = cfile if os.path.isabs(str(cfile)) else os.path.join(a_configdir[0], str(cfile))
        if not os.path.isfile(fn):
            print('ERROR: SPATIAL_BINNING COV_FILE %s not found (it is read from configFiles/)' % fn)
        else:
            t = np.atleast_2d(np.genfromtxt(fn, comments='#'))
            print('covariance: measured curve %s, ratio %.3f (N=%d) ... %.3f (N=%d); COVARIANCE/COV_C/COV_NMAX '
                  'ignored' % (cfile, t[0, 1], t[0, 0], t[-1, 1], t[-1, 0]))
        cov = None
    else:
        cc, cn = get('SPATIAL_BINNING', 'COV_C'), get('SPATIAL_BINNING', 'COV_NMAX')
        if not _unset(cc) or not _unset(cn):
            print('covariance: noise x %s (1 + %s log10 N), N capped at %s' % (
                '1' if _unset(cc) else cc, cov, 'never' if _unset(cn) else cn))
    if cov == 0:
        print('NOTE: COVARIANCE 0 assumes uncorrelated spaxels; drizzled KCWI cubes are not '
              '(kcwi_covar measures it from mock-noise stacks; gist_snrcheck from a run)')
    if meth != 'hierarchical':
        return

    # the spectral term
    def num(k):
        try:
            return float(get('SPATIAL_BINNING', k) or 0)
        except (TypeError, ValueError):
            return 0.
    if num('SPEC_Z') <= 0 and num('SPEC_WEIGHT') <= 0:
        return
    if cov == 0 and num('SPEC_Z') > 0:
        print('WARNING: SPEC_Z > 0 with COVARIANCE 0: the summed variance of a bin is then too small, '
              'so the spectral test refuses merges that are consistent. Measure COVARIANCE first.')
    smask = get('SPATIAL_BINNING', 'SPEC_MASK')
    if smask is not None and str(smask).lower() == 'false':
        ivs, mlab = [], 'no SPEC_MASK'
        print('NOTE: SPEC_MASK False: emission lines and sky residuals enter the spectral test (sky '
              'residuals are coherent between spaxels and make it refuse merges).')
    elif _unset(smask) or smask == get('KIN', 'SPEC_MASK'):
        ivs, mlab = list(iv), 'the KIN mask %s' % get('KIN', 'SPEC_MASK')
    else:
        ivs, mlab = None, 'SPEC_MASK %s (not written here: not checked)' % smask
    named = get('SPATIAL_BINNING', 'SPEC_FEATURES')
    named = None if _unset(named) else str(named)
    try:
        maxmask = float(get('SPATIAL_BINNING', 'SPEC_MAXMASK') or 0.5)
    except ValueError:
        maxmask = 0.5
    feats = []
    if named:
        try:
            from gistPipeline.spatialBinning.hierarchical import _features
            feats = [(n, lo, hi) for n, lo, hi in _features(named)]
        except ImportError as e:
            print('NOTE: cannot import the hierarchical module to check SPEC_FEATURES (%s)' % e)
        except ValueError as e:
            print('ERROR: %s' % e)
    feats += [('%g-%g' % w,) + tuple(w) for w in (_win_list(get('SPATIAL_BINNING', 'SPEC_WINDOWS')) or [])]
    if not feats and not named:
        l0 = get('SPATIAL_BINNING', 'SPEC_LMIN'); l1 = get('SPATIAL_BINNING', 'SPEC_LMAX')
        feats = [('range', lo_d if _unset(l0) else float(l0), hi_d if _unset(l1) else float(l1))]
    if ivs is not None:
        for nm, lo, hi in feats:
            g = np.linspace(lo, hi, 801)
            cut = (g < lo_d) | (g > hi_d)
            for a_, b_, _ in ivs:
                cut |= (g >= a_) & (g <= b_)
            frac = cut.mean()
            out = nm != 'range' and frac > maxmask
            print('spectral term %-8s %6.0f-%6.0f A: %3.0f%% masked by %s or without data%s' % (
                nm, lo, hi, 100 * frac, mlab, ' -> LEFT OUT (SPEC_MAXMASK %g)' % maxmask if out else
                (' -- WARNING: mostly masked' if frac > 0.5 else '')))
    norm = get('SPATIAL_BINNING', 'SPEC_NORM')
    if (norm is not None and str(norm).lower() == 'local') or (_unset(norm) and named):
        print('spectral term: each feature normalised by its own total (SPEC_NORM local): '
              'depths/shapes compared, not the slope between features')
        return
    if norm is not None and str(norm).lower() == 'total':
        print('spectral term: amplitude = total of the chunks (an emission line left in them sets the normalisation)')
        return
    nwin = _win_list(norm)
    s0, s1 = float(get('READ_DATA', 'LMIN_SNR')), float(get('READ_DATA', 'LMAX_SNR'))
    if nwin is None:
        nwin, nlab = [(s0, s1)], 'the S/N window'
    else:
        nlab = 'SPEC_NORM'
    nivs = list(ivs or []) + (list(snr_iv) if not _unset(get('READ_DATA', 'SNR_MASK')) else [])
    for lo, hi in nwin:
        if lo_d is not None and (hi < lo_d or lo > hi_d):
            print('WARNING: amplitude window %g-%g A (%s) is outside the data' % (lo, hi, nlab))
            continue
        free, hit = _free_fraction(lo, hi, nivs)
        print('spectral term amplitude %g-%g A (%s): %.0f%% left after the line/sky masks%s' % (
            lo, hi, nlab, 100 * free, ' -- WARNING: mostly masked' if free < 0.5 else ''))


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------
def plot_sky_variance(fn, a, iv):
    """R(lambda) = V/V_cont - 1 of the faint spaxels per channel, the threshold, and the
    sky masks it gives (observed frame)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = [l for l in ('blue', 'red') if l in a._sky_R]
    fig, axs = plt.subplots(len(labels), 1, figsize=(15, 3.2 * len(labels)), squeeze=False)
    for ax, label in zip(axs[:, 0], labels):
        w, R = a._sky_R[label]
        ax.plot(w, R, lw=0.6, color='k')
        ax.axhline(a.sky_excess, color='r', lw=1, ls='--', label='sky_excess = %g' % a.sky_excess)
        for lo, hi, lab in a._sky_iv:
            if w[0] <= hi and lo <= w[-1]:
                ax.axvspan(lo, hi, color='gold', alpha=0.4, lw=0)
        top = max(3. * a.sky_excess, np.nanpercentile(R, 99.5))
        ax.set_ylim(-0.2, top)
        ax.set_xlim(w[0], w[-1])
        ax.set_ylabel('R = V / V_cont - 1')
        ax.set_title('%s channel: sky-line share of the variance (faint spaxels); shaded = masked' % label, fontsize=9)
        ax.legend(loc='upper left', fontsize=8)
    axs[-1, 0].set_xlabel('observed wavelength [A]')
    fig.tight_layout()
    fig.savefig(fn, dpi=120)
    plt.close(fig)
    print('wrote', fn)


def plot(fn, ch, z, iv):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(15, 7))
    for label, col in (('blue', 'C0'), ('red', 'C3')):
        c = ch[label]
        if c['flux'] is None:
            continue
        f = c['flux']
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tot = np.nanmedian(f, axis=0)
            sel = np.isfinite(tot) & (tot > np.nanpercentile(tot, 75))      # brightest quarter
            spec = np.nanmean(f[:, sel], axis=1) * sel.sum()                 # masked voxels do not dent the sum
        for a_ in ax:
            a_.plot(c['wave'] / (1 + z), spec, color=col, lw=0.6)
    for a_ in ax:
        for lo, hi, lab in iv:
            a_.axvspan(lo, hi, color='grey' if 'gap' in lab.split('+') else 'gold', alpha=0.35, lw=0)
        a_.set_xlabel('rest-frame wavelength [A]')
    for lo, hi, lab in iv:
        ax[0].text(0.5 * (lo + hi), 1.0, lab, rotation=90, fontsize=5, va='top', ha='center',
                   transform=ax[0].get_xaxis_transform())
    yl = ax[1].get_ylim()
    ax[1].set_ylim(yl[0], yl[0] + 0.3 * (yl[1] - yl[0]))       # zoom on the continuum
    ax[0].set_title('sum of the brightest 25%% of spaxels; masks shaded (z = %.5f)' % z)
    fig.tight_layout()
    fig.savefig(fn, dpi=130)
    print('wrote', fn)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('parfile', nargs='?', help='parameter file ([prep] options, [MODULE] MasterConfig values); '
                    'command-line options override it')
    ap.add_argument('--example', action='store_true', help='print a complete commented parameter file '
                    '(every GIST option with its default) and exit')
    ap.add_argument('--input', help='cube base name, e.g. inputData/IRAS03582+6012 (blue; red = <input>-red)')
    ap.add_argument('--suffix-blue', default='', help='blue stack = <input><suffix>, e.g. _skl (READ_DATA SUFFIX_BLUE)')
    ap.add_argument('--suffix-red', default='', help='red stack = <input><suffix>-red (READ_DATA SUFFIX_RED)')
    ap.add_argument('--z', type=float, help='redshift used by GIST (GENERAL REDSHIFT)')
    ap.add_argument('--tag', help='suffix of the files written')
    ap.add_argument('--configdir', default='configFiles')
    ap.add_argument('--header-only', action='store_true', help='gap from WAVGOOD keywords instead of the reader')
    ap.add_argument('--line-hw', type=float, default=500., help='half-width of line masks [km/s] (default 500)')
    ap.add_argument('--broad', default='', help='comma list of line names masked with --broad-hw (e.g. Hb,Ha)')
    ap.add_argument('--broad-hw', type=float, default=3000., help='half-width for --broad lines [km/s]')
    ap.add_argument('--drop', default='', help='comma list of line names not to mask (e.g. NaD)')
    ap.add_argument('--lines-from-emi', action='store_true', help="mask the 'f' lines of --emi instead of the built-in list")
    ap.add_argument('--sky', choices=['variance', 'model', 'list', 'none'], default='variance',
                    help='sky-line masks: variance = where sky lines add more than sky_excess to the '
                         'variance of the faint spaxels, measured in the cube (default); model = every line '
                         'of sky_lines_paranal.dat brighter, at the instrument resolution, than sky_thresh x '
                         '[OI]5577; list = the short built-in list +- sky_hw; none')
    ap.add_argument('--sky-excess', type=float, default=0.25,
                    help='sky = variance: mask where the sky lines raise the variance by more than this '
                         'fraction (R = V/V_cont - 1; default 0.25)')
    ap.add_argument('--sky-thresh', type=float, default=0.02,
                    help='sky = model: mask where the sky lines exceed this fraction of the [OI]5577 peak '
                         '(default 0.02; smaller masks more)')
    ap.add_argument('--sky-fwhm', default='auto', help='sky = model: instrument FWHM: auto (LSF_DATA file), '
                    'a number [A], or R<resolving power>')
    ap.add_argument('--sky-hw', type=float, default=4., help='sky = list (and sky_file): half-width of each '
                    'sky-line mask [A, observed] (default 4)')
    ap.add_argument('--sky-file', help='extra observed-frame sky lines, one wavelength per line (+- sky_hw)')
    ap.add_argument('--no-sky', action='store_true')
    ap.add_argument('--no-telluric', action='store_true', help='do not mask the O2 A/B bands')
    ap.add_argument('--gap-pad', type=float, default=5., help='rest-frame A added on each side of the gap')
    ap.add_argument('--snr-line-hw', type=float, default=1000., help='S/N mask: emission-line half-width [km/s] '
                    '(broad lines keep --broad-hw; default 1000)')
    ap.add_argument('--no-snr-mask', action='store_true', help='do not write specMask_SNR_<tag> / set READ_DATA SNR_MASK')
    ap.add_argument('--snr-auto', choices=['blue', 'red', 'any'], help='set LMIN_SNR/LMAX_SNR to the cleanest '
                    'window in that channel')
    ap.add_argument('--snr-width', type=float, default=300., help='width of the suggested S/N windows [A rest]')
    ap.add_argument('--ebv-dir', default='', help='READ_DATA EBV = auto: folder with the SFD maps '
                    '(SFD_dust_4096_ngp.fits, _sgp.fits)')
    ap.add_argument('--ebv-scale', type=float, default=0.86, help='READ_DATA EBV = auto: E(B-V) = this x SFD '
                    '(default 0.86, Schlafly & Finkbeiner 2011)')
    ap.add_argument('--extra', default='', help='extra rest-frame intervals, e.g. 8432-8459,8567-8631')
    ap.add_argument('--emi', help='GandALF emissionLines.config to copy for the GAS module')
    ap.add_argument('--gas-protect', type=float, default=500., help='GAS: no sky/telluric mask within this many km/s of a fitted line')
    ap.add_argument('--template', help='take the run row from this MasterConfig instead of the built-in defaults')
    ap.add_argument('--no-config', action='store_true', help='write the masks only, no MasterConfig')
    ap.add_argument('--from-row', type=int, default=-1, help='template row to start from (default last)')
    ap.add_argument('--run-id', help='GENERAL RUN_ID of the new row')
    ap.add_argument('--set', action='append', default=[], metavar='MODULE.KEY=VALUE',
                    help='override / add a MasterConfig value; repeatable')
    ap.add_argument('--append', action='store_true', help='keep the template rows and append the new one')
    ap.add_argument('--plot', action='store_true', help='write masks_<tag>.png (needs the reader) and origin_<tag>.png')
    a0, _ = ap.parse_known_args()
    if a0.example:
        sys.stdout.write(example_par())
        return
    par_sets = []
    if a0.parfile:
        prep, par_sets = read_par(a0.parfile, ap)
        ap.set_defaults(**prep)          # string defaults are converted with each option's type
    a = ap.parse_args()
    cli_sets = list(a.set)
    a.set = par_sets + cli_sets          # command line last: it wins
    miss = [o for o in ('input', 'z', 'tag') if getattr(a, o) is None]
    if miss:
        ap.error('need %s (on the command line or in [prep] of a parameter file)' % ', '.join(miss))

    z = a.z
    ch, exact = channel_ranges(a.input, a.header_only, suffix_blue=a.suffix_blue, suffix_red=a.suffix_red)
    print('cubes: blue %s, red %s' % (a.input + a.suffix_blue, a.input + a.suffix_red + '-red'))
    gap = gap_rest(ch, z, a.gap_pad)
    print('blue %.1f-%.1f A, red %.1f-%.1f A observed (%s)' % (ch['blue']['w0'], ch['blue']['w1'], ch['red']['w0'],
          ch['red']['w1'], 'reader trimming' if exact else 'WAVGOOD headers'))
    print('data %.1f-%.1f A rest; gap %s' % (ch['blue']['w0'] / (1 + z), ch['red']['w1'] / (1 + z),
          'none' if gap is None else '%.1f-%.1f A rest (incl. %.0f A pad)' % (gap[0], gap[1], a.gap_pad)))

    raw = build_intervals(a, z, ch, gap)
    iv = merge(raw)
    names = {'kin': 'specMask_KIN_' + a.tag, 'sfh': 'specMask_SFH_' + a.tag}
    write_specmask(os.path.join(a.configdir, names['kin']), iv, z, 'KIN')
    # SFH: lines fitted together with the stars (SFH GAS_LINES) are NOT masked there
    gas_names, gl = sfh_gas_lines(a)
    if gas_names:
        balmer = set(gas_names) & gl.BALMER
        broad = set(x.strip() for x in a.broad.split(',') if x.strip())
        if balmer and broad & set(n for n, _ in LINES if n.startswith('H')):
            raise SystemExit('SFH GAS_LINES includes Balmer lines (%s) but broad = %s declares broad Balmer '
                             'emission: a narrow gas component cannot fit it. Drop "balmer" from GAS_LINES.'
                             % (', '.join(sorted(balmer)), a.broad))
        gas_wl = gl.gas_wavelengths(gas_names)
        raw_sfh = [x for x in raw if not (x[3] == 'line' and any(abs(0.5 * (x[0] + x[1]) - g) < 1. for g in gas_wl))]
        iv_sfh = merge(raw_sfh)
        print('SFH: %d gas templates (%d lines) fitted with the stars (GAS_LINES), unmasked in %s: %s%s' % (
            len(gas_names), len(gas_wl), names['sfh'], ', '.join(gas_names),
            '' if not balmer else ' -- Balmer emission fitted as NARROW: only valid without a broad component'))
        for nm in gas_names:
            for lam, _ in gl.GAS_LINES[nm]:
                hit = [lab for lo, hi, lab in iv_sfh if lo <= lam <= hi]
                if hit:
                    print('NOTE: %s %.1f A falls on the SFH mask (%s): those pixels do not constrain it'
                          % (nm, lam, ', '.join(hit)))
    else:
        iv_sfh = iv
    write_specmask(os.path.join(a.configdir, names['sfh']), iv_sfh, z, 'SFH')
    print('wrote %s, %s (%d intervals, %.0f A masked)' % (names['kin'], names['sfh'], len(iv), sum(h - l for l, h, _ in iv)))
    # S/N mask: the same features, emission lines wider (wings, outflows), gap excluded
    # (the reader drops gap pixels itself) -- for READ_DATA SNR_MASK
    b = copy.copy(a)
    b.line_hw = max(a.line_hw, a.snr_line_hw)
    snr_iv = merge([x for x in build_intervals(b, z, ch, gap) if x[3] != 'gap'])
    names['snr_iv'] = snr_iv
    if not a.no_snr_mask:
        names['snr'] = 'specMask_SNR_' + a.tag
        write_specmask(os.path.join(a.configdir, names['snr']), snr_iv, z, 'S/N (READ_DATA SNR_MASK)')
        print('wrote %s (%d intervals; lines +-%.0f km/s)' % (names['snr'], len(snr_iv), b.line_hw))

    if a.emi:
        names['emi'] = 'emissionLines_%s.config' % a.tag
        nonline = gas_masks(a, raw)
        ign = write_emi(a.emi, os.path.join(a.configdir, names['emi']), nonline, z, ch, gap)
        print('wrote %s (%d mask entries; ignored for lack of data: %s)' % (names['emi'], len(nonline),
              ', '.join('%s %s' % (n, l) for _, n, l in ign) or 'none'))

    if not a.no_config:
        print('wrote', make_config(a, z, ch, gap, names, iv))

    if a.plot and getattr(a, '_sky_R', None):
        plot_sky_variance(os.path.join(a.configdir, 'sky_%s.png' % a.tag), a, iv)
    if a.plot:
        if exact:
            plot(os.path.join(a.configdir, 'masks_%s.png' % a.tag), ch, z, iv)
        else:
            print('--plot needs the reader (no --header-only, gistPipeline importable)')

    used = os.path.join(a.configdir, 'gist_prep_%s.in' % a.tag)
    write_used_par(used, a, ap, par_sets, cli_sets)
    print('wrote %s (the settings used)' % used)

    print('\nrest-frame masks:')
    for lo, hi, lab in iv:
        print('  %8.1f - %8.1f  %s' % (lo, hi, lab))


if __name__ == '__main__':
    main()
