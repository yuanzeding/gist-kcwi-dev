"""
Narrow emission lines fitted together with the stellar populations in the SFH
module (SFH GAS_LINES), and the list gist_prep.py uses to leave exactly those
lines unmasked in specMask_SFH_<tag>. One table for both, so the lines that are
fitted and the lines that are unmasked always agree.

Wavelengths: REST-FRAME AIR [A] (as the KCWI cubes and gist_prep's masks).

Each entry is ONE gas template: a list of (wavelength, relative FLUX). Lines in
one template share a single amplitude, so only ratios fixed by atomic physics are
tied:
  [OIII] 4959/5007 = 1/2.98, [NII] 6548/6583 = 1/2.94, [OI] 6364/6300 = 1/3.0,
  [NeIII] 3968/3869 = 1/3.32 (A-values; Osterbrock & Ferland 2006), and, in the
  'Balmer' template only, the whole Balmer series at the Case B decrement
  (T = 1e4 K, n_e = 100 cm^-3; Storey & Hummer 1995), reddened in the fit by
  SFH GAS_REDDENING.
Density-sensitive pairs ([OII] 3726/3729, [SII] 6716/6731, [SII] 4069/4076,
[NI] 5198/5200) are separate lines; with SFH GAS_RATIOS (default) their ratio is kept
inside the range atomic physics allows (RATIO_BOUNDS), see build_basis.

Sets for SFH GAS_LINES (combine with commas, e.g. forbidden,balmer_tied):
  forbidden    [OII], [NeIII], [SII]4069/76, [OIII]4363, [OIII], [FeIII]4658, [NI],
               [OI], [NII], [SII], [ArIII]
  balmer       H10 ... Hbeta, Halpha as SEPARATE lines (free fluxes; with GAS_RATIOS
               each bluer line is limited to <= 1.15 x its Case B ratio to the next
               redder one -- dust can only lower it)
  balmer_tied  the single 'Balmer' template (H16 ... Halpha at Case B): one flux for
               the series, the decrement set by GAS_REDDENING. Far less degenerate
               with the stellar Balmer absorption than free lines (the high-order
               lines, where the absorption is strongest, are tied to Hbeta/Hgamma).
  helium       He I 4026, 4471, 4922, 5016, 5876, 6678, 7065 and He II 4686
               (permitted lines: narrow in a Seyfert 2, broad in a type 1)
  balmer and balmer_tied are alternatives (not both). Only for narrow Balmer
  emission: a broad component would be absorbed into the stellar absorption.
"""
from collections import OrderedDict

# Case B Balmer decrement relative to Hbeta (T = 1e4 K, n_e = 100 cm^-3)
_CASEB = OrderedDict([
    ('H16', (3703.86, 0.0127)), ('H15', (3711.97, 0.0155)), ('H14', (3721.94, 0.0192)),
    ('H13', (3734.37, 0.0241)), ('H12', (3750.15, 0.0307)), ('H11', (3770.63, 0.0398)),
    ('H10', (3797.90, 0.0530)), ('H9', (3835.39, 0.0731)), ('H8', (3889.05, 0.105)),
    ('Heps', (3970.07, 0.159)), ('Hdelta', (4101.74, 0.259)), ('Hgamma', (4340.47, 0.468)),
    ('Hbeta', (4861.33, 1.0)), ('Halpha', (6562.80, 2.86)),
])

GAS_LINES = OrderedDict([
    ('[OII]3726',   [(3726.03, 1.)]),
    ('[OII]3729',   [(3728.82, 1.)]),
    ('[NeIII]3869', [(3868.76, 1.), (3967.47, 1. / 3.32)]),
    ('[SII]4069',   [(4068.60, 1.)]),
    ('[SII]4076',   [(4076.35, 1.)]),
    ('[OIII]4363',  [(4363.21, 1.)]),
    ('[FeIII]4658', [(4658.10, 1.)]),
    ('HeII4686',    [(4685.71, 1.)]),     # permitted, can be broad in AGN: not in 'forbidden'
    ('[ArIV]4711',  [(4711.37, 1.)]),     # blends with HeI 4713: not in 'forbidden'
    ('[ArIV]4740',  [(4740.20, 1.)]),
    ('[OIII]5007',  [(5006.84, 1.), (4958.91, 1. / 2.98)]),
    ('[NI]5198',    [(5197.90, 1.)]),
    ('[NI]5200',    [(5200.26, 1.)]),
    ('[OI]6300',    [(6300.30, 1.), (6363.78, 1. / 3.0)]),
    ('[NII]6583',   [(6583.45, 1.), (6548.05, 1. / 2.94)]),
    ('[SII]6716',   [(6716.44, 1.)]),
    ('[SII]6731',   [(6730.82, 1.)]),
    ('[ArIII]7136', [(7135.80, 1.)]),
    ('HeI4026',     [(4026.19, 1.)]),
    ('HeI4471',     [(4471.48, 1.)]),
    ('HeI4922',     [(4921.93, 1.)]),
    ('HeI5016',     [(5015.68, 1.)]),
    ('HeI5876',     [(5875.62, 1.)]),
    ('HeI6678',     [(6678.15, 1.)]),
    ('HeI7065',     [(7065.19, 1.)]),
    ('H10',         [(3797.90, 1.)]),
    ('H9',          [(3835.39, 1.)]),
    ('H8',          [(3889.05, 1.)]),
    ('Heps',        [(3970.07, 1.)]),
    ('Hdelta',      [(4101.74, 1.)]),
    ('Hgamma',      [(4340.47, 1.)]),
    ('Hbeta',       [(4861.33, 1.)]),
    ('Halpha',      [(6562.80, 1.)]),
    # the whole series in one template, Case B; first entry = the named line (Hbeta)
    ('Balmer',      [_CASEB['Hbeta']] + [v for k, v in _CASEB.items() if k != 'Hbeta']),
])

GAS_SETS = OrderedDict([
    ('forbidden', ['[OII]3726', '[OII]3729', '[NeIII]3869', '[SII]4069', '[SII]4076', '[OIII]4363',
                   '[FeIII]4658', '[OIII]5007', '[NI]5198', '[NI]5200', '[OI]6300', '[NII]6583',
                   '[SII]6716', '[SII]6731', '[ArIII]7136']),
    ('balmer', ['H10', 'H9', 'H8', 'Heps', 'Hdelta', 'Hgamma', 'Hbeta', 'Halpha']),
    ('balmer_tied', ['Balmer']),
    ('helium', ['HeI4026', 'HeI4471', 'HeI4922', 'HeI5016', 'HeI5876', 'HeI6678', 'HeI7065', 'HeII4686']),
])
BALMER = set(GAS_SETS['balmer']) | {'Balmer'}

# Density-sensitive ratios allowed by atomic physics: lo <= F(a)/F(b) <= hi
# (Osterbrock & Ferland 2006; Sanders et al. 2016 for [OII]; slightly widened)
RATIO_BOUNDS = [
    ('[OII]3729', '[OII]3726', 0.35, 1.50),
    ('[SII]6716', '[SII]6731', 0.44, 1.45),
    ('[SII]4076', '[SII]4069', 0.25, 0.45),
    ('[NI]5198', '[NI]5200', 0.40, 2.50),       # loose: weak lines, only keeps them physical
]
# free Balmer lines: F(bluer)/F(next redder) <= BALMER_TOL x Case B (dust only lowers it)
BALMER_TOL = 1.15
# gas templates are scaled so that weight x REF_LAM x (1+z) velscale/c = flux of the named line
REF_LAM = 5000.


def build_basis(names, ratios=True):
    """
    The gas templates actually given to pPXF, as non-negative combinations of the
    lines `names` (GAS_LINES entries). Returns (labels, M): M[j, b] = flux of line j in
    basis template b (in units of its weight), so line fluxes = M @ weights.
      ratios = False: one basis template per line (M = identity).
      ratios = True (SFH GAS_RATIOS): a ratio-limited pair (RATIO_BOUNDS) becomes two
        templates with F_a/F_b = lo and = hi -- any non-negative combination has its
        ratio inside [lo, hi] (pPXF's limit_doublets idea; keeps the fast, robust
        bounded least squares instead of general inequality constraints); free Balmer
        lines (blue -> red) become the cone of F_k <= r_k F_k+1 (r_k = BALMER_TOL x
        Case B): template k = line k at 1 with every redder line at the minimum the
        decrement allows. Fluxes are then sums over templates; their formal errors are
        added in quadrature (the covariance of the templates is ignored).
    """
    names = list(names)
    j_of = dict((n, j) for j, n in enumerate(names))
    cols, labels, used = [], [], set()
    if ratios:
        for a, b, lo, hi in RATIO_BOUNDS:
            if a in j_of and b in j_of and a not in used and b not in used:
                for r in (lo, hi):
                    c = [0.] * len(names); c[j_of[a]] = r; c[j_of[b]] = 1.
                    cols.append(c); labels.append('%s/%s=%.2f' % (a, b, r))
                used |= {a, b}
        chain = [n for n in _CASEB if n in j_of and n != 'Balmer' and n not in used]
        if len(chain) > 1:
            for k, n in enumerate(chain):
                c = [0.] * len(names); c[j_of[n]] = 1.
                f = 1.
                for blue, red in zip(chain[k:-1], chain[k + 1:]):
                    f /= BALMER_TOL * _CASEB[blue][1] / _CASEB[red][1]
                    c[j_of[red]] = f
                cols.append(c); labels.append('Balmer>=' + n)
            used |= set(chain)
    for n in names:
        if n not in used:
            c = [0.] * len(names); c[j_of[n]] = 1.
            cols.append(c); labels.append(n)
    import numpy as np
    return labels, np.array(cols, float).T


def parse_gas_lines(v):
    """SFH GAS_LINES -> list of template names (in table order), or [] if unset.
    Accepts set names (forbidden, balmer, balmer_tied, helium) and template names,
    comma-separated, one word (MasterConfig), e.g. 'forbidden', 'forbidden,balmer_tied',
    '[OII]3726,[OII]3729,[OIII]5007'."""
    if v is None or v is False or str(v).strip().lower() in ('', 'none', 'false'):
        return []
    lower = dict((k.lower(), k) for k in GAS_LINES)
    want = []
    for tok in str(v).replace(' ', '').split(','):
        if not tok:
            continue
        if tok.lower() in GAS_SETS:
            want += GAS_SETS[tok.lower()]
        elif tok.lower() in lower:
            want.append(lower[tok.lower()])
        else:
            raise ValueError('SFH GAS_LINES: unknown line %r (known: %s; sets: %s)'
                             % (tok, ', '.join(GAS_LINES), ', '.join(GAS_SETS)))
    want = set(want)
    if 'Balmer' in want and want & (BALMER - {'Balmer'}):
        raise ValueError('SFH GAS_LINES: balmer_tied (the single Case-B Balmer template) and separate '
                         'Balmer lines (%s) cannot be fitted together' % ', '.join(sorted(want & (BALMER - {'Balmer'}))))
    return [k for k in GAS_LINES if k in want]


def gas_wavelengths(names):
    """All rest-frame wavelengths of the given templates."""
    return [lam for n in names for lam, _ in GAS_LINES[n]]
