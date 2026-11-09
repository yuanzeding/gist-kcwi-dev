#!/usr/bin/env python
"""
get_ssp.py -- download E-MILES SSP models from the MILES team's IAC share and set
them up as a GIST template library, in one step, on the machine that runs GIST.

    get_ssp                                   # BaSTI, Salpeter (UN), 5 metallicities -> ./spectralTemplates/EMILES_BASTI_UN/
    get_ssp --list                            # what the archive holds (downloads it first, once)
    get_ssp --imf KU --metals all             # Kroupa, every metallicity
    get_ssp --iso PADOVA00                    # Padova00 isochrones (for an isochrone comparison)
    get_ssp --source ~/Downloads/EMILES_BASTI_BASE_UN_FITS.tar.gz   # an archive you already have
    get_ssp --remote-list                     # what the IAC share holds (names, sizes), e.g. the
                                              #   alpha-variable (Ep0.00 / Ep0.40) archives
    get_ssp --archive <NAME>.tar.gz --metals -0.35,0.06,0.26,0.40 --alpha Ep0.00,Ep0.40
                                              # any archive of the share, by the name --remote-list shows
    get_ssp --from-dir ~/software/SSP/EMILES --list            # an unpacked folder: what it holds
    get_ssp --from-dir ~/software/SSP/EMILES --ages dex:0.1    # -> ~/software/SSP/EMILES_BASTI_UN1.30_baseFe_dex0.1/
                                              # (symbolic links, one library for every target)

Libraries are built in ONE shared place (default ~/software/SSP/<name>/), not per target:
the MasterConfig points at them with absolute paths (SFH LIBRARY, LSF_TEMP).

What it does
  1. The source: --from-dir (an unpacked folder; files are symbolically linked, not
     copied), or an archive: EMILES_<ISO>_BASE_<IMF>_FITS.tar.gz (native resolution,
     2.51 A FWHM in the optical; the _LIS*/_Sigma228 versions are smoothed for line
     indices -- not wanted for full-spectrum fitting), downloaded into --cache with curl
     (resumes after an interruption; a complete archive is not downloaded again).
  2. Selects ONE IMF slope (--slope, default 1.30: the UN/BI archives hold every slope),
     the --metals, --alpha tags and --ages, and checks that they form a complete grid.
  3. Puts them, flat and with the original MILES names (as GIST's miles reader expects),
     into the library folder, and writes lsf_MILES (2.51 A) next to the libraries.

IMF codes: UN = unimodal (slope 1.30 = Salpeter, as the XSL files used here), KU =
Kroupa universal, KB = Kroupa revised, BI = bimodal, CH = Chabrier.

baseFe vs alpha-variable. The BASE models have the abundance pattern of the MILES
stars (about scaled-solar at solar [M/H], alpha-enhanced at low [M/H]); the Ep0.00 /
Ep0.40 models are scaled-solar and [alpha/Fe] = +0.4 (Vazdekis et al. 2015). GIST's
miles reader refuses a folder that mixes baseFe with Ep files, so unpack them into
different folders (the default folder name follows the archive name).
"""
from __future__ import print_function

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile

import numpy as np

SHARE = 'https://cloud.iac.es/public.php/dav/files/aYECNyEQfqgYwt4/E-MILES/'
DEFAULT_METALS = '-0.66,-0.35,0.06,0.26,0.40'
LSF_MILES = ('# Column 1: Wavelength vector in Angstroem; Column 2: FWHM of templates for the respective '
             'wavelength\n# E-MILES: 2.51 A FWHM in the optical (Falcon-Barroso et al. 2011; Vazdekis et al. 2016)\n'
             '3500.0  2.51\n9000.0  2.51\n')
NAME = re.compile(r'Z([mp])(\d+\.\d+)T(\d+\.\d+)')
ALPHA = re.compile(r'_E[mp]\d\.\d\d')


def _metal_token(v):
    """-0.35 -> 'Zm0.35', 0.06 -> 'Zp0.06' (MILES file-name convention)."""
    x = float(v)
    return 'Z%s%.2f' % ('m' if x < 0 else 'p', abs(x))


def download(url, dest):
    """curl with resume; True when the file is complete."""
    if shutil.which('curl') is None:
        sys.exit('curl is not available on this machine')
    tmp = dest + '.part'
    if os.path.isfile(dest):
        return True
    print('downloading %s\n         -> %s' % (url, dest))
    rc = subprocess.call(['curl', '-L', '--fail', '-C', '-', '--retry', '5', '--retry-delay', '10',
                          '-o', tmp, url])
    if rc not in (0, 33):          # 33: server does not support resume -- curl restarts
        sys.exit('download interrupted or failed (curl exit %d); run the same command again to resume' % rc)
    os.rename(tmp, dest)
    return True


def remote_list(sub=''):
    """Names and sizes of the files in a folder of the share (WebDAV PROPFIND)."""
    import xml.etree.ElementTree as ET
    if shutil.which('curl') is None:
        sys.exit('curl is not available on this machine')
    url = SHARE + (sub.strip('/') + '/' if sub else '')
    r = subprocess.run(['curl', '-s', '-L', '--fail', '-X', 'PROPFIND', '-H', 'Depth: 1', url],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        sys.exit('PROPFIND %s failed (curl exit %d): %s' % (url, r.returncode, r.stderr.decode(errors='replace')))
    ns = {'d': 'DAV:'}
    root = ET.fromstring(r.stdout)
    rows = []
    for resp in root.findall('d:response', ns):
        href = resp.find('d:href', ns).text
        size = resp.find('.//d:getcontentlength', ns)
        isdir = resp.find('.//d:resourcetype/d:collection', ns) is not None
        name = href.rstrip('/').split('/')[-1]
        rows.append((name + ('/' if isdir else ''), int(size.text) if size is not None and size.text else 0))
    print('%s:' % url)
    for name, size in rows[1:]:
        print('  %-60s %8.1f MB' % (name, size / 1e6) if size else '  %s' % name)


SLOPE = re.compile(r'^E([a-z]{2})(\d\.\d\d)Z')


def _parse(base):
    """(imf code, slope, [M/H], age, alpha tag) of an E-MILES file name, or None."""
    g = NAME.search(base)
    if not g or not base.endswith('.fits'):
        return None
    sl = SLOPE.search(base)
    al = ALPHA.search(base)
    return (sl.group(1) if sl else '?', sl.group(2) if sl else '?',
            round((-1 if g.group(1) == 'm' else 1) * float(g.group(2)), 2), float(g.group(3)),
            'baseFe' if 'baseFe' in base else (al.group(0)[1:] if al else '?'))


def _thin_ages(ages, spec):
    """'all', 'dex:0.1' (keep ages at least 0.1 dex apart, first and last always kept) or a list [Gyr]."""
    ages = sorted(ages)
    if spec in (None, '', 'all'):
        return ages
    if spec.startswith('dex:'):
        step = float(spec[4:])
        keep = [ages[0]]
        for t in ages[1:-1]:
            if np.log10(t) - np.log10(keep[-1]) >= step - 1e-6 and np.log10(ages[-1]) - np.log10(t) >= 0.5 * step:
                keep.append(t)
        return keep + [ages[-1]]
    want = [float(x) for x in spec.split(',') if x.strip()]
    out = []
    for w in want:                                  # nearest available age
        out.append(min(ages, key=lambda t: abs(np.log10(t) - np.log10(w))))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--iso', default='BASTI', choices=['BASTI', 'PADOVA00'], help='isochrones (default BASTI)')
    ap.add_argument('--imf', default='UN', choices=['UN', 'KU', 'KB', 'BI', 'CH'], help='IMF (default UN = Salpeter)')
    ap.add_argument('--slope', default='1.30', help='IMF slope in the file names (default 1.30 = Salpeter for UN; '
                    'the UN/BI archives hold every slope 0.30-3.30)')
    ap.add_argument('--metals', default=DEFAULT_METALS,
                    help='[M/H] values, comma-separated, or "all" (default %s)' % DEFAULT_METALS)
    ap.add_argument('--alpha', help='alpha tags, e.g. Ep0.00,Ep0.40 (default: all in the source; baseFe and Ep '
                    'files cannot share a library)')
    ap.add_argument('--ages', default='all', help='all (default), dex:0.1 (ages >= 0.1 dex apart), or a list in Gyr')
    ap.add_argument('--from-dir', help='an already unpacked E-MILES folder (searched recursively) instead of an archive')
    ap.add_argument('--copy', action='store_true', help='--from-dir: copy the files (default: symbolic links, no '
                    'duplication)')
    ap.add_argument('--dest', help='library folder (default: <cache>/EMILES_<ISO>_<IMF><slope>_<alpha>[_<ages>]); '
                    'one shared folder for all targets')
    ap.add_argument('--replace', action='store_true', help='empty an existing --dest of its .fits files/links first')
    ap.add_argument('--cache', default=os.path.expanduser('~/software/SSP'), help='where the archive and the '
                    'libraries are kept (default ~/software/SSP)')
    ap.add_argument('--configdir', default='', help='also write lsf_MILES here (default: only in --cache)')
    ap.add_argument('--source', help='use this local archive instead of downloading')
    ap.add_argument('--list', action='store_true', help='list what the source holds and stop')
    ap.add_argument('--remote-list', nargs='?', const='', metavar='SUBFOLDER',
                    help='list the files of the IAC share (WebDAV) and stop')
    ap.add_argument('--archive', help='download this archive of the share (name as --remote-list shows it) '
                    'instead of EMILES_<ISO>_BASE_<IMF>_FITS.tar.gz')
    argv = sys.argv[1:]
    for k in range(len(argv) - 1):              # allow '--metals -0.66,...' (argparse would take it for an option)
        if argv[k] == '--metals' and argv[k + 1].startswith('-'):
            argv[k:k + 2] = ['--metals=' + argv[k + 1]]
            break
    a = ap.parse_args(argv)
    a.cache = os.path.abspath(os.path.expanduser(a.cache))

    if a.remote_list is not None:
        remote_list(a.remote_list)
        return

    # --- the source: an unpacked folder, or an archive (downloaded if needed)
    tf = None
    if a.from_dir:
        src = os.path.abspath(os.path.expanduser(a.from_dir))
        if not os.path.isdir(src):
            sys.exit('%s is not a folder' % src)
        items = []
        for root, dirs, files in os.walk(src):
            for f in files:
                p = _parse(f)
                if p:
                    items.append((os.path.join(root, f), f, p))
        label = src
    else:
        fname = a.archive or 'EMILES_%s_BASE_%s_FITS.tar.gz' % (a.iso, a.imf)
        if a.source:
            arch = os.path.expanduser(a.source)
            if not os.path.isfile(arch):
                sys.exit('%s not found' % arch)
        else:
            if not os.path.isdir(a.cache):
                os.makedirs(a.cache)
            arch = os.path.join(a.cache, os.path.basename(fname))
            download(SHARE + fname, arch)
        try:
            tf = tarfile.open(arch, 'r:*')
            items = [(m, os.path.basename(m.name), _parse(os.path.basename(m.name)))
                     for m in tf.getmembers() if m.isfile()]
            items = [x for x in items if x[2]]
        except (tarfile.TarError, OSError, EOFError) as e:
            sys.exit('%s does not open as a tar archive (%s): delete it and download again' % (arch, e))
        label = os.path.basename(arch)
    if not items:
        sys.exit('no E-MILES-named .fits files found in %s' % label)

    def summary(its, head):
        sl = sorted(set(p[1] for _, _, p in its)); me = sorted(set(p[2] for _, _, p in its))
        ag = sorted(set(p[3] for _, _, p in its)); al = sorted(set(p[4] for _, _, p in its))
        imf = sorted(set(p[0] for _, _, p in its))
        print('%s: %d spectra; IMF %s, slopes %s; [M/H] %s; %d ages %.4g-%.4g Gyr; alpha %s' % (
            head, len(its), ','.join(imf), ','.join(sl), ', '.join('%+.2f' % z for z in me), len(ag), ag[0], ag[-1],
            ', '.join(al)))
        return me, ag
    metals, ages_all = summary(items, label)
    if a.list:
        return

    # --- selection: IMF slope, [M/H], alpha, ages
    sel = [x for x in items if x[2][1] == a.slope]
    if not sel:
        sys.exit('no files with IMF slope %s (see --list)' % a.slope)
    if a.metals.strip().lower() != 'all':
        want = [round(float(z), 2) for z in a.metals.split(',') if z.strip()]
        miss = [z for z in want if z not in metals]
        if miss:
            sys.exit('not in the source: [M/H] %s (available: %s)' % (', '.join('%+.2f' % z for z in miss),
                                                                      ', '.join('%+.2f' % z for z in metals)))
        sel = [x for x in sel if x[2][2] in want]
    tags = sorted(set(x[2][4] for x in sel))
    if a.alpha:
        want_a = [t.strip() for t in a.alpha.split(',') if t.strip()]
        sel = [x for x in sel if x[2][4] in want_a]
        tags = sorted(set(x[2][4] for x in sel))
        if not sel:
            sys.exit('no files with alpha tag %s' % a.alpha)
    if 'baseFe' in tags and len(tags) > 1:
        sys.exit('the selection mixes baseFe and %s: GIST cannot use both in one library -- choose with --alpha'
                 % ', '.join(t for t in tags if t != 'baseFe'))
    keep_ages = _thin_ages(sorted(set(x[2][3] for x in sel)), a.ages)
    sel = [x for x in sel if x[2][3] in set(keep_ages)]
    # a complete grid, each combination once
    combos = {}
    for x in sel:
        k = x[2][2:5]
        if k in combos:
            sys.exit('two files for [M/H] %+.2f, %g Gyr, %s: %s and %s' % (k[0], k[1], k[2], combos[k][1], x[1]))
        combos[k] = x
    nm, na, nt = len(set(k[0] for k in combos)), len(set(k[1] for k in combos)), len(set(k[2] for k in combos))
    if len(combos) != nm * na * nt:
        sys.exit('incomplete grid: %d files for %d [M/H] x %d ages x %d alpha' % (len(combos), nm, na, nt))
    summary(sel, 'selected')

    # --- the library folder (shared by all targets)
    imfc = sel[0][2][0].upper()
    isos = set('BASTI' if '_iT' in x[1] else ('PADOVA00' if '_iP' in x[1] else a.iso) for x in sel)
    iso = isos.pop() if len(isos) == 1 else a.iso
    agetag = '' if a.ages in ('', 'all') else '_' + a.ages.replace(':', '').replace(',', '-')[:20]
    dest = os.path.abspath(os.path.expanduser(a.dest)) if a.dest else os.path.join(
        a.cache, 'EMILES_%s_%s%s_%s%s' % (iso, imfc, a.slope, '+'.join(tags), agetag))
    if os.path.isdir(dest):
        old = [f for f in os.listdir(dest) if f.endswith('.fits')]
        if old and not a.replace:
            new_names = set(x[1] for x in sel)
            if set(old) != new_names:
                sys.exit('%s already holds %d other templates: use --replace to rebuild it, or another --dest'
                         % (dest, len(old)))
        for f in old if a.replace else []:
            os.remove(os.path.join(dest, f))
    else:
        os.makedirs(dest)
    n = 0
    for obj, base, p in sel:
        out = os.path.join(dest, base)
        if os.path.lexists(out):
            n += 1; continue
        if tf is not None:
            with tf.extractfile(obj) as s_, open(out, 'wb') as d_:
                shutil.copyfileobj(s_, d_)
        elif a.copy:
            shutil.copy2(obj, out)
        else:
            os.symlink(obj, out)
        n += 1
    if tf is not None:
        tf.close()
    how = 'extracted' if tf is not None else ('copied' if a.copy else 'linked')
    print('%s %d spectra (%d ages x %d [M/H] x %d alpha) into %s/' % (how, n, na, nm, nt, dest))

    lsf_files = [os.path.join(a.cache, 'lsf_MILES')] + ([os.path.join(a.configdir, 'lsf_MILES')] if a.configdir else [])
    for lsf in lsf_files:
        if os.path.isdir(os.path.dirname(lsf)) and not os.path.isfile(lsf):
            with open(lsf, 'w') as f:
                f.write(LSF_MILES)
            print('wrote %s (2.51 A FWHM)' % lsf)
    print('\ngist_prep [SFH] for this library (absolute paths: GIST joins them to templateDir / configDir '
          'unchanged, so one copy serves every target):\n'
          '    TEMPLATES = miles\n    LIBRARY   = %s/\n    LSF_TEMP  = %s\n    MATCH_RES = auto'
          % (dest, lsf_files[0]))


if __name__ == '__main__':
    main()
