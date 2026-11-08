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

Run it from the target directory (the one with configFiles/ and spectralTemplates/).

What it does
  1. Downloads EMILES_<ISO>_BASE_<IMF>_FITS.tar.gz (native resolution, 2.51 A FWHM in
     the optical; the _LIS*/_Sigma228 versions are smoothed for line indices and the
     _LINE folders are index tables -- not wanted for full-spectrum fitting) into a
     cache folder (default ~/software/SSP), with curl and resuming after an interruption
     (just run it again). A complete archive already in the cache is not downloaded again.
  2. Checks that the archive opens.
  3. Unpacks only the chosen metallicities, all ages, flat into the template folder,
     with the original MILES file names (as GIST's miles.py reader expects).
  4. Writes configFiles/lsf_MILES (2.51 A) if it is missing.

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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--iso', default='BASTI', choices=['BASTI', 'PADOVA00'], help='isochrones (default BASTI)')
    ap.add_argument('--imf', default='UN', choices=['UN', 'KU', 'KB', 'BI', 'CH'], help='IMF (default UN = Salpeter)')
    ap.add_argument('--metals', default=DEFAULT_METALS,
                    help='[M/H] values to unpack, comma-separated, or "all" (default %s)' % DEFAULT_METALS)
    ap.add_argument('--dest', help='template folder (default spectralTemplates/EMILES_<ISO>_<IMF>)')
    ap.add_argument('--cache', default=os.path.expanduser('~/software/SSP'), help='where the archive is kept')
    ap.add_argument('--configdir', default='configFiles', help='where lsf_MILES is written')
    ap.add_argument('--source', help='use this local archive instead of downloading')
    ap.add_argument('--list', action='store_true', help='list the metallicities and ages in the archive and stop')
    ap.add_argument('--remote-list', nargs='?', const='', metavar='SUBFOLDER',
                    help='list the files of the IAC share (WebDAV) and stop')
    ap.add_argument('--archive', help='download this archive of the share (name as --remote-list shows it) '
                    'instead of EMILES_<ISO>_BASE_<IMF>_FITS.tar.gz')
    ap.add_argument('--alpha', help='unpack only these alpha tags, e.g. Ep0.00,Ep0.40 (default: all)')
    argv = sys.argv[1:]
    for k in range(len(argv) - 1):              # allow '--metals -0.66,...' (argparse would take it for an option)
        if argv[k] == '--metals' and argv[k + 1].startswith('-'):
            argv[k:k + 2] = ['--metals=' + argv[k + 1]]
            break
    a = ap.parse_args(argv)

    if a.remote_list is not None:
        remote_list(a.remote_list)
        return
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
        tf = tarfile.open(arch, 'r:gz')
        members = [m for m in tf.getmembers() if m.isfile() and m.name.endswith('.fits')]
    except (tarfile.TarError, OSError, EOFError) as e:
        sys.exit('%s does not open as a .tar.gz (%s): delete it and download again' % (arch, e))
    parsed = []
    for m in members:
        g = NAME.search(os.path.basename(m.name))
        if g:
            parsed.append((m, ('-' if g.group(1) == 'm' else '') + g.group(2), float(g.group(3))))
    if not parsed:
        sys.exit('no MILES-named .fits files found in %s' % arch)
    metals = sorted(set(float(p[1]) for p in parsed))
    ages = sorted(set(p[2] for p in parsed))
    print('%s: %d spectra, [M/H] = %s; %d ages, %.4g-%.4g Gyr'
          % (os.path.basename(arch), len(parsed), ', '.join('%+.2f' % z for z in metals), len(ages), ages[0], ages[-1]))
    if a.list:
        return

    if a.metals.strip().lower() == 'all':
        want = set(_metal_token(z) for z in metals)
    else:
        want = set(_metal_token(z) for z in a.metals.split(',') if z.strip())
        missing = [t for t in want if not any(t in os.path.basename(p[0].name) for p in parsed)]
        if missing:
            sys.exit('not in the archive: %s (available [M/H]: %s)'
                     % (', '.join(sorted(missing)), ', '.join('%+.2f' % z for z in metals)))

    if a.alpha:
        atoks = [t.strip() for t in a.alpha.split(',') if t.strip()]
        parsed = [p for p in parsed if any(t in os.path.basename(p[0].name) for t in atoks)]
        if not parsed:
            sys.exit('no file with alpha tag %s in the archive' % a.alpha)
    kinds = set('baseFe' if 'baseFe' in os.path.basename(p[0].name) else
                (ALPHA.search(os.path.basename(p[0].name)).group(0) if ALPHA.search(os.path.basename(p[0].name))
                 else '?') for p in parsed)
    if a.archive or a.source:
        stem = os.path.basename(arch).replace('.tar.gz', '').replace('.tgz', '').replace('_FITS', '')
        default_dest = os.path.join('spectralTemplates', stem)
    else:
        default_dest = os.path.join('spectralTemplates', 'EMILES_%s_%s' % (a.iso, a.imf))
    dest = a.dest or default_dest
    if os.path.isdir(dest):
        have = set('baseFe' if 'baseFe' in f else 'alpha' for f in os.listdir(dest) if f.endswith('.fits'))
        new = set('baseFe' if k == 'baseFe' else 'alpha' for k in kinds)
        if have and have != new:
            sys.exit('%s already holds %s templates; these are %s: GIST cannot mix baseFe and alpha-variable '
                     'files in one library -- give another --dest' % (dest, '/'.join(sorted(have)), '/'.join(sorted(new))))
    print('alpha tags unpacked: %s' % ', '.join(sorted(kinds)))
    if not os.path.isdir(dest):
        os.makedirs(dest)
    n = 0
    for m, z, t in parsed:
        base = os.path.basename(m.name)
        if not any(tok in base for tok in want):
            continue
        out = os.path.join(dest, base)                  # flat, original MILES name
        if not os.path.isfile(out):
            with tf.extractfile(m) as src, open(out, 'wb') as dst:
                shutil.copyfileobj(src, dst)
        n += 1
    tf.close()
    print('unpacked %d spectra (%d ages x %d metallicities) into %s/' % (n, len(ages), len(want), dest))

    lsf = os.path.join(a.configdir, 'lsf_MILES')
    if os.path.isdir(a.configdir) and not os.path.isfile(lsf):
        with open(lsf, 'w') as f:
            f.write(LSF_MILES)
        print('wrote %s (2.51 A FWHM)' % lsf)
    print('\nGIST settings for this library (SFH only, KIN keeps its own; gist_prep [SFH]):\n'
          '    TEMPLATES = miles    LIBRARY = %s/    LSF_TEMP = lsf_MILES    MATCH_RES = auto'
          % os.path.basename(os.path.normpath(dest)))


if __name__ == '__main__':
    main()
