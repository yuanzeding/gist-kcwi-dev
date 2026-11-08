#!/usr/bin/env python
"""
run_gist.py -- run GIST with sensible threading, logging and partial re-runs.

    python run_gist.py configFiles/MasterConfig_v2                 # all modules, background, log file
    python run_gist.py configFiles/MasterConfig_v2 --redo SFH      # re-run SFH only (+ LS)
    python run_gist.py configFiles/MasterConfig_v2 --redo KIN --fg # KIN and everything after it, foreground
    python run_gist.py configFiles/MasterConfig_v2 --redo ALL      # everything, incl. binning and spectra

Run it from the target directory (the one with configFiles/defaultDir).

What it does
 * BLAS threads = 1 per worker (OMP/MKL/OpenBLAS/numexpr). GIST already runs NCPU
   worker processes; with a threaded BLAS each of them would also start one thread
   per core, and NCPU x ncores threads fight over the same cores.
 * --redo MODULE: GIST skips a module whose output files exist when OW_OUTPUT is
   False. The outputs of MODULE and of every module after it (KIN -> GAS -> SFH -> LS;
   SFH with FIXED=True and GAS use the KIN results) are moved to
   <output>/superseded_<time>/, so only those modules run again. READ_DATA and
   SPATIAL_MASKING always run (they have no outputs to skip on).
 * runs gistPipeline under nohup and writes gist_<time>.log (unless --fg).
"""
from __future__ import print_function

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

ORDER = ['BINNING', 'SPECTRA', 'KIN', 'GAS', 'SFH', 'LS']
PATTERNS = {'BINNING': ['_table.fits', '_mask.fits'],
            'SPECTRA': ['_AllSpectra.fits', '_BinSpectra.fits', '_BinSpectra_linear.fits'],
            'KIN': ['_kin*.fits'], 'GAS': ['_gas*.fits'], 'SFH': ['_sfh*.fits'], 'LS': ['_ls*.fits']}
THREAD_VARS = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS']


def runs(masterconfig):
    """[(RUN_ID, OUTPUT, OW_OUTPUT)] of every row."""
    body = [l.rstrip('\n') for l in open(masterconfig) if not l.startswith('#')]
    if any(not l.strip() for l in body):
        sys.exit('%s has blank lines: GIST counts them as runs and crashes. Remove them.' % masterconfig)
    mods = [m.strip() for m in body[0].split('|') if m.strip()]
    keys = [k.split() for k in body[1].split('|')]
    out = []
    for line in body[2:]:
        vals = [v.split() for v in line.split('|')]
        g = dict(zip(keys[mods.index('GENERAL')], vals[mods.index('GENERAL')]))
        out.append((g['RUN_ID'], g['OUTPUT'], g['OW_OUTPUT'].lower() == 'true'))
    return out


def output_dir(defaultdir, output, run_id):
    base = '.'
    for line in open(defaultdir):
        if line.startswith('outputDir='):
            base = line.split('=', 1)[1].strip()
    return os.path.join(base, output, run_id)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('masterconfig')
    ap.add_argument('--default-dir', default='configFiles/defaultDir')
    ap.add_argument('--redo', choices=ORDER + ['ALL'], help='re-run this module and the ones after it')
    ap.add_argument('--fg', action='store_true', help='run in the foreground')
    ap.add_argument('--threads', type=int, default=1, help='BLAS threads per worker (default 1)')
    ap.add_argument('--dry-run', action='store_true', help='only show what would be moved / run')
    a = ap.parse_args()

    stamp = time.strftime('%Y%m%d_%H%M%S')
    for run_id, output, ow in runs(a.masterconfig):
        d = output_dir(a.default_dir, output, run_id)
        if ow and a.redo and a.redo != 'ALL':
            sys.exit('%s: OW_OUTPUT = True, so every module re-runs anyway. Set OW_OUTPUT False for --redo.' % run_id)
        if a.redo and os.path.isdir(d):
            todo = ORDER if a.redo == 'ALL' else ORDER[ORDER.index(a.redo):]
            files = sorted(set(f for m in todo for p in PATTERNS[m] for f in glob.glob(os.path.join(d, run_id + p))))
            dest = os.path.join(d, 'superseded_' + stamp)
            print('%s: moving %d files (%s) to %s' % (run_id, len(files), '+'.join(todo), dest))
            if not a.dry_run and files:
                os.makedirs(dest)
                for f in files:
                    shutil.move(f, dest)

    env = dict(os.environ)
    for v in THREAD_VARS:
        env[v] = str(a.threads)
    cmd = ['gistPipeline', '--config', a.masterconfig, '--default-dir', a.default_dir]
    if shutil.which('gistPipeline') is None:
        sys.exit('gistPipeline is not on the PATH: activate the GIST environment (e.g. conda activate gist-dev)')
    print('running:', ' '.join(cmd), '(BLAS threads per worker: %d)' % a.threads)
    if a.dry_run:
        return
    if a.fg:
        sys.exit(subprocess.call(cmd, env=env))
    log = 'gist_%s.log' % stamp
    with open(log, 'w') as f:
        p = subprocess.Popen(['nohup'] + cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                             preexec_fn=os.setpgrp)
    print('pid %d, log %s  (follow: tail -f %s; stop: kill -- -%d)' % (p.pid, log, log, p.pid))


if __name__ == '__main__':
    main()
