#!/usr/bin/env python
"""
kcwi_sync_bifrost -- copy newly reduced KCWI stacks to the matching gist-KCWI/inputData
folders on bifrost, with rsync (only new or changed files are sent).

    python kcwi_sync_bifrost.py LEDA66688-final              # one source
    python kcwi_sync_bifrost.py LEDA66688-final UGC2369N -n  # dry run: show what would be sent
    python kcwi_sync_bifrost.py --sources sources.txt        # the kcwi_skylevel batch file works too
    python kcwi_sync_bifrost.py --all                        # every stack in the folder

Run it in the stack folder (default: the current directory; --stackdir to change).

What is sent for a source NAME: the stack cubes of every version of it,
    NAME_[ievm]cubes[_wcs].fits, NAME-red_..., NAME_skl_..., NAME_skl-red_..., NAME_drp_...
(i.e. files that start with NAME followed by '_' or '-red' and end in cubes.fits or
cubes_wcs.fits; *.bak* files are never sent). --lists also sends NAME*.list / .par.
Symbolic links (apply --stack links the _vcubes/_mcubes) are sent as the files they
point to.

--covar sends the kcwi_covar curves (covar_mock/curves/covar_*.txt) to <ROOT>/covar/,
where gist_prep finds them by cube name (SPATIAL_BINNING COV_FILE = auto):
    python kcwi_sync_bifrost.py --covar                 # curves only
    python kcwi_sync_bifrost.py --all --covar           # stacks and curves

Where it goes: <ROOT>/<FOLDER>/gist-KCWI/inputData on <HOST>. FOLDER is found, in order:
  1. --map FILE   lines 'NAME FOLDER' (e.g. 'LEDA66688-final LEDA66688')
  2. the bifrost folder that already holds NAME's cubes (one ssh call lists them all)
  3. a bifrost folder named like NAME: as is, without '-final', TMASX/TMASS -> 2MASX/2MASS,
     without a trailing N/S (UGC2369S -> UGC2369) -- only if that folder exists
A source with no folder is skipped (use --map, or --create to make <ROOT>/NAME/gist-KCWI/inputData).

One ssh connection is shared by all calls (ControlMaster), so you are asked for the
password / key once.
"""
import argparse
import glob
import os
import re
import shlex
import subprocess
import sys

HOST = 'yuanze@bifrost.caltech.edu'
ROOT = '/disk/bifrost/yuanze/multiAGN'
SUB = 'gist-KCWI/inputData'
CUBE_RE = r'(cubes|cubes_wcs)\.fits$'


def find_rsync():
    """Prefer a GNU rsync (Homebrew); macOS's own openrsync lacks some options."""
    for r in ('/opt/homebrew/bin/rsync', '/usr/local/bin/rsync', 'rsync'):
        try:
            v = subprocess.run([r, '--version'], capture_output=True, text=True).stdout + \
                subprocess.run([r, '--version'], capture_output=True, text=True).stderr
        except OSError:
            continue
        if 'openrsync' in v:
            return r, ['-avL']
        return r, ['-avhL', '--progress']
    sys.exit('rsync not found')


def ssh_opts():
    cp = os.path.expanduser('~/.ssh/cm-kcwisync-%r@%h:%p')
    return ['-o', 'ControlMaster=auto', '-o', 'ControlPath=' + cp, '-o', 'ControlPersist=10m']


def remote(host, cmd):
    """Run a shell command on the host (or locally for --host local); return stdout."""
    if host == 'local':
        return subprocess.run(['bash', '-c', cmd], capture_output=True, text=True).stdout
    # wrap in bash: the login shell on the host may be tcsh
    r = subprocess.run(['ssh'] + ssh_opts() + [host, 'bash -c ' + shlex.quote(cmd)], capture_output=True, text=True)
    if r.returncode != 0 and not r.stdout:
        sys.exit('ssh %s failed: %s' % (host, r.stderr.strip()))
    return r.stdout


def source_files(name, stackdir, lists=False):
    out = []
    for f in sorted(os.listdir(stackdir)):
        if not (f.startswith(name + '_') or f.startswith(name + '-red') or f.startswith(name + '.')):
            continue
        if '.bak' in f:
            continue
        rest = f[len(name):]
        # do not pick up another source whose name starts with this one (e.g. NAME_x is ok, NAMEx is not)
        if re.search(CUBE_RE, f) or (lists and re.search(r'\.(list|par)$', f)):
            if rest.startswith('-red') or rest.startswith('_') or rest.startswith('.'):
                out.append(f)
    return out


def all_sources(stackdir):
    names = set()
    for f in os.listdir(stackdir):
        m = re.match(r'^(.+?)(-red)?_icubes(_wcs)?\.fits$', f)
        if m and '.bak' not in f:
            n = m.group(1)
            n = re.sub(r'(_skl|_drp|_sklq)$', '', n)
            names.add(n)
    return sorted(names)


def read_map(fn):
    m = {}
    if fn:
        for line in open(fn):
            s = line.split('#')[0].split()
            if len(s) >= 2:
                m[s[0]] = s[1]
    return m


def candidates(name):
    c = [name, re.sub(r'-final$', '', name)]
    for n in list(c):
        c.append(re.sub(r'^TMASX', '2MASX', n)); c.append(re.sub(r'^TMASS', '2MASS', n))
    for n in list(c):
        c.append(re.sub(r'(\d)[NS]$', r'\1', n))
    seen = []
    for n in c:
        if n not in seen:
            seen.append(n)
    return seen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('names', nargs='*', help='stack base names (e.g. LEDA66688-final)')
    ap.add_argument('--sources', help='file with one source per line (first word used), e.g. the kcwi_skylevel batch file')
    ap.add_argument('--all', action='store_true', help='every stack in --stackdir')
    ap.add_argument('--stackdir', default='.', help='folder with the stacks (default: current directory)')
    ap.add_argument('--host', default=HOST, help='ssh host (default %s; "local" = this machine, for testing)' % HOST)
    ap.add_argument('--root', default=ROOT, help='folder holding the source folders on the host (default %s)' % ROOT)
    ap.add_argument('--map', help="file with lines 'NAME FOLDER' (overrides the automatic choice)")
    ap.add_argument('--lists', action='store_true', help='also send NAME*.list / .par')
    ap.add_argument('--create', action='store_true', help='make <ROOT>/NAME/gist-KCWI/inputData for sources with no folder')
    ap.add_argument('-n', '--dry-run', action='store_true', help='show what would be sent, send nothing')
    ap.add_argument('--covar', nargs='?', const='covar_mock/curves', metavar='DIR',
                    help='also send the kcwi_covar curves (covar_*.txt in DIR, default <stackdir>/covar_mock/curves) '
                         'to <ROOT>/covar/ on the host, the folder gist_prep reads them from (COV_FILE = auto)')
    a = ap.parse_args()
    global RSYNC, RSYNC_FLAGS
    RSYNC, RSYNC_FLAGS = find_rsync()

    if a.covar is not None:
        cdir = a.covar if os.path.isabs(a.covar) else os.path.join(a.stackdir, a.covar)
        curves = sorted(glob.glob(os.path.join(cdir, 'covar_*.txt')))
        if not curves:
            sys.exit('no covar_*.txt in %s' % cdir)
        dest = '%s/covar/' % a.root
        print('%d covariance curves from %s -> %s:%s' % (len(curves), cdir, a.host, dest))
        if not a.dry_run:
            remote(a.host, 'mkdir -p %s' % shlex.quote(dest))
        cmd = [RSYNC] + RSYNC_FLAGS + (['-n'] if a.dry_run else [])
        if a.host != 'local':
            cmd += ['-e', 'ssh ' + ' '.join(ssh_opts())]
        cmd += curves + [dest if a.host == 'local' else '%s:%s' % (a.host, dest)]
        if subprocess.run(cmd).returncode != 0:
            sys.exit('rsync of the covariance curves failed')
        if not (a.names or a.sources or a.all):
            print('\ndone%s' % (' (dry run: nothing sent)' if a.dry_run else ''))
            return

    names = list(a.names)
    if a.sources:
        for line in open(a.sources):
            s = line.split('#')[0].split()
            if s:
                n = s[0][:-len('.list')] if s[0].endswith('.list') else s[0]
                names.append(os.path.basename(n))
    if a.all:
        names += all_sources(a.stackdir)
    names = list(dict.fromkeys(names))
    if not names:
        ap.error('give source names, --sources FILE or --all')

    # one listing of the host: which inputData folders exist, and which cubes they hold
    q = shlex.quote(a.root)
    listing = remote(a.host, 'cd %s 2>/dev/null && for d in */%s; do [ -d "$d" ] && '
                             'for f in "$d"/*cubes*.fits; do echo "$f"; done; echo "$d/"; done' % (q, SUB))
    folders, holds = set(), {}
    for line in listing.splitlines():
        line = line.strip()
        if not line:
            continue
        folder = line.split('/')[0]
        folders.add(folder)
        holds.setdefault(os.path.basename(line), set()).add(folder)
    fmap = read_map(a.map)

    plan = []
    for name in names:
        files = source_files(name, a.stackdir, a.lists)
        if not files:
            print('%-24s no stack files in %s; skipped' % (name, a.stackdir))
            continue
        if name in fmap:
            folder, how = fmap[name], 'map file'
        else:
            have = [holds.get(f, set()) for f in files]
            have = set.union(*have) if have else set()
            if len(have) == 1:
                folder, how = have.pop(), 'already holds its cubes'
            elif len(have) > 1:
                print('%-24s its cubes are in several folders (%s); use --map' % (name, ', '.join(sorted(have))))
                continue
            else:
                folder = next((c for c in candidates(name) if c in folders), None)
                how = 'name match'
                if folder is None:
                    if a.create:
                        folder, how = name, 'new folder (--create)'
                    else:
                        print('%-24s no gist-KCWI folder on the host (tried %s); use --map or --create'
                              % (name, ', '.join(candidates(name))))
                        continue
        plan.append((name, folder, how, files))

    if not plan:
        sys.exit('nothing to send')
    print('\n%-24s -> %s:%s/<folder>/%s' % ('source', a.host, a.root, SUB))
    for name, folder, how, files in plan:
        print('%-24s -> %-22s (%s; %d files)' % (name, folder, how, len(files)))

    failed = []
    for name, folder, how, files in plan:
        dest = '%s/%s/%s/' % (a.root, folder, SUB)
        if how.startswith('new folder') and not a.dry_run:
            remote(a.host, 'mkdir -p %s' % shlex.quote(dest))
        # -L: send what symlinks point to; -a keeps mtimes, so later runs send only new or changed files
        cmd = [RSYNC] + RSYNC_FLAGS
        if a.dry_run:
            cmd.append('-n')
        if a.host != 'local':
            cmd += ['-e', 'ssh ' + ' '.join(ssh_opts())]
        target = dest if a.host == 'local' else '%s:%s' % (a.host, dest)
        cmd += [os.path.join(a.stackdir, f) for f in files] + [target]
        print('\n== %s -> %s' % (name, target))
        if subprocess.run(cmd).returncode != 0:
            failed.append(name)
    if failed:
        sys.exit('rsync failed for: ' + ', '.join(failed))
    print('\ndone%s' % (' (dry run: nothing sent)' if a.dry_run else ''))


if __name__ == '__main__':
    main()
