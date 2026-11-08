"""
Command-line entry points for the scripts in tools/ (registered in setup.py), so they
can be called by name from anywhere in the GIST environment, e.g.

    gist_prep prep_v4.in
    run_gist configFiles/MasterConfig_v4 --redo SFH
    gist_regul configFiles/MasterConfig_v4 --ncpu 5

Each command runs tools/<name>.py from the source tree as a script (as
"python tools/<name>.py ..." would): edits to the scripts take effect at once, without
reinstalling, and each script still finds the files that sit next to it (e.g.
tools/sky_lines_paranal.dat). This needs the editable install (pip install -e .), which
is how gistPipeline is installed here; after adding a NEW tool, add it to TOOLS below and
to setup.py, and run pip install -e . once more.
"""
import os
import runpy
import sys

TOOLS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tools')

TOOLS = ['gist_prep', 'run_gist', 'gist_regul', 'gist_snrcheck', 'gist_view', 'gist_replot',
         'gist_fitresid', 'cube_windows', 'kcwi_skylevel', 'kcwi_dichroic_step', 'kcwi_sync_bifrost',
         'get_ssp', 'kcwi_covar']


def _run(name):
    path = os.path.join(TOOLS_DIR, name + '.py')
    if not os.path.isfile(path):
        sys.exit('%s: %s not found (gistPipeline is expected as an editable install of the source tree '
                 'that holds tools/)' % (name, path))
    sys.argv[0] = path                      # usage / help messages show the script
    sys.path.insert(0, TOOLS_DIR)           # as when the script is run directly
    runpy.run_path(path, run_name='__main__')


def gist_prep():          _run('gist_prep')
def run_gist():           _run('run_gist')
def gist_regul():         _run('gist_regul')
def gist_snrcheck():      _run('gist_snrcheck')
def gist_view():          _run('gist_view')
def gist_replot():        _run('gist_replot')
def gist_fitresid():      _run('gist_fitresid')
def cube_windows():       _run('cube_windows')
def kcwi_skylevel():      _run('kcwi_skylevel')
def kcwi_dichroic_step(): _run('kcwi_dichroic_step')
def kcwi_sync_bifrost():  _run('kcwi_sync_bifrost')
def get_ssp():            _run('get_ssp')
def kcwi_covar():         _run('kcwi_covar')
