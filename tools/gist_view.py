#!/usr/bin/env python
"""
gist_view.py -- a Qt-free viewer for GIST results (maps + spectra), drawn with
matplotlib's Tk backend, which works over plain X11 forwarding (ssh -Y, XQuartz).

    python gist_view.py results/<RUN>                     # interactive window
    python gist_view.py results/<RUN> --quantity SIGMA    # start on a quantity
    python gist_view.py results/<RUN> --bin 12 --save b12.png   # no display: write a PNG

Left: a map of any quantity in the run (stellar kinematics, SFH, gas, line
strengths, and the table's FLUX / SNR / SNRBIN / NSPAX). Masked spaxels are left
blank (the run's _mask.fits), so a masked spaxel whose BIN_ID is 0 is not
painted with bin 0.
Right: the spectrum of the bin under the mouse click with every best fit of the
run (KIN, SFH, GAS) and its residuals, and the SFH weights against age.

Mouse and keys (in the window)
    click on the map       show that bin
    n / p                  next / previous quantity  (also the list on the left)
    left / right arrows    previous / next bin
    c                      colour limits: percentile 2-98 <-> 16-84
    s                      save the current view as <RUN>_view_bin<N>.png
    q                      quit (matplotlib default)

Backend: TkAgg, then GTK3Agg, WXAgg (whichever imports); --backend to choose.
Needs only numpy, astropy and matplotlib (plotbin's 'sauron' colour map if present).
"""
import argparse
import glob
import os
import sys

import numpy as np
from astropy.io import fits


def pick_backend(name, save):
    import matplotlib
    if save:
        matplotlib.use('Agg')
        return 'Agg'
    for b in ([name] if name else ['TkAgg', 'GTK3Agg', 'WXAgg']):
        try:
            matplotlib.use(b, force=True)
            import matplotlib.pyplot as plt     # noqa: F401  (imports the backend)
            plt.figure().clf()
            plt.close('all')
            return b
        except Exception as e:                  # pragma: no cover
            last = e
            continue
    sys.exit('no interactive matplotlib backend works here (%s). Is DISPLAY set (ssh -Y)? '
             'Use --save for a PNG instead.' % last)


# ---------------------------------------------------------------------------
# reading a run
# ---------------------------------------------------------------------------
class Run(object):
    def __init__(self, rundir):
        self.dir = rundir.rstrip('/')
        self.name = os.path.basename(self.dir)
        p = os.path.join(self.dir, self.name)
        self.prefix = p
        if not os.path.isfile(p + '_table.fits'):
            sys.exit('%s: no %s_table.fits -- is this a GIST run directory?' % (self.dir, self.name))
        th = fits.open(p + '_table.fits')
        t = th[1].data
        self.pix = float(th[0].header.get('PIXSIZE', np.nan))
        self.X = np.asarray(t.X, float)
        self.Y = np.asarray(t.Y, float)
        self.BIN = np.asarray(t.BIN_ID, int)
        self.table = t
        use = self.BIN >= 0
        if os.path.isfile(p + '_mask.fits'):
            m = np.asarray(fits.open(p + '_mask.fits')[1].data.MASK)
            if m.size == use.size:
                use &= (m == 0)
        self.use = use
        self.nbins = int(self.BIN[use].max()) + 1 if use.any() else 0
        if not np.isfinite(self.pix):
            d = np.diff(np.unique(np.round(self.X, 4)))
            self.pix = float(np.median(d[d > 0]))

        # quantities: (label, kind, values) ; kind 'bin' (per bin) or 'spax' (per spaxel)
        self.quant = []
        for suffix, tag in [('_kin.fits', 'KIN'), ('_sfh.fits', 'SFH'), ('_sfh-gas.fits', 'SFHgas'),
                            ('_gas_BIN.fits', 'GAS'), ('_ls_OrigRes.fits', 'LS'), ('_ls_AdapRes.fits', 'LSa')]:
            fn = p + suffix
            if not os.path.isfile(fn):
                continue
            d = fits.open(fn)[1].data
            for c in d.columns.names:
                v = np.asarray(d[c])
                if v.ndim == 1 and v.size == self.nbins and np.issubdtype(v.dtype, np.number):
                    lab = c if tag in ('KIN', 'SFH') else '%s %s' % (tag, c)
                    if tag == 'SFH' and c in ('V', 'SIGMA', 'H3', 'H4', 'H5', 'H6') or c.startswith('FORM_ERR') and tag == 'SFH':
                        lab = 'SFH ' + c
                    self.quant.append((lab, 'bin', v.astype(float)))
        for c in ('FLUX', 'SNR', 'SNRBIN', 'NSPAX'):
            if c in t.columns.names:
                self.quant.append((c, 'spax', np.asarray(t[c], float)))

        # spectra and fits
        self.spec = None
        if os.path.isfile(p + '_BinSpectra.fits'):
            h = fits.open(p + '_BinSpectra.fits')
            self.spec = np.asarray(h[1].data.SPEC)
            self.espec = np.asarray(h[1].data.ESPEC)
            self.lam = np.exp(np.asarray(h['LOGLAM'].data.LOGLAM))
        self.fits_ = []
        for fn in sorted(glob.glob(p + '_*bestfit*.fits')):
            h = fits.open(fn)
            try:
                bf = np.asarray(h['BESTFIT'].data.BESTFIT)
                lam = np.exp(np.asarray(h['LOGLAM'].data.LOGLAM))
            except Exception:
                continue
            if bf.ndim != 2 or bf.shape[0] != self.nbins:
                continue
            good = np.asarray(h['GOODPIX'].data.GOODPIX) if 'GOODPIX' in h else None
            tag = os.path.basename(fn)[len(self.name) + 1:].replace('.fits', '')
            self.fits_.append((tag, lam, bf, good))
        self.weights = None
        if os.path.isfile(p + '_sfh-weights.fits'):
            h = fits.open(p + '_sfh-weights.fits')
            w = np.asarray(h['WEIGHTS'].data.WEIGHTS)
            if w.ndim == 2 and w.shape[0] == self.nbins:
                self.weights = w
                self.logage = np.asarray(h['GRID'].data.LOGAGE)
                self.metal = np.asarray(h['GRID'].data.METAL)

        # map grid (east to the left, as in the GIST maps: horizontal axis = -X)
        u = -self.X
        self.u0, self.y0 = u.min(), self.Y.min()
        self.j = np.rint((u - self.u0) / self.pix).astype(int)
        self.i = np.rint((self.Y - self.y0) / self.pix).astype(int)
        self.shape = (self.i.max() + 1, self.j.max() + 1)
        self.extent = [self.u0 - self.pix / 2, self.u0 + (self.shape[1] - 0.5) * self.pix,
                       self.y0 - self.pix / 2, self.y0 + (self.shape[0] - 0.5) * self.pix]
        self.grid_id = np.full(self.shape, -1, int)          # spaxel index on the grid
        self.grid_id[self.i, self.j] = np.arange(self.X.size)

    def image(self, k):
        lab, kind, v = self.quant[k]
        img = np.full(self.shape, np.nan)
        s = np.where(self.use)[0]
        val = v[self.BIN[s]] if kind == 'bin' else v[s]
        if lab == 'FLUX':
            with np.errstate(all='ignore'):
                val = np.log10(np.where(val > 0, val, np.nan))
        img[self.i[s], self.j[s]] = val
        return img

    def bin_at(self, xd, yd):
        j = int(np.rint((xd - self.u0) / self.pix))
        i = int(np.rint((yd - self.y0) / self.pix))
        if 0 <= i < self.shape[0] and 0 <= j < self.shape[1]:
            s = self.grid_id[i, j]
            if s >= 0 and self.use[s]:
                return int(self.BIN[s])
        return None


# ---------------------------------------------------------------------------
# the viewer
# ---------------------------------------------------------------------------
class Viewer(object):
    def __init__(self, run, quantity=None, b=None):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import RadioButtons
        self.plt = plt
        self.r = run
        self.k = 0
        if quantity:
            names = [q[0] for q in run.quant]
            if quantity not in names:
                sys.exit('quantity %s not in this run; have: %s' % (quantity, ', '.join(names)))
            self.k = names.index(quantity)
        flux = run.table.FLUX if 'FLUX' in run.table.columns.names else None
        if b is None:
            b = int(run.BIN[run.use][np.nanargmax(np.asarray(flux)[run.use])]) if flux is not None else 0
        self.b = b
        self.pct = (2, 98)
        try:
            from plotbin.sauron_colormap import register_sauron_colormap
            register_sauron_colormap()
            self.cmap = 'sauron'
        except Exception:
            self.cmap = 'RdYlBu_r'

        self.fig = plt.figure(figsize=(14, 7.6))
        try:
            self.fig.canvas.manager.set_window_title('gist_view: ' + run.name)
        except Exception:
            pass
        self.axr = self.fig.add_axes([0.005, 0.08, 0.10, 0.84])
        self.axm = self.fig.add_axes([0.175, 0.12, 0.31, 0.76])
        self.axc = self.fig.add_axes([0.175, 0.05, 0.31, 0.018])
        has_w = run.weights is not None
        self.axs = self.fig.add_axes([0.53, 0.50 if has_w else 0.36, 0.45, 0.42 if has_w else 0.56])
        self.axd = self.fig.add_axes([0.53, 0.37 if has_w else 0.08, 0.45, 0.12 if has_w else 0.26], sharex=self.axs)
        self.axw = self.fig.add_axes([0.53, 0.07, 0.45, 0.22]) if has_w else None

        labels = [q[0] for q in run.quant]
        self.radio_labels = labels[:28]
        for t in self.axr.spines.values():
            t.set_visible(False)
        self.radio = RadioButtons(self.axr, self.radio_labels, active=min(self.k, len(self.radio_labels) - 1))
        for t in self.radio.labels:
            t.set_fontsize(8)
        self.radio.on_clicked(self._on_radio)
        self.cbar = None
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.draw_map()
        self.draw_bin()

    # --- maps
    def draw_map(self):
        r, ax = self.r, self.axm
        ax.cla()
        self.axc.cla()
        img = r.image(self.k)
        lab, kind, v = r.quant[self.k]
        fin = img[np.isfinite(img)]
        if fin.size:
            lo, hi = np.percentile(fin, self.pct)
            if lab in ('V', 'SFH V'):                     # velocity: symmetric about the median
                med = np.median(fin)
                d = max(abs(hi - med), abs(med - lo))
                lo, hi = med - d, med + d
        else:
            lo, hi = 0, 1
        im = ax.imshow(img, origin='lower', extent=r.extent, cmap=self.cmap, vmin=lo, vmax=hi,
                       interpolation='nearest')
        ax.set_xlabel(r'$\Delta\alpha$ [arcsec]')
        ax.set_ylabel(r'$\Delta\delta$ [arcsec]')
        ax.set_title('%s   %s' % (r.name, 'log10 FLUX' if lab == 'FLUX' else lab), fontsize=10)
        self.cbar = self.fig.colorbar(im, cax=self.axc, orientation='horizontal')
        self.axc.tick_params(labelsize=8)
        self._mark()
        self.fig.canvas.draw_idle()

    def _mark(self):
        r, ax = self.r, self.axm
        for a in getattr(self, '_marks', []):
            try:
                a.remove()
            except Exception:
                pass
        s = np.where(r.use & (r.BIN == self.b))[0]
        if s.size:
            self._marks = ax.plot(-r.X[s], r.Y[s], 's', ms=2.5, mfc='none', mec='k', mew=0.6, alpha=0.8)
        else:
            self._marks = []

    # --- spectra
    def draw_bin(self):
        r, b = self.r, self.b
        ax, axd = self.axs, self.axd
        ax.cla()
        axd.cla()
        info = []
        for lab in ('V', 'SIGMA', 'H3', 'H4', 'AGE', 'METAL'):
            q = [x for x in r.quant if x[0] == lab]
            if q:
                info.append('%s %.3g' % (lab, q[0][2][b]))
        s = np.where(r.use & (r.BIN == b))[0]
        snr = r.table.SNRBIN[s[0]] if s.size and 'SNRBIN' in r.table.columns.names else np.nan
        ax.set_title('bin %d: %d spaxels, S/N %.1f   %s' % (b, s.size, snr, '  '.join(info)), fontsize=9)
        colors = ['C3', 'C0', 'C2', 'C1', 'C4']
        if r.spec is not None:
            f, e = r.spec[b], r.espec[b]
            ok = np.isfinite(e) & (e < 100 * np.nanmedian(e))
            ax.fill_between(r.lam, np.where(ok, f - e, np.nan), np.where(ok, f + e, np.nan), color='0.85', lw=0, step='mid')
            ax.plot(r.lam, f, color='k', lw=0.6, drawstyle='steps-mid', label='data')
            fin = f[np.isfinite(f)]
            if fin.size:
                lo, hi = np.percentile(fin, [0.5, 99.5])
                ax.set_ylim(lo - 0.1 * (hi - lo), hi + 0.15 * (hi - lo))
        for (tag, lam, bf, good), c in zip(r.fits_, colors):
            ax.plot(lam, bf[b], color=c, lw=0.8, label=tag)
            if r.spec is not None:
                i0 = int(np.argmin(np.abs(r.lam - lam[0])))
                n = min(lam.size, r.lam.size - i0)
                res = r.spec[b, i0:i0 + n] - bf[b, :n]
                if good is not None:
                    m = np.zeros(n, bool)
                    m[good[good < n]] = True
                    res = np.where(m, res, np.nan)
                axd.plot(lam[:n], res, color=c, lw=0.5, drawstyle='steps-mid')
        if r.spec is not None:
            e = r.espec[b]
            ok = np.isfinite(e) & (e < 100 * np.nanmedian(e))      # no-data pixels carry a huge error
            axd.fill_between(r.lam, -np.where(ok, e, np.nan), np.where(ok, e, np.nan), color='0.85', lw=0, step='mid')
            lim = 4 * np.nanpercentile(e[ok], 90) if ok.any() else 1
            for ln in axd.get_lines():
                yv = np.asarray(ln.get_ydata(), float)
                yv = yv[np.isfinite(yv)]
                if yv.size:
                    lim = max(lim, 1.2 * np.percentile(np.abs(yv), 99))
            axd.set_ylim(-lim, lim)
        axd.axhline(0, color='k', lw=0.5)
        axd.set_ylabel('data - fit', fontsize=8)
        axd.set_xlabel(u'rest-frame wavelength [Å]')
        ax.legend(fontsize=7, loc='upper right', ncol=3)
        ax.tick_params(labelbottom=False)
        if self.axw is not None:
            aw = self.axw
            aw.cla()
            w = r.weights[b]
            ages = np.unique(r.logage)
            tot = np.array([w[r.logage == a].sum() for a in ages])
            aw.bar(ages, tot / max(tot.sum(), 1e-30), width=np.min(np.diff(ages)) * 0.8 if ages.size > 1 else 0.1,
                   color='C1')
            aw.set_xlabel('log age [Gyr]')
            aw.set_ylabel('weight fraction', fontsize=8)
        self._mark()
        self.fig.canvas.draw_idle()

    # --- events
    def _on_radio(self, label):
        self.k = [q[0] for q in self.r.quant].index(label)
        self.draw_map()

    def _on_click(self, ev):
        if ev.inaxes is not self.axm or ev.xdata is None:
            return
        b = self.r.bin_at(ev.xdata, ev.ydata)
        if b is not None:
            self.b = b
            self.draw_bin()

    def _on_key(self, ev):
        n = len(self.r.quant)
        if ev.key in ('n', 'p'):
            self.k = (self.k + (1 if ev.key == 'n' else -1)) % n
            if self.r.quant[self.k][0] in self.radio_labels:
                self.radio.set_active(self.radio_labels.index(self.r.quant[self.k][0]))
            else:
                self.draw_map()
        elif ev.key in ('right', 'left'):
            self.b = (self.b + (1 if ev.key == 'right' else -1)) % self.r.nbins
            self.draw_bin()
        elif ev.key == 'c':
            self.pct = (16, 84) if self.pct == (2, 98) else (2, 98)
            self.draw_map()
        elif ev.key == 's':
            fn = '%s_view_bin%d.png' % (self.r.name, self.b)
            self.fig.savefig(fn, dpi=120)
            print('wrote', fn)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('rundir', help='the run directory, e.g. results/2MASXJ1735+2047bin35_v3')
    ap.add_argument('--quantity', help='quantity to start on (e.g. V, SIGMA, AGE, FLUX)')
    ap.add_argument('--bin', type=int, help='bin to start on (default: the brightest)')
    ap.add_argument('--backend', help='matplotlib backend (default: TkAgg, GTK3Agg or WXAgg)')
    ap.add_argument('--save', help='no window: write this PNG and exit')
    ap.add_argument('--list', action='store_true', help='list the quantities in the run and exit')
    a = ap.parse_args()
    run = Run(a.rundir)
    if a.list:
        print('%d bins; quantities: %s' % (run.nbins, ', '.join(q[0] for q in run.quant)))
        print('best fits: %s' % (', '.join(f[0] for f in run.fits_) or 'none'))
        return
    be = pick_backend(a.backend, a.save)
    v = Viewer(run, a.quantity, a.bin)
    if a.save:
        v.fig.savefig(a.save, dpi=120)
        print('wrote', a.save)
        return
    print('backend %s: click a spaxel; n/p quantity, arrows bin, c colour range, s save' % be)
    v.plt.show()


if __name__ == '__main__':
    main()
