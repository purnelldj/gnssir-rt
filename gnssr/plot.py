"""
written by David Purnell
https://github.com/purnelldj
"""
import numpy as np
from gnssr.helper import gps2datenum, cubspl_nans
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
from pathlib import Path


def plotrhspline(rh_arr, plotfig=True, savefig=False, plotdt=15*60, plotknots=False, plotrh=False,
                 plotspec=True, figoutstr='rhsplineout.png', **kwargs):

    # then get arcs and plot
    arcs = rh_arr[:, 1]
    print('mean height of arcs is ' + str(np.nanmean(arcs)))
    arcs_dn = gps2datenum(np.array(rh_arr[:, 0], dtype=float))
    print('avg of ' + str(round(len(arcs) / ((rh_arr[-1, 0] - rh_arr[0, 0]) / 86400))) + ' arcs per day')
    print('with ' + str(len(np.unique(arcs))) + ' total arcs')
    if plotfig:
        plt.rcParams.update({'font.family': 'Times New Roman'})
        if 'figsize' in kwargs:
            figsize = kwargs.get('figsize')
            _, ax = plt.subplots(figsize=(figsize[0], figsize[1]))
        else:
            _, ax = plt.subplots(figsize=(9, 4))
        if plotrh:
            plot_primary_secondary_peaks = False
            if plot_primary_secondary_peaks:
                parc_1, = plt.plot_date(arcs_dn[rh_arr[:, 11] == 1], arcs[rh_arr[:, 11] == 1], '.', markersize=2)
                parc_1.set_label('primary peaks')
                parc_2, = plt.plot_date(arcs_dn[rh_arr[:, 11] == 2], arcs[rh_arr[:, 11] == 2], '.', markersize=2)
                parc_2.set_label('secondary peaks')
            else:
                parc, = plt.plot_date(arcs_dn, arcs, '.', markersize=2, color='gray')
                parc.set_label('Arcs')

    # then get spline(s)
    if 'kval_spectral' and 'knots' in kwargs:
        kval_spectral = kwargs.get('kval_spectral')
        knots = kwargs.get('knots')
        knots_dn = gps2datenum(np.array(knots, dtype=float))
        tt = np.linspace(knots[0], knots[-1], int((knots[-1] - knots[0]) / plotdt))
        spectral = cubspl_nans(tt, knots, kval_spectral)
        tt_dn = gps2datenum(tt)
        if plotfig and plotspec:
            #pspec, = plt.plot_date(dn_spectral_rmse, spectral_rmse, '.')
            pspec, = plt.plot_date(tt_dn, spectral, '-', color='hotpink')
            pspec.set_label('GNSS-R spline fit')
            if plotknots:
                kval_spectral_plot = kval_spectral
                pknot, = plt.plot_date(knots_dn, kval_spectral_plot, '.', markersize=10)
                pknot.set_label('knots')

    if plotfig:
        dformat = DateFormatter('%d/%m')
        plt.ylabel('Water level (m)')
        ax.xaxis.set_major_formatter(dformat)
        if 'xlims' in kwargs:
            xlims = kwargs.get('xlims')
            ax.set_xlim(xlims[0], xlims[1])
        else:
            ax.set_xlim(np.min(arcs_dn), np.min(arcs_dn[-1]))
        if 'ylims' in kwargs:
            ylims = kwargs.get('ylims')
            ax.set_ylim(ylims[0], ylims[1])
        ax.legend()
        if not savefig or 'outdir' not in kwargs:
            plt.show()
        else:
            outdir = kwargs.get('outdir')
            Path(outdir).mkdir(parents=True, exist_ok=True)
            plt.savefig(outdir + '/' + figoutstr, format='png', dpi=300)
            plt.close()


