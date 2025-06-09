from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter

from gnssir.helper import cubspl_nans, gps2datenum


def plotrhspline(
    rh_arr: np.ndarray,
    plotfig: bool = True,
    savefig: bool = False,
    plotdt: float = 15 * 60,
    plotknots: bool = False,
    plotrh: bool = True,
    plotspec: bool = True,
    figoutstr: str = "rhsplineout.png",
    **kwargs: Any,
) -> None:
    """
    Plot reflector heights and optional spline fit for GNSS-IR analysis.

    This function creates a visualization of GNSS-IR reflector heights and optionally
    fits a spline curve to the data. It supports both interactive display and saving
    to file with customizable parameters.

    Parameters
    ----------
    rh_arr : np.ndarray
        Array containing GNSS-IR data with columns:
        - Column 0: GPS time
        - Column 1: Reflector heights
        - Column 11: Peak type (1 for primary, 2 for secondary)
    plotfig : bool, optional
        Whether to create the plot, by default True
    savefig : bool, optional
        Whether to save the plot to file, by default False
    plotdt : float, optional
        Time step for spline interpolation in seconds, by default 15 minutes
    plotknots : bool, optional
        Whether to plot spline knots, by default False
    plotrh : bool, optional
        Whether to plot reflector heights, by default True
    plotspec : bool, optional
        Whether to plot spline fit, by default True
    figoutstr : str, optional
        Output filename for saved plot, by default "rhsplineout.png"
    **kwargs : Any
        Additional keyword arguments:
        - figsize: Tuple[float, float] for figure dimensions
        - xlims: Tuple[float, float] for x-axis limits
        - ylims: Tuple[float, float] for y-axis limits
        - outdir: str for output directory path
        - kval_spectral: np.ndarray for spline coefficients
        - knots: np.ndarray for spline knot positions

    Notes
    -----
    The function creates a publication-quality plot with Times New Roman font and
    customizable formatting. It handles both primary and secondary peaks, and can
    optionally fit and display a spline curve through the data.
    """
    # Extract arcs and convert times to datetime numbers
    arcs = rh_arr[:, 1]
    print("mean height of arcs is " + str(np.nanmean(arcs)))
    arcs_dn = gps2datenum(np.array(rh_arr[:, 0], dtype=float))
    print("avg of " + str(round(len(arcs) / ((rh_arr[-1, 0] - rh_arr[0, 0]) / 86400))) + " arcs per day")
    print("with " + str(len(np.unique(arcs))) + " total arcs")

    # Create plot if requested
    if plotfig:
        plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10})
        if "figsize" in kwargs:
            figsize = kwargs.get("figsize")
            _, ax = plt.subplots(figsize=(figsize[0], figsize[1]))
        else:
            _, ax = plt.subplots(figsize=(5, 2.5))

        # Plot reflector heights if requested
        if plotrh:
            plot_primary_secondary_peaks = False
            if plot_primary_secondary_peaks:
                # Plot primary peaks
                (parc_1,) = plt.plot_date(
                    arcs_dn[rh_arr[:, 11] == 1],
                    arcs[rh_arr[:, 11] == 1],
                    ".",
                    markersize=2,
                )
                parc_1.set_label("primary peaks")
                # Plot secondary peaks
                (parc_2,) = plt.plot_date(
                    arcs_dn[rh_arr[:, 11] == 2],
                    arcs[rh_arr[:, 11] == 2],
                    ".",
                    markersize=2,
                )
                parc_2.set_label("secondary peaks")
            else:
                # Plot all peaks in gray
                (parc,) = plt.plot_date(arcs_dn, arcs, ".", markersize=2, color="gray")
                parc.set_label("Arcs")

    # Calculate and plot spline fit if requested
    if "kval_spectral" and "knots" in kwargs:
        kval_spectral = kwargs.get("kval_spectral")
        knots = kwargs.get("knots")
        knots_dn = gps2datenum(np.array(knots, dtype=float))

        # Interpolate spline at regular time intervals
        tt = np.linspace(knots[0], knots[-1], int((knots[-1] - knots[0]) / plotdt))
        spectral = cubspl_nans(tt, knots, kval_spectral)
        tt_dn = gps2datenum(tt)

        # Plot spline fit if requested
        if plotfig and plotspec:
            (pspec,) = plt.plot_date(tt_dn, spectral, "-", color="hotpink")
            pspec.set_label("Spline fit")
            if plotknots:
                kval_spectral_plot = kval_spectral
                (pknot,) = plt.plot_date(knots_dn, kval_spectral_plot, ".", markersize=10)
                pknot.set_label("knots")

    # Configure plot appearance
    if plotfig:
        plt.ylabel("Reflector height (m)")

        # Set x-axis limits and format
        if "xlims" in kwargs:
            xlims = kwargs.get("xlims")
        else:
            xlims = [np.min(arcs_dn), np.min(arcs_dn[-1])]
        ax.set_xlim(xlims[0], xlims[1])

        # Choose appropriate date format based on time span
        if xlims[1] - xlims[0] < 2:
            dformat = DateFormatter("%Hh")
        else:
            dformat = DateFormatter("%m-%d")
        ax.xaxis.set_major_formatter(dformat)

        # Set y-axis limits if specified
        if "ylims" in kwargs:
            ylims = kwargs.get("ylims")
            ax.set_ylim(ylims[0], ylims[1])

        ax.legend()

        # Display or save plot
        if not savefig or "outdir" not in kwargs:
            plt.show()
        else:
            outdir = kwargs.get("outdir")
            Path(outdir).mkdir(parents=True, exist_ok=True)
            plt.savefig(outdir + "/" + figoutstr, format="png", dpi=300)
            plt.close()
