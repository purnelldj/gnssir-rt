import numpy as np
from scipy import interpolate
from scipy.optimize import least_squares
from typing import Tuple


def elv_fit(t: np.ndarray, elv: np.ndarray, knots: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit elevation data using cubic spline interpolation with least squares
    optimization.

    Parameters
    ----------
    t : np.ndarray
        Time points corresponding to elevation measurements
    elv : np.ndarray
        Elevation measurements
    knots : np.ndarray
        Knot points for spline interpolation

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Tuple containing:
        - Time points used for fitting
        - Fitted elevation values

    Raises
    ------
    Exception
        If time points are outside the range of knots
    """
    # Use nearest neighbor interpolation to get initial control points
    nearestf = interpolate.interp1d(t, elv, kind="nearest")
    cp_in = nearestf(knots[1:-1])
    cp_in = np.append(elv[0], cp_in)
    cp_in = np.append(cp_in, elv[-1])

    elvcompare_t = elv
    tfit_t = t
    if np.min(tfit_t) < np.min(knots) or np.max(tfit_t) > np.max(knots):
        raise Exception("Time points must be within the range of knots")

    def resid_spline(cp: np.ndarray) -> np.ndarray:
        """Calculate residuals between measured and interpolated elevations."""
        ffit = interpolate.interp1d(knots, cp, kind="cubic")
        resid = elvcompare_t - ffit(tfit_t)
        return resid

    # Optimize control points using least squares
    cp_out = least_squares(resid_spline, cp_in, method="trf")
    ffit_t = interpolate.interp1d(knots, cp_out.x, kind="cubic")
    elvfit_t = ffit_t(tfit_t)
    elvfit_t = np.array(elvfit_t, dtype=float)
    return tfit_t, elvfit_t


def elv_interp_array_sat(tsnrd: np.ndarray, kspac: float) -> np.ndarray:
    """
    Interpolate elevation data for a single satellite with gaps handling.

    Parameters
    ----------
    tsnrd : np.ndarray
        Array containing satellite data with columns [sat_id, elevation, ...]
    kspac : float
        Knot spacing for spline interpolation

    Returns
    -------
    np.ndarray
        Interpolated satellite data array
    """
    ttsnrd_out = np.empty((0, 5))
    tdiff = np.ediff1d(tsnrd[:, 3])
    breaks = np.where(tdiff > kspac)[0]
    breaks = breaks + 1
    breaks = np.append(0, breaks)
    breaks = np.append(breaks, tsnrd.shape[0] - 1)
    breaks = np.array(breaks, dtype=int)
    tfit = []
    elvfit = []

    # Process each section between gaps
    for i in range(1, len(breaks)):
        ttsnrd = np.array(tsnrd[breaks[i - 1] : breaks[i]], dtype=float).copy()
        if ttsnrd.shape[0] < 2:
            continue

        t_in = ttsnrd[:, 3]
        elv_in = ttsnrd[:, 1]

        # Calculate knot points
        knot_s = np.min(t_in) - np.mod(np.min(t_in), kspac)
        knot_e = np.max(t_in) - np.mod(np.max(t_in), kspac) + kspac
        if knot_e < knot_s + 3 * kspac:
            continue

        knots = np.linspace(knot_s, knot_e, int((knot_e - knot_s) / kspac + 1))
        tfit, elvfit = elv_fit(t_in, elv_in, knots)

        # Update elevation values with interpolated values
        ttsnrd = ttsnrd[np.isin(t_in, tfit)]
        try:
            ttsnrd[:, 1] = elvfit.copy()
        except Exception as e:
            print(e)
            print("hoping this stops happening... ")
            continue
        ttsnrd_out = np.vstack((ttsnrd_out, ttsnrd))
    return ttsnrd_out


def elv_interp_array(snrdata: np.ndarray, kspac: float) -> np.ndarray:
    """
    Interpolate elevation data for multiple satellites.

    Parameters
    ----------
    snrdata : np.ndarray
        Array containing SNR data for multiple satellites
    kspac : float
        Knot spacing for spline interpolation

    Returns
    -------
    np.ndarray
        Interpolated SNR data array for all satellites
    """
    snrdata_interp = np.empty((0, 5))
    for sat in np.unique(snrdata[:, 0]):
        tsnrdata = snrdata[snrdata[:, 0] == sat].copy().astype(float)
        tsnrdata = tsnrdata[np.argsort(tsnrdata[:, 3])]
        if len(np.unique(tsnrdata[:, 1])) < 3:
            continue
        tsnrdata_interp = elv_interp_array_sat(tsnrdata, kspac)
        snrdata_interp = np.vstack((snrdata_interp, tsnrdata_interp.copy()))
    return snrdata_interp
