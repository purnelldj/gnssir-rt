import numpy as np
from astropy.time import Time
from matplotlib.dates import date2num
from scipy import interpolate


def readsnrtxt(snrfile):
    """ """
    snrdata = np.loadtxt(snrfile)
    if snrdata.shape[1] != 5:
        raise Exception("snr data should have shape (n, 5)")
    return snrdata


def datetime2gps(dt):
    timeobj = Time(dt, format="datetime")
    gpstime = timeobj.gps
    return gpstime


def gps2datetime(gt):
    timeobj = Time(gt, format="gps", scale="utc")
    dt = timeobj.datetime
    return dt


def gps2datenum(gt):
    timeobj = Time(gt, format="gps", scale="utc")
    dt = timeobj.datetime
    dn = date2num(dt)
    return dn


def glonasswlen(prn, gsignal):
    """
    this only works for L1 and L2
    """
    channel = [
        1,
        -4,
        5,
        6,
        1,
        -4,
        5,
        6,
        -2,
        -7,
        0,
        -1,
        -2,
        -7,
        0,
        -1,
        4,
        -3,
        3,
        2,
        4,
        -3,
        3,
        2,
    ]
    offset = 101  # 101 onwards is glonass
    try:
        channel_t = np.array([channel[i - offset] for i in prn], dtype=float)
    except TypeError:
        try:
            channel_t = channel[prn - offset]
        except IndexError:
            print("found an unknown glonass satellite with PRN " + str(prn))
            exit()
    if gsignal == "L1":
        lcar = 299792458 / (1602e06 + channel_t * 0.5625e06)
    elif gsignal == "L2":
        lcar = 299792458.0 / (1246e06 + channel_t * 0.4375e06)
    else:
        print("gsignal not recognised")
        lcar = np.nan
    return lcar


def cubspl_nans(tplot, knots, kval):
    tfilter = np.isnan(kval) == 0
    kval_red = kval[tfilter]
    knots_red = knots[tfilter]
    naninds = np.where(np.isnan(kval))[0]
    cubspl_f = interpolate.interp1d(knots_red, kval_red, kind="cubic")
    tfilter = np.logical_and(tplot >= np.min(knots_red), tplot <= np.max(knots_red))
    tplot_red = tplot[tfilter]
    rh_out = cubspl_f(tplot_red)
    # now add in parts before and after
    if np.ma.size(tplot, axis=0) > 1:
        if tplot_red[0] > tplot[0]:
            si = np.where(tplot[:] == tplot_red[0])[0]
            tnan = np.empty(si)
            tnan[:] = np.nan
            rh_out = np.append(tnan, rh_out)
        if tplot_red[-1] < tplot[-1]:
            ei = np.where(tplot[:] == tplot_red[-1])[0]
            tnan = np.empty(len(tplot) - ei - 1)
            tnan[:] = np.nan
            rh_out = np.append(rh_out, tnan)
    if len(naninds) > 0:
        for ni in naninds:
            if ni == 0:
                lb = knots[0]
                rh_out[0] = np.nan
            else:
                lb = knots[ni - 1]
            if ni == len(knots) - 1:
                ub = knots[-1]
                rh_out[-1] = np.nan
            else:
                ub = knots[ni + 1]
            tind = np.where(np.logical_and(tplot > lb, tplot < ub))
            rh_out[tind] = np.nan
    return rh_out


def residuals_cubspl_spectral(kval, knots, rh_arr):
    """
    function needed for inverse analysis
    """
    tfilter = np.logical_and(rh_arr[:, 0] >= knots[0], rh_arr[:, 0] <= knots[-1])
    rh_arr = rh_arr[tfilter]
    dt_even = 1 * 60
    t_even = np.linspace(knots[0], knots[-1], int((knots[-1] - knots[0]) / dt_even) + 1)
    cubspl_f = interpolate.interp1d(knots, kval, kind="cubic")
    cubspl_even = cubspl_f(t_even)
    dhdt_even = np.gradient(cubspl_even, dt_even)
    f = interpolate.interp1d(t_even, dhdt_even)
    tgpst = np.array(rh_arr[:, 0], dtype=float)
    reflh = np.array(rh_arr[:, 1], dtype=float)
    tane_dedt = np.array(rh_arr[:, 3], dtype=float)
    dhdt = f(tgpst)
    cubspl_adj = cubspl_f(tgpst) + dhdt * tane_dedt
    residual_spectral = cubspl_adj - reflh
    return residual_spectral
