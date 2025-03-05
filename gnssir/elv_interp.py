import numpy as np
from scipy import interpolate
from scipy.optimize import least_squares


def elv_fit(t, elv, knots):
    nearestf = interpolate.interp1d(t, elv, kind="nearest")
    cp_in = nearestf(knots[1:-1])
    cp_in = np.append(elv[0], cp_in)
    cp_in = np.append(cp_in, elv[-1])
    elvcompare_t = elv
    tfit_t = t
    if np.min(tfit_t) < np.min(knots) or np.max(tfit_t) > np.max(knots):
        raise Exception("this should not happen!")

    def resid_spline(cp):
        ffit = interpolate.interp1d(knots, cp, kind="cubic")
        resid = elvcompare_t - ffit(tfit_t)
        return resid

    cp_out = least_squares(resid_spline, cp_in, method="trf")
    ffit_t = interpolate.interp1d(knots, cp_out.x, kind="cubic")
    elvfit_t = ffit_t(tfit_t)
    # tfit_t = tfit_t[elvfit_t > 0]
    # elvfit_t = elvfit_t[elvfit_t > 0]
    elvfit_t = np.array(elvfit_t, dtype=float)
    return tfit_t, elvfit_t


def elv_interp_array_sat(tsnrd, kspac):
    ttsnrd_out = np.empty((0, 5))
    tdiff = np.ediff1d(tsnrd[:, 3])
    breaks = np.where(tdiff > kspac)[0]
    breaks = breaks + 1
    breaks = np.append(0, breaks)
    breaks = np.append(breaks, tsnrd.shape[0] - 1)
    breaks = np.array(breaks, dtype=int)
    tfit = []
    elvfit = []
    # then looping through section by section in between the gaps
    for i in range(1, len(breaks)):
        # print(f"{i} / {len(breaks) - 1}")
        ttsnrd = np.array(tsnrd[breaks[i - 1] : breaks[i]], dtype=float).copy()
        if ttsnrd.shape[0] < 2:
            # print('too small')
            continue
        t_in = ttsnrd[:, 3]
        elv_in = ttsnrd[:, 1]
        knot_s = np.min(t_in) - np.mod(np.min(t_in), kspac)
        knot_e = np.max(t_in) - np.mod(np.max(t_in), kspac) + kspac
        if knot_e < knot_s + 3 * kspac:
            # print('not enough knots..')
            continue
        # ttdiff = np.ediff1d(t_in)
        knots = np.linspace(knot_s, knot_e, int((knot_e - knot_s) / kspac + 1))
        tfit, elvfit = elv_fit(t_in, elv_in, knots)
        # ttt = tt[np.in1d(t_in, tfit)]
        ttsnrd = ttsnrd[np.isin(t_in, tfit)]
        try:
            ttsnrd[:, 1] = elvfit.copy()
        except Exception as e:
            print(e)
            print("hoping this stops happening... ")
            continue
        ttsnrd_out = np.vstack((ttsnrd_out, ttsnrd))
    return ttsnrd_out


def elv_interp_array(snrdata, kspac):
    snrdata_interp = np.empty((0, 5))
    for sat in np.unique(snrdata[:, 0]):
        # print(f"sat now is {str(int(sat))}")
        tsnrdata = snrdata[snrdata[:, 0] == sat].copy().astype(float)
        tsnrdata = tsnrdata[np.argsort(tsnrdata[:, 3])]
        if len(np.unique(tsnrdata[:, 1])) < 3:
            continue
        tsnrdata_interp = elv_interp_array_sat(tsnrdata, kspac)
        snrdata_interp = np.vstack((snrdata_interp, tsnrdata_interp.copy()))
    return snrdata_interp
