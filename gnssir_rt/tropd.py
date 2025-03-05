import os
import pickle

import numpy as np

from gnssir_rt.make_gpt import gpt2_1w, makegptfile


def tropd(gpst, rh_arr, iminelv=4, imaxelv=5, irh=1, iahgt=12, it=0, adjtype="allelv", **kwargs):
    """
    it = 0 means time variable GPT or it = 1 is static
    """
    lla = kwargs.get("lla")
    gptfile = kwargs.get("gptfile")
    if "gpt_init_file" in kwargs:
        gpt_init_file = kwargs.get("gpt_init_file")
    else:
        gpt_init_file = "gnssir_rt/gpt_1wA.pickle"
    if "metdatafile" in kwargs:
        metdatafile = kwargs.get("metdatafile")
        f = open(metdatafile, "rb")
        metdata = pickle.load(f)
        f.close()
        dtt = metdata[:, 0]
        idt = [idt for idt in range(len(dtt)) if dtt[idt] - gpst == np.min(np.abs(dtt - gpst))]
        pant = metdata[idt, 1]
        tant = metdata[idt, 2]
        eant = metdata[idt, 3]
    else:
        if not os.path.isfile(gptfile):
            makegptfile(gptfile, gpt_init_file, lla[0], lla[1])
        dmjd = gps2dmjd(gpst)
        pant, tant, _, _, eant, _, _, _, _ = gpt2_1w(gptfile, dmjd, lla[0], lla[1], lla[2], it)

    if adjtype == "minmaxelv":
        rh_arr[:, irh] = rh_arr[:, irh] + rh_arr[:, iahgt]
        rh_fac = corr_rh_facs(
            np.min(rh_arr[:, iminelv]),
            np.max(rh_arr[:, imaxelv]),
            pant,
            tant,
            eant,
        )
        rh_arr_adj = rh_arr
        rh_arr_adj[:, irh] = rh_arr[:, irh] + rh_arr[:, irh] * rh_fac
        rh_arr_adj[:, irh] = rh_arr_adj[:, irh] - rh_arr[:, iahgt]
    elif adjtype == "allelv":
        rh_arr[:, irh] = rh_arr[:, irh] + rh_arr[:, iahgt]
        rh_facs = corr_rh_facs(rh_arr[:, iminelv], rh_arr[:, imaxelv], pant, tant, eant)
        rh_arr[:, irh] = rh_arr[:, irh] + rh_arr[:, irh] * rh_facs
        rh_arr[:, irh] = rh_arr[:, irh] - rh_arr[:, iahgt]
        rh_arr_adj = rh_arr
        rh_fac = np.nanmean(rh_facs)
    elif adjtype == "noadj":
        rh_fac = corr_rh_facs(
            np.min(rh_arr[:, iminelv]),
            np.max(rh_arr[:, imaxelv]),
            pant,
            tant,
            eant,
        )
        rh_arr_adj = rh_arr
    return rh_fac, rh_arr_adj


def corr_rh_facs(elv_min, elv_max, pant, tant, eant):
    elv_min_corr = bend_eqn(pant, tant, elv_min)
    elv_max_corr = bend_eqn(pant, tant, elv_max)
    meanbendelv = (elv_min + elv_min_corr + elv_max + elv_max_corr) / 2

    elv_min_rad = elv_min / 180 * np.pi
    elv_max_rad = elv_max / 180 * np.pi
    elv_min_corr_rad = elv_min_corr / 180 * np.pi
    elv_max_corr_rad = elv_max_corr / 180 * np.pi
    meanbendcorrrad = (elv_min_corr_rad + elv_max_corr_rad) / 2
    meanrad = (elv_min_rad + elv_max_rad) / 2

    Nant = N_eqn(pant, tant, eant)

    if hasattr(elv_max_rad, "__len__") or elv_max_rad != elv_min_rad:
        xi = (elv_max_corr_rad - elv_min_corr_rad) / (elv_max_rad - elv_min_rad)
    else:
        xi = 0

    der = xi
    e = meanrad
    edash = meanbendelv / 180 * np.pi
    de = meanbendcorrrad
    rh_fac1 = Nant / (np.sin(edash) ** 2) * (np.cos(edash) / np.cos(e)) * (1 + der)
    rh_fac2 = -der + (np.sin(de) * np.tan(e) + 1 - np.cos(de)) * (1 + der)
    rh_facs = rh_fac1 + rh_fac2
    # rh_facs = 1 - (1 + Nant)  * (np.cos(edash) / np.cos(e)) * (1 + der)
    return rh_facs


def relhumtemp2e(relhum, temp):
    """
    RH in % (e.g., 50)
    temp in dC
    """
    tempk = temp + 273.16
    log10svp = (
        10.79574 * (1 - 273.16 / tempk)
        - 5.028 * np.log10(tempk / 273.16)
        + 1.50475 * 10**-4 * (1 - 10 ** (-8.2969 * (tempk / 273.16 - 1)))
        + 0.42873 * 10**-3 * (10 ** (-4.76955 * (1 - 273.16 / tempk) - 1))
        + 0.78614
    )
    svp = 10**log10svp
    e = relhum / 100 * svp
    return e


def bend_eqn(p, t, elv):
    bending_corr_arc_min = (
        510 / (9 / 5 * t + 492) * p / 1010.16 * 1 / np.tan(np.deg2rad(elv + 7.31 / (elv + 4.4)))
    )
    bending_corr_deg = bending_corr_arc_min / 60
    return bending_corr_deg


def N_eqn(p, t, e):
    k1 = 77.604
    k2 = 70.4
    k3 = 373900
    pd = p - e
    tk = t + 273.15
    N = (k1 * pd / tk + k2 * e / tk + k3 * e / tk**2) / 1e6
    return N


def gps2dmjd(gpst):
    # dmjd is 59215 on 1 Jan 2021
    # gpstime is 1293494418 (seconds)
    # that's 14971 days and 18 (leap) seconds since GPS time started
    # so, to get from gpstime to mjd
    # dmjd = (gpst - 18) / 86400 - 14971 + 59215 = (gpst - 18) / 86400 + 44244
    dmjd = (gpst - 18) / 86400 + 44244
    return dmjd
