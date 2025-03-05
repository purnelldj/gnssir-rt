import datetime
import pickle
import time
from os import listdir
from pathlib import Path

import numpy as np
from astropy.timeseries import LombScargle
from scipy.optimize import leastsq
from scipy.signal import find_peaks

from gnssir_rt.elv_interp import elv_interp_array
from gnssir_rt.helper import (
    cubspl_nans,
    datetime2gps,
    glonasswlen,
    gps2datetime,
    readsnrtxt,
    residuals_cubspl_spectral,
)
from gnssir_rt.plot import plotrhspline
from gnssir_rt.tropd import tropd


def snr2arc(
    snrdata,
    rhlims,
    gsignal="L1",
    polydeg=2,
    gaptlim=5 * 60,
    detrend=True,
    **kwargs,
):
    """
    reads an array of snr data and organises into:
    reflector height estimates, stats and detrended snr data for inverse analysis
    :param snrdata: array of organised SNR data
    :param rhlims: upper and lower reflector height limits (in metres) for quality control
    :param gsignal: default 'L1'
    :param polydeg: degree of polynomial to fit and subtract from SNR data
    :param gaptlim: if there is a gap in time bigger than [gaptlim] seconds in a particular arc then it will be ignored
    :return rh_arr: numpy array of reflector height estimtes and stats
    :return snrdt_arr: numpy array of detrended SNR data for inverse analysis
    """

    if detrend:
        # now need to convert SNR to linear scale (from dB-Hz)
        snrdata[:, 4] = 10 ** (snrdata[:, 4] / 20)

    if "tempres" in kwargs:
        tempres = kwargs.get("tempres")
        tfilter = np.where(np.mod(snrdata[:, 3] - snrdata[0, 3], tempres) == 0)[0]
        snrdata = snrdata[tfilter]

    if "satconsts" in kwargs:
        satconsts = kwargs.get("satconsts")
        if "G" not in satconsts:
            nogps = snrdata[:, 0] > 100
            snrdata = snrdata[nogps]
        if "R" not in satconsts:
            noglo = np.logical_or(snrdata[:, 0] < 100, snrdata[:, 0] > 200)
            snrdata = snrdata[noglo]
        if "E" not in satconsts:
            nogal = np.logical_or(snrdata[:, 0] < 200, snrdata[:, 0] > 300)
            snrdata = snrdata[nogal]

    # elvlims and azilims
    if "elvlims" in kwargs:
        elvlims = kwargs.get("elvlims")
        tfilter = np.logical_and(snrdata[:, 1] > elvlims[0], snrdata[:, 1] < elvlims[1])
        snrdata = snrdata[tfilter]
    if "azilims" in kwargs:
        azilims = kwargs.get("azilims")
        if azilims[0] < azilims[1]:
            tfilter = np.logical_and(snrdata[:, 2] > azilims[0], snrdata[:, 2] < azilims[1])
        else:
            tfilter = np.logical_or(snrdata[:, 2] > azilims[0], snrdata[:, 2] < azilims[1])
        snrdata = snrdata[tfilter]

    rh_arr = np.empty((0, 12), dtype=object)
    snrdt_arr = np.empty((0, 5), dtype=object)

    nobadsats = np.logical_or(snrdata[:, 0] < 33, snrdata[:, 0] > 99)
    snrdata = snrdata[nobadsats, :]

    for sat in np.unique(snrdata[:, 0]):
        tfilter = snrdata[:, 0] == sat
        tempd = snrdata[tfilter]
        if gsignal != "L1" and gsignal != "L2":
            print("code only currently works for L1 and L2 - exiting")
            exit()
        if np.logical_or(sat < 100, np.logical_and(sat > 200, sat < 300)):
            if gsignal == "L1":
                lcar = 299792458 / 1575.42e06
            elif gsignal == "L2":
                lcar = 299792458 / 1227.60e06
        elif np.logical_and(sat > 100, sat < 200):
            if sat > 124:
                continue
            lcar = glonasswlen(int(sat), gsignal)
        maxf = 2 * (rhlims[0] + rhlims[1]) / lcar
        precisionf = 2 * 0.001 / lcar  # 1 mm
        f = np.linspace(precisionf, maxf, int(maxf / precisionf))
        tempd = tempd[tempd[:, 3].argsort()]  # sort by time
        elv_tosort = np.array(tempd[:, 1], dtype=float)
        date_tosort = np.array(tempd[:, 3], dtype=float)
        ddate = np.ediff1d(date_tosort)
        delv = np.ediff1d(elv_tosort)
        bkpt = len(ddate)
        bkpt = np.append(bkpt, np.where(ddate > gaptlim)[0])  # gaps bigger than gaptlim
        bkpt = np.append(bkpt, np.where(np.diff(np.sign(delv)))[0])  # elevation rate changes direction
        bkpt = np.unique(bkpt)
        bkpt = np.sort(bkpt)
        for ii in range(len(bkpt)):
            if ii == 0:
                sind = 0
            else:
                sind = bkpt[ii - 1] + 1
            eind = bkpt[ii] + 1
            if eind - sind < 20:
                # print('arc not big enough')
                continue
            elvt = np.array(tempd[sind:eind, 1], dtype=float)
            if len(np.unique(elvt)) == 1:
                # print('unchanging elevation')
                continue
            azit = np.array(tempd[sind:eind, 2], dtype=float)
            sinelvt = np.sin(elvt / 180 * np.pi)
            datet = np.array(tempd[sind:eind, 3], dtype=float)
            if detrend:
                snrt = np.array(tempd[sind:eind, 4], dtype=float)
                z = np.polyfit(sinelvt, snrt, polydeg)
                p = np.poly1d(z)
                snrdt = snrt - p(sinelvt)
            else:
                snrdt = np.array(tempd[sind:eind, 4], dtype=float)
            pgram = LombScargle(sinelvt, snrdt, normalization="psd").power(f)
            pgram = 2 * np.sqrt(pgram / len(sinelvt))  # normalizing
            reflh = 0.5 * f * lcar
            tfilter = np.logical_and(reflh > rhlims[0], reflh < rhlims[1])
            pgram_sub = pgram[tfilter]
            reflh_sub = reflh[tfilter]
            peaks = find_peaks(pgram_sub)[0]
            peakssort = np.argsort(pgram_sub[peaks])
            maxind = np.argmax(pgram_sub)
            peaks = peaks[peakssort]
            pktn = np.max(pgram_sub) / np.mean(pgram_sub)
            curind = maxind
            if curind != 0 and curind != len(pgram_sub) - 1:  # no peaks at either end of window
                temp_arr = np.empty((1, 12), dtype=object)
                # here just saving some stats for each satellite arc
                arcid = str(int(sat)) + "_" + str(int(np.min(datet)))
                temp_arr[0, 0] = int(np.round(np.mean(datet)))  # time of arc
                temp_arr[0, 1] = reflh_sub[curind]  # reflector height
                temp_arr[0, 2] = sat  # sat prn
                dthdt = ((elvt[-1] - elvt[0]) / 180 * np.pi) / (datet[-1] - datet[0])
                temp_arr[0, 3] = np.tan(np.mean(elvt) / 180 * np.pi) / dthdt  # tane / (de/dt)
                temp_arr[0, 4] = np.min(elvt)  # min elv
                temp_arr[0, 5] = np.max(elvt)  # max elv
                temp_arr[0, 6] = np.mean(azit)  # mean azimuth
                temp_arr[0, 7] = pgram_sub[curind]  # peak of lsp
                temp_arr[0, 8] = np.var(snrdt)  # variance of lsp
                temp_arr[0, 9] = datet[-1] - datet[0]  # length (in time) of arc
                temp_arr[0, 10] = pktn  # peak-to-noise ratio
                temp_arr[0, 11] = arcid
                rh_arr = np.vstack((rh_arr, temp_arr))
                satt = np.empty((len(datet)), dtype=int)
                satt[:] = sat
                arcidt = np.empty((len(datet)), dtype=object)
                arcidt[:] = arcid
                temp_arr = np.column_stack([datet, satt, sinelvt, snrdt, arcidt])
                snrdt_arr = np.vstack((snrdt_arr, temp_arr))
    # make sure that arrays are sorted by time
    rh_arr = rh_arr[rh_arr[:, 0].argsort()]
    snrdt_arr = snrdt_arr[snrdt_arr[:, 0].argsort()]
    return rh_arr, snrdt_arr


def snr2arcs(
    sdt,
    edt,
    snrdir,
    arcdir,
    rhlims,
    arclim,
    iterdt_arcs,
    antennaids=[""],
    **kwargs,
):
    snrdir_parent = snrdir
    arcdir_parent = arcdir
    for aid in antennaids:
        if len(aid) > 0:
            snrdir = snrdir_parent + "/" + aid
            arcdir = arcdir_parent + "/" + aid
        print(f"snrdir: {snrdir}")
        print(f"arcdir: {arcdir}")
        snr2arc_iterate(sdt, edt, snrdir, arcdir, rhlims, arclim, iterdt_arcs, **kwargs)


def snr2arc_iterate(
    sdt,
    edt,
    snrdir,
    arcdir,
    rhlims,
    arclim,
    iterdt_arcs,
    snrfilelen=False,
    gsignal="L1",
    elvinterp=True,
    kdt_elvinterp=1800,
    offset_elvinterp=1800,
    **kwargs,
):
    """
    :param sdt: datetime format of start date and time
    :param edt: datetime format of end date and time
    :param snrdir: string to directory where snr data is contained (i.e., from nmea2snr function output)
    :param arcdir: string path to output directory
    :param rhlims: reflector height limits, e.g., [4, 6] (in meters)
    :param arclim: time limit applied to arcs (seconds)
    :param iterdt_arcs: timestep (seconds)
    :param snrfilelen:
    :param gsignal: 'L1'
    :param elvinterp: if you want to interpolate the elevation for nema data
    :param kdt_elvinterp: knot spacing of elevation angle interpolation
    """

    iterdt_td = datetime.timedelta(seconds=iterdt_arcs)
    arclim_td = datetime.timedelta(seconds=arclim)
    if snrfilelen:
        snrfilelen_td = datetime.timedelta(seconds=snrfilelen)
    tdt = sdt - iterdt_td
    while tdt < edt - iterdt_td:
        tdt = tdt + iterdt_td
        tdts = tdt
        tdte = tdt + arclim_td
        print(str(tdt.date()) + " " + str(tdt.time()))
        tgts = datetime2gps(tdts)
        tgte = datetime2gps(tdte)
        if elvinterp:
            tdts = tdts - datetime.timedelta(seconds=offset_elvinterp)
            tgts_interp = datetime2gps(tdts)
        else:
            tgts_interp = tgts
        snrdata = np.empty((0, 5))

        # NOW ITERATING THROUGH
        ttdt = tdts - snrfilelen_td
        if elvinterp and offset_elvinterp > 0:
            ttdt = tdts - snrfilelen_td
        while ttdt < tdte - snrfilelen_td:
            ttdt = ttdt + snrfilelen_td
            snrfilestr = snrdir + "/" + ttdt.strftime("%y_%m_%d_%H") + ".snr"
            try:
                snrdatat = readsnrtxt(snrfilestr)
                snrdata = np.vstack((snrdata, snrdatat))
            except FileNotFoundError:
                print("missing snr file...")
                continue

        snrdata = np.unique(snrdata, axis=0)
        snrdata = snrdata[snrdata[:, 0].argsort()]
        snrdata = snrdata[snrdata[:, 3].argsort()]
        tfilter = np.logical_and(snrdata[:, 3] >= tgts_interp, snrdata[:, 3] < tgte)
        snrdata = snrdata[tfilter]
        if np.ma.size(snrdata, 0) == 0:
            print("mssing data")
            continue

        if elvinterp:
            print("interpolating elevation")
            timey = time.time()
            snrdata = elv_interp_array(snrdata, kdt_elvinterp)
            tfilter = np.logical_and(snrdata[:, 3] >= tgts, snrdata[:, 3] < tgte)
            print(f"took {(time.time() - timey):.1f} seconds to interpolate elv ")
            snrdata = snrdata[tfilter]
            if np.ma.size(snrdata, 0) == 0:
                print("mssing data")
                continue

        timey = time.time()
        rh_arr, snrdt_arr = snr2arc(snrdata, rhlims, gsignal=gsignal, **kwargs)
        print(f"took {(time.time() - timey):.2f} seconds to convert to arcs ")
        Path(arcdir).mkdir(parents=True, exist_ok=True)
        invfilestr = arcdir + "/" + str(tdt.strftime("%y_%m_%d_%H_%M")) + ".pkl"
        f = open(invfilestr, "wb")
        pickle.dump(rh_arr, f)
        pickle.dump(snrdt_arr, f)
        f.close()
        print("dumped a pickle with " + str(np.ma.size(rh_arr, axis=0)) + " arcs")


def collectarcs(
    arcdir,
    sdt,
    edt,
    hgts,
    antennaids,
    collectsnr=False,
    arclim=24 * 60 * 60,
    **kwargs,
):
    if len(antennaids) != len(hgts):
        print("need to give an input height for each antenna")
        exit()
    arcdir_parent = arcdir
    arcdir = [arcdir_parent + "/" + aid for aid in antennaids]
    arclim_td = datetime.timedelta(seconds=arclim)
    rh_arr = np.empty((0, 13))
    snrdt_arr = np.empty((0, 5))
    tfdates = []
    for arcd in arcdir:
        tfs = listdir(arcd)
        tfs = [tf for tf in tfs if tf[-4:] == ".pkl"]
        ttfdates = [datetime.datetime.strptime(tf[0:14], "%y_%m_%d_%H_%M") for tf in tfs]
        tfdates = np.append(tfdates, ttfdates)
    tfdates = np.unique(tfdates)
    try:
        # sdt = tfdates[np.min(np.where(tfdates >= sdt)[0])]
        sdt = tfdates[np.max(np.where(tfdates <= sdt)[0])]
    except ValueError:
        print("no arc files found")
        return rh_arr, snrdt_arr
    tdt = sdt - arclim_td
    # while tdt < edt - arclim_td:
    while tdt < edt:
        tdt = tdt + arclim_td
        rh_arrt = np.empty((0, 13))
        snrdt_arrt = np.empty((0, 6))
        for ii in range(len(arcdir)):
            tdtf = arcdir[ii] + "/" + str(tdt.strftime("%y_%m_%d_%H_%M")) + ".pkl"
            try:
                f = open(tdtf, "rb")
                rh_arrtt = pickle.load(f)
                snrdt_arrtt = pickle.load(f)
                f.close()
                rh_arrtt[:, 1] = rh_arrtt[:, 1] - hgts[ii]
                snrdt_arrtt = np.column_stack((snrdt_arrtt, np.empty(np.ma.size(snrdt_arrtt, axis=0))))
                snrdt_arrtt[:, 5] = hgts[ii]
                arcid = False
                if arcid:
                    newarcid = [str(ii) + "_" + oldid for oldid in rh_arrtt[:, 11]]
                    rh_arrtt[:, 11] = newarcid
                    newarcid = [str(ii) + "_" + oldid for oldid in snrdt_arrtt[:, 4]]
                    snrdt_arrtt[:, 4] = newarcid
                rh_arrtt = np.column_stack((rh_arrtt, np.empty(np.ma.size(rh_arrtt, axis=0))))
                rh_arrtt[:, 12] = hgts[ii]
                rh_arrt = np.vstack((rh_arrt, rh_arrtt))
                snrdt_arrt = np.vstack((snrdt_arrt, snrdt_arrtt))
            except IOError:
                # print('missing file')
                continue
        rh_arrt, snrdt_arrt = arcsqc(rh_arrt, snrdt_arrt, **kwargs)
        rh_arr = np.vstack((rh_arr, rh_arrt))
        if collectsnr:
            snrdt_arr = np.vstack((snrdt_arr, snrdt_arrt))

    # now delete points otuside of range
    sgt = datetime2gps(sdt)
    egt = datetime2gps(edt)
    tfilter = np.logical_and(rh_arr[:, 0] >= sgt, rh_arr[:, 0] < egt)
    rh_arr = rh_arr[tfilter]
    tfilter = np.logical_and(snrdt_arr[:, 0] >= sgt, snrdt_arr[:, 0] < egt)
    snrdt_arr = snrdt_arr[tfilter]
    # [print(gps2datetime(gt)) for gt in rh_arr[:, 0]]
    if np.ma.size(rh_arr, axis=0) == 0:
        print("no arcs found after qc")
    else:
        rh_arr = np.array(rh_arr, dtype=float)
        rh_arr = np.unique(rh_arr, axis=0)
        snrdt_arr = np.array(snrdt_arr, dtype=float)
        snrdt_arr = np.unique(snrdt_arr, axis=0)
    return rh_arr, snrdt_arr


def arcsqc(rh_arr, snrdt_arr, qc_std=True, **kwargs):
    """ """

    if "rhlims" in kwargs:
        rhlims = kwargs.get("rhlims")
        temprh = rh_arr[:, 1]
        # if 'rh_datum' in kwargs:
        #    rh_datum = kwargs.get('rh_datum')
        #    temprh = rh_datum - temprh
        tfilter = np.logical_and(temprh > rhlims[0], temprh < rhlims[1])
        rh_arr = rh_arr[tfilter, :]

    if "elvlims" in kwargs:
        elvlims = kwargs.get("elvlims")
        tfilter = np.logical_and(rh_arr[:, 4] > elvlims[0], rh_arr[:, 5] < elvlims[1])
        rh_arr = rh_arr[tfilter, :]

    if "azilims" in kwargs:
        azilims = kwargs.get("azilims")
        if azilims[0] < azilims[1]:
            tfilter = np.logical_and(rh_arr[:, 6] > azilims[0], rh_arr[:, 6] < azilims[1])
        else:
            tfilter = np.logical_or(rh_arr[:, 6] > azilims[0], rh_arr[:, 6] < azilims[1])
        rh_arr = rh_arr[tfilter, :]

    if "minpsd" in kwargs:
        minpsd = kwargs.get("minpsd")
        if np.ma.size(rh_arr, axis=0) > 0:
            rh_arr = rh_arr[np.logical_or(rh_arr[:, 7] > minpsd, rh_arr[:, 11] == 1), :]

    if "pktnlim" in kwargs:
        pktnlim = kwargs.get("pktnlim")
        if np.ma.size(rh_arr, axis=0) > 0:
            rh_arr = rh_arr[rh_arr[:, 10] > pktnlim, :]

    if qc_std:
        if "stdfac" in kwargs:
            stdfac = kwargs.get("stdfac")
        else:
            stdfac = 3
        fixed_std = False
        if "fixed_std" in kwargs:
            fixed_std = kwargs.get("fixed_std")
        temprh = rh_arr[:, 1]
        keepid = []
        for tti in range(len(temprh)):
            temprhtt = [temprh[i] for i in range(len(temprh)) if i != tti]
            tempmean = np.mean(temprhtt)
            tempdiff = np.abs(temprh[tti] - tempmean)
            if fixed_std:
                if tempdiff < fixed_std:
                    keepid = np.append(keepid, tti)
            else:
                tempstd = np.std(temprh)
                if tempdiff < stdfac * tempstd:
                    keepid = np.append(keepid, tti)
        keepid = np.array(keepid, dtype=int)
        # print(keepid.shape, rh_arr.shape)
        rh_arr = rh_arr[keepid]

    if "snrqc" in kwargs:
        snrqc = kwargs.get("snrqc")
        if snrqc:
            snrdt_arr = snrdt_arr[np.isin(snrdt_arr[:, 4], np.unique(rh_arr[:, 11]))]
    snrdt_arr[:, 4] = snrdt_arr[:, 5]  # the column going into arcs2spline needs to be the hgt
    snrdt_arr = snrdt_arr[:, :5]
    return rh_arr, snrdt_arr


def arcsplot(arcdir, sdt, edt, hgts, antennaids, arclim=60 * 60, **kwargs):
    rh_arr, _ = collectarcs(arcdir, sdt, edt, hgts, antennaids, arclim=arclim, **kwargs)
    plotrhspline(rh_arr, plotrh=True, **kwargs)


def arcs2spline(rh_arr, snrdt_arr, knots, doplot=False, tropd_adj=True, **kwargs):
    """
    first and last knots will be ignored
    """
    invout = {}
    kdt = np.max(np.ediff1d(knots))
    if np.ma.size(rh_arr, axis=0) == 0:
        print("no reflector height data - exit")
        return invout, rh_arr
    if "satconsts" in kwargs:
        satconsts = kwargs.get("satconsts")
    else:
        allsats = np.unique(rh_arr[:, 2])
        satconsts = []
        if len(np.where(np.logical_and(allsats > 0, allsats < 100))[0]) > 0:
            satconsts = np.append(satconsts, "G")
        if len(np.where(np.logical_and(allsats > 100, allsats < 200))[0]) > 0:
            satconsts = np.append(satconsts, "R")
        if len(np.where(np.logical_and(allsats > 200, allsats < 300))[0]) > 0:
            satconsts = np.append(satconsts, "E")
    if np.ma.size(rh_arr, axis=0) < 2:
        print("not enough data - exit")
        return invout, rh_arr

    if tropd_adj:
        if "lla" and "gptfile" in kwargs:
            tdt = np.median(knots)
            rh_fac, rh_arr_adj = tropd(tdt, rh_arr, **kwargs)
            rh_arr = rh_arr_adj
            # print('the mean tropd adjustment should be around ' +
            #      str(np.round((rh_fac * np.nanmean(rh_arr[:, 1] + rh_arr[:, 12])) * 100, 2)) + ' cm')
        else:
            print("missing parameters for tropd adjustment - add to kwargs")

    temp_dn = np.sort(rh_arr[:, 0])
    # temp_dn = np.append(knots[0], temp_dn)
    # temp_dn = np.append(temp_dn, knots[-1])
    maxtgap = np.max(np.ediff1d(temp_dn))
    mintgap = np.min(np.ediff1d(temp_dn))
    if mintgap < 0:
        print("issue - values not in order")
        exit()
    # print('max gap is ' + str(int(maxtgap / 60)) + ' minutes')

    if maxtgap > kdt * 1.05:  # giving 5% margin?
        print("gap in data bigger than node spacing - continue with risk of instabilities if you want")
        print("continuing")
        # return invout, rh_arr

    def residuals_spectral_ls(kval):
        residuals = residuals_cubspl_spectral(kval, knots, rh_arr)
        return residuals

    kval_0 = np.nanmean(rh_arr[:, 1]) * np.ones(len(knots))
    # prels = time.time()
    try:
        ls_out = leastsq(residuals_spectral_ls, kval_0, full_output=True)
    except TypeError:
        print("input error - probably no data")
        return invout, rh_arr
    kval_spectral = ls_out[0]
    # print('ls took %.3f seconds' %(time.time() - prels))
    for ii in range(len(knots) - 1):
        findrh = np.logical_and(rh_arr[:, 0] >= knots[ii], rh_arr[:, 0] <= knots[ii + 1])
        if np.ma.size(rh_arr[findrh], axis=0) == 0:
            print("got rid of a kval")
            print("#################################")
            kval_spectral[ii + 1] = np.nan
    invout["knots"] = knots
    invout["kval_spectral"] = kval_spectral

    if doplot:
        plotrhspline(rh_arr, knots=knots, kval_spectral=kval_spectral, **kwargs)

    return invout, rh_arr


def arcs2splines(
    arcdir,
    sdt,
    edt,
    iterdt_spline,
    kdt,
    prek,
    postk,
    hgts,
    antennaids,
    arclim=60 * 60,
    savespline=False,
    **kwargs,
):
    """ """
    tdt = datetime2gps(sdt) - iterdt_spline
    knots_all = np.empty(0)
    kval_spectral = np.empty(0)
    rh_arr_all = np.empty((0, 12))
    while tdt < datetime2gps(edt) - iterdt_spline:
        tdt = tdt + iterdt_spline
        tdt_dt = gps2datetime(tdt)
        # load all of the file names and then look for data that's within window
        print(str(tdt_dt))
        tdts = tdt - prek * kdt  # start of time period
        tdte = tdt + postk * kdt  # end of time period
        knots = np.linspace(tdts, tdte, int(prek + postk + 1))
        knots_all = np.append(knots_all, tdt)
        rh_arr, snrdt_arr = collectarcs(
            arcdir,
            gps2datetime(knots[0]),
            gps2datetime(knots[-1]),
            hgts,
            antennaids,
            arclim=arclim,
            **kwargs,
        )

        if np.ma.size(rh_arr, axis=0) == 0:
            kval_spectral = np.append(kval_spectral, np.nan)
            continue

        invout, rh_arr = arcs2spline(rh_arr, snrdt_arr, knots, **kwargs)
        rh_arr = rh_arr[
            np.logical_and(
                rh_arr[:, 0] > tdt - iterdt_spline,
                rh_arr[:, 0] <= tdt + iterdt_spline,
            ),
            :,
        ]
        if len(invout) == 0:
            kval_spectral = np.append(kval_spectral, np.nan)
            continue
        if "latency" in kwargs:
            # for testing out in increments
            latency = kwargs.get("latency")
            knott = invout["knots"]
            tplott = np.linspace(knott[0], knott[-1], int(knott[-1] - knott[0] + 1))
            try:
                spectspline = cubspl_nans(tplott, invout["knots"], invout["kval_spectral"])
                idlat = latency + 1
                spect = spectspline[-idlat]
                offt = tplott[-idlat]
                knots_all[-1] = offt
                kval_spectral = np.append(kval_spectral, spect)
            except ValueError:
                kval_spectral = np.append(kval_spectral, np.nan)
        else:
            invout["kval_spectral"] = invout["kval_spectral"][prek]
            kval_spectral = np.append(kval_spectral, invout["kval_spectral"])

        rh_arr_all = np.vstack((rh_arr_all, rh_arr[:, :12]))

    if savespline and "outdir" in kwargs:
        outdir = kwargs.get("outdir")
        Path(outdir).mkdir(parents=True, exist_ok=True)
        invoutstr = outdir + "/splineout.pkl"
        invout = {}
        invout["knots"] = knots_all
        invout["kval_spectral"] = kval_spectral
        f = open(invoutstr, "wb")
        pickle.dump(rh_arr_all, f)
        pickle.dump(invout, f)
        f.close()

    plotrhspline(rh_arr_all, knots=knots_all, kval_spectral=kval_spectral, **kwargs)

    return invout
