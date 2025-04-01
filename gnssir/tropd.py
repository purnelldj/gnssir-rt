import os
import pickle
from typing import Any, Tuple, Union

import numpy as np

from gnssir.make_gpt import gpt2_1w, makegptfile


def tropd(
    gpst: float,
    rh_arr: np.ndarray,
    iminelv: int = 4,
    imaxelv: int = 5,
    irh: int = 1,
    iahgt: int = 12,
    it: int = 0,
    adjtype: str = "allelv",
    **kwargs: Any,
) -> Tuple[float, np.ndarray]:
    """
    Apply tropospheric delay corrections to reflector height estimates.

    This function applies tropospheric delay corrections to reflector height estimates
    using either GPT (Global Pressure and Temperature) model or meteorological data.
    The corrections account for atmospheric bending and refractive index effects.

    Parameters
    ----------
    gpst : float
        GPS time in seconds
    rh_arr : np.ndarray
        Array containing reflector height estimates and related data
    iminelv : int, optional
        Column index for minimum elevation angle, by default 4
    imaxelv : int, optional
        Column index for maximum elevation angle, by default 5
    irh : int, optional
        Column index for reflector height, by default 1
    iahgt : int, optional
        Column index for antenna height, by default 12
    it : int, optional
        GPT model type (0=time variable, 1=static), by default 0
    adjtype : str, optional
        Type of adjustment to apply:
        - "allelv": Apply correction for each elevation angle
        - "minmaxelv": Apply correction using min/max elevation angles
        - "noadj": No adjustment, just calculate correction factor
        by default "allelv"
    **kwargs : Any
        Additional keyword arguments:
        - lla: List[float], [latitude, longitude, height] in degrees and meters
        - gptfile: str, path to GPT model file
        - gpt_init_file: str, path to initial GPT model file
        - metdatafile: str, path to meteorological data file

    Returns
    -------
    Tuple[float, np.ndarray]
        - rh_fac: float, mean tropospheric delay correction factor
        - rh_arr_adj: np.ndarray, array with corrected reflector heights

    Notes
    -----
    The function uses either GPT model or meteorological data to calculate
    tropospheric delay corrections. The corrections account for atmospheric
    bending and refractive index effects on GNSS signals.
    """
    # Get location and model parameters
    lla = kwargs.get("lla")
    gptfile = kwargs.get("gptfile")
    if "gpt_init_file" in kwargs:
        gpt_init_file = kwargs.get("gpt_init_file")
    else:
        gpt_init_file = "gnssir_rt/gpt_1wA.pickle"

    # Get meteorological data
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
        # Use GPT model if no meteorological data
        if not os.path.isfile(gptfile):
            makegptfile(gptfile, gpt_init_file, lla[0], lla[1])
        dmjd = gps2dmjd(gpst)
        pant, tant, _, _, eant, _, _, _, _ = gpt2_1w(gptfile, dmjd, lla[0], lla[1], lla[2], it)

    # Apply corrections based on adjustment type
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


def corr_rh_facs(
    elv_min: Union[float, np.ndarray],
    elv_max: Union[float, np.ndarray],
    pant: float,
    tant: float,
    eant: float,
) -> Union[float, np.ndarray]:
    """
    Calculate tropospheric delay correction factors.

    This function calculates correction factors for tropospheric delay effects
    based on elevation angles and atmospheric parameters.

    Parameters
    ----------
    elv_min : Union[float, np.ndarray]
        Minimum elevation angle(s) in degrees
    elv_max : Union[float, np.ndarray]
        Maximum elevation angle(s) in degrees
    pant : float
        Atmospheric pressure in hPa
    tant : float
        Temperature in Celsius
    eant : float
        Water vapor pressure in hPa

    Returns
    -------
    Union[float, np.ndarray]
        Correction factor(s) for tropospheric delay

    Notes
    -----
    The function uses atmospheric bending and refractive index equations to
    calculate the correction factors. It can handle both single values and
    arrays of elevation angles.
    """
    # Calculate bending corrections
    elv_min_corr = bend_eqn(pant, tant, elv_min)
    elv_max_corr = bend_eqn(pant, tant, elv_max)
    meanbendelv = (elv_min + elv_min_corr + elv_max + elv_max_corr) / 2

    # Convert to radians
    elv_min_rad = elv_min / 180 * np.pi
    elv_max_rad = elv_max / 180 * np.pi
    elv_min_corr_rad = elv_min_corr / 180 * np.pi
    elv_max_corr_rad = elv_max_corr / 180 * np.pi
    meanbendcorrrad = (elv_min_corr_rad + elv_max_corr_rad) / 2
    meanrad = (elv_min_rad + elv_max_rad) / 2

    # Calculate refractive index
    Nant = N_eqn(pant, tant, eant)

    # Calculate bending ratio
    if hasattr(elv_max_rad, "__len__") or elv_max_rad != elv_min_rad:
        xi = (elv_max_corr_rad - elv_min_corr_rad) / (elv_max_rad - elv_min_rad)
    else:
        xi = 0

    # Calculate correction factors
    der = xi
    e = meanrad
    edash = meanbendelv / 180 * np.pi
    de = meanbendcorrrad
    rh_fac1 = Nant / (np.sin(edash) ** 2) * (np.cos(edash) / np.cos(e)) * (1 + der)
    rh_fac2 = -der + (np.sin(de) * np.tan(e) + 1 - np.cos(de)) * (1 + der)
    rh_facs = rh_fac1 + rh_fac2
    return rh_facs


def relhumtemp2e(relhum: float, temp: float) -> float:
    """
    Convert relative humidity and temperature to water vapor pressure.

    This function calculates water vapor pressure using the Magnus-Tetens
    approximation based on relative humidity and temperature.

    Parameters
    ----------
    relhum : float
        Relative humidity in percent (0-100)
    temp : float
        Temperature in Celsius

    Returns
    -------
    float
        Water vapor pressure in hPa

    Notes
    -----
    The function uses the Magnus-Tetens approximation to calculate
    saturation vapor pressure and then applies the relative humidity
    to get the actual water vapor pressure.
    """
    # Convert temperature to Kelvin
    tempk = temp + 273.16

    # Calculate saturation vapor pressure using Magnus-Tetens approximation
    log10svp = (
        10.79574 * (1 - 273.16 / tempk)
        - 5.028 * np.log10(tempk / 273.16)
        + 1.50475 * 10**-4 * (1 - 10 ** (-8.2969 * (tempk / 273.16 - 1)))
        + 0.42873 * 10**-3 * (10 ** (-4.76955 * (1 - 273.16 / tempk) - 1))
        + 0.78614
    )
    svp = 10**log10svp

    # Calculate water vapor pressure
    e = relhum / 100 * svp
    return e


def bend_eqn(p: float, t: float, elv: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate atmospheric bending correction.

    This function calculates the atmospheric bending correction for GNSS signals
    based on pressure, temperature, and elevation angle.

    Parameters
    ----------
    p : float
        Atmospheric pressure in hPa
    t : float
        Temperature in Celsius
    elv : Union[float, np.ndarray]
        Elevation angle(s) in degrees

    Returns
    -------
    Union[float, np.ndarray]
        Bending correction(s) in degrees

    Notes
    -----
    The function uses a simplified atmospheric bending model that accounts for
    pressure and temperature effects on signal propagation.
    """
    # Calculate bending correction in arc minutes
    bending_corr_arc_min = (
        510 / (9 / 5 * t + 492) * p / 1010.16 * 1 / np.tan(np.deg2rad(elv + 7.31 / (elv + 4.4)))
    )
    # Convert to degrees
    bending_corr_deg = bending_corr_arc_min / 60
    return bending_corr_deg


def N_eqn(p: float, t: float, e: float) -> float:
    """
    Calculate atmospheric refractive index.

    This function calculates the atmospheric refractive index using pressure,
    temperature, and water vapor pressure.

    Parameters
    ----------
    p : float
        Total atmospheric pressure in hPa
    t : float
        Temperature in Celsius
    e : float
        Water vapor pressure in hPa

    Returns
    -------
    float
        Atmospheric refractive index (unitless)

    Notes
    -----
    The function uses the standard formula for calculating the atmospheric
    refractive index, which includes terms for dry air and water vapor.
    """
    # Constants for refractive index calculation
    k1 = 77.604
    k2 = 70.4
    k3 = 373900

    # Calculate dry pressure and temperature in Kelvin
    pd = p - e
    tk = t + 273.15

    # Calculate refractive index
    N = (k1 * pd / tk + k2 * e / tk + k3 * e / tk**2) / 1e6
    return N


def gps2dmjd(gpst: float) -> float:
    """
    Convert GPS time to Modified Julian Date.

    This function converts GPS time (seconds since GPS epoch) to Modified
    Julian Date (MJD).

    Parameters
    ----------
    gpst : float
        GPS time in seconds since GPS epoch

    Returns
    -------
    float
        Modified Julian Date

    Notes
    -----
    The function accounts for the GPS epoch (January 6, 1980) and leap seconds
    to convert to MJD. The reference point is MJD 59215 on January 1, 2021.
    """
    # Convert GPS time to MJD
    # dmjd is 59215 on 1 Jan 2021
    # gpstime is 1293494418 (seconds)
    # that's 14971 days and 18 (leap) seconds since GPS time started
    # so, to get from gpstime to mjd
    # dmjd = (gpst - 18) / 86400 - 14971 + 59215 = (gpst - 18) / 86400 + 44244
    dmjd = (gpst - 18) / 86400 + 44244
    return dmjd
