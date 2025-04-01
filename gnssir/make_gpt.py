"""
GPT (Global Pressure and Temperature) model implementation.

This module provides functions for calculating atmospheric parameters using the GPT model.
It is based on the original TU Vienna codes for GMF (Global Mapping Function) and has been
adapted for use in GNSS-IR applications.

Edited for: https://ieeexplore.ieee.org/document/10078314
"""

import os
import pickle
from typing import Tuple

import numpy as np


def makegptfile(gptfile: str, gpt_init_file: str, site_lat: float, site_lon: float) -> None:
    """
    Create a station-specific refraction correction grid file.

    This function generates a grid file containing atmospheric parameters for a specific
    station location. It uses bilinear interpolation to calculate values between grid points.

    Parameters
    ----------
    gptfile : str
        Path to output file for station-specific refraction data
    gpt_init_file : str
        Path to input file containing the global GPT grid data
    site_lat : float
        Station latitude in degrees (NOT radians)
    site_lon : float
        Station longitude in degrees (NOT radians)

    Notes
    -----
    The function reads a pre-computed global grid from a pickle file and interpolates
    values for the specific station location. The output file contains atmospheric
    parameters needed for refraction correction.
    """
    print(
        "A station specific refraction output file will be written to ",
        gptfile,
    )

    # Convert to radians
    dlat = site_lat * np.pi / 180
    dlon = site_lon * np.pi / 180

    # Read VMF gridfile in pickle format
    if os.path.isfile(gpt_init_file):
        f = open(gpt_init_file, "rb")
        [
            All_pgrid,
            All_Tgrid,
            All_Qgrid,
            All_dTgrid,
            All_U,
            All_Hs,
            All_ahgrid,
            All_awgrid,
            All_lagrid,
            All_Tmgrid,
        ] = pickle.load(f)
        f.close()

    # Initialize indices for grid points
    indx = np.zeros(4, dtype=int)
    indx_lat = np.zeros(4, dtype=int)
    indx_lon = np.zeros(4, dtype=int)

    # Convert longitude to positive degrees
    if dlon < 0:
        plon = (dlon + 2 * np.pi) * 180 / np.pi
    else:
        plon = dlon * 180 / np.pi

    # Transform to polar distance in degrees
    ppod = (-dlat + np.pi / 2) * 180 / np.pi

    # Find nearest grid point indices
    ipod = np.floor(ppod + 1)
    ilon = np.floor(plon + 1)

    # Calculate normalized differences
    diffpod = ppod - (ipod - 0.5)
    difflon = plon - (ilon - 0.5)

    # Handle edge cases
    if ipod == 181:
        ipod = 180
    if ilon == 361:
        ilon = 1
    if ilon == 0:
        ilon = 360

    # Calculate grid indices
    indx[0] = (ipod - 1) * 360 + ilon
    indx_lat[0] = 90 - ipod + 1
    indx_lon[0] = ilon - 1

    # Determine if bilinear interpolation should be used
    if (ppod > 0.5) and (ppod < 179.5):
        # Calculate indices for bilinear interpolation
        ipod1 = ipod + np.sign(diffpod)
        ilon1 = ilon + np.sign(difflon)

        # Handle longitude wrap-around
        if ilon1 == 361:
            ilon1 = 1
        if ilon1 == 0:
            ilon1 = 360

        # Calculate remaining grid indices
        indx[1] = (ipod1 - 1) * 360 + ilon
        indx[2] = (ipod - 1) * 360 + ilon1
        indx[3] = (ipod1 - 1) * 360 + ilon1

        # Calculate lat/lon for all grid points
        indx_lat[1] = 90 - ipod1 + np.sign(diffpod)
        indx_lon[1] = ilon - 1
        indx_lat[2] = 90 - ipod + 1
        indx_lon[2] = ilon1 - np.sign(difflon)
        indx_lat[3] = 90 - ipod1 + np.sign(diffpod)
        indx_lon[3] = ilon1 - np.sign(difflon)

    # Adjust indices for Python's 0-based indexing
    indx = indx - 1
    indx_list = indx.tolist()

    # Write station-specific data to file
    fout = open(gptfile, "w+")
    for i, a in enumerate(indx_list):
        for k in [0, 1, 2, 3, 4]:
            fout.write(
                " {0:4.0f} {1:5.0f} {2:13.4f} {3:10.4f} {4:10.6f} "
                "{5:10.4f} {6:12.5f} {7:12.5f} {8:10.6f} {9:10.6f} "
                "{10:10.6f} {11:10.4f} \n".format(
                    indx_lat[i],
                    indx_lon[i],
                    All_pgrid[a, k],
                    All_Tgrid[a, k],
                    All_Qgrid[a, k] * 1000,
                    All_dTgrid[a, k] * 1000,
                    All_U[a, 0],
                    All_Hs[a, 0],
                    All_ahgrid[a, k] * 1000,
                    All_awgrid[a, k] * 1000,
                    All_lagrid[a, k],
                    All_Tmgrid[a, k],
                )
            )
    fout.close()
    print("station specific refraction file written")


def gpt2_1w(
    gptfile: str, dmjd: float, dlat: float, dlon: float, hell: float, it: int
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """
    Calculate atmospheric parameters using the GPT2 model.

    This function computes various atmospheric parameters for a given location and time
    using the GPT2 model. It supports both static and time-varying calculations.

    Parameters
    ----------
    gptfile : str
        Path to the station-specific GPT grid file
    dmjd : float
        Modified Julian date (scalar, only one epoch per call)
    dlat : float
        Ellipsoidal latitude in radians [-pi/2:+pi/2]
    dlon : float
        Longitude in radians [-pi:pi] or [0:2pi]
    hell : float
        Ellipsoidal height in meters
    it : int
        Time variation flag:
        - 1: no time variation (static quantities)
        - 0: with time variation (annual and semiannual terms)

    Returns
    -------
    Tuple[float, float, float, float, float, float, float, float, float]
        Tuple containing:
        - p: pressure in hPa
        - T: temperature in degrees Celsius
        - dT: temperature lapse rate in degrees per km
        - Tm: mean temperature of water vapor in degrees Kelvin
        - e: water vapor pressure in hPa
        - ah: hydrostatic mapping function coefficient at zero height (VMF1)
        - aw: wet mapping function coefficient (VMF1)
        - la: water vapor decrease factor
        - undu: geoid undulation in meters
    """
    # Convert longitude to positive degrees
    if dlon < 0:
        plon = (dlon + 2 * np.pi) * 180 / np.pi
    else:
        plon = dlon * 180 / np.pi

    # Transform to polar distance in degrees
    ppod = (-dlat + np.pi / 2) * 180 / np.pi

    # Find nearest grid point indices
    ipod = np.floor(ppod + 1)
    ilon = np.floor(plon + 1)

    # Calculate normalized differences
    diffpod = ppod - (ipod - 0.5)
    difflon = plon - (ilon - 0.5)

    # Change reference epoch to January 1 2000
    dmjd1 = dmjd - 51544.5

    # Constants
    pi2 = 2 * np.pi
    pi4 = 4 * np.pi
    gm = 9.80665  # mean gravity in m/s**2
    dMtr = 28.965e-3  # molar mass of dry air in kg/mol
    Rg = 8.3143  # universal gas constant in J/K/mol

    # Calculate time variation factors
    if it == 1:
        cosfy = 0
        coshy = 0
        sinfy = 0
        sinhy = 0
    else:
        cosfy = np.cos(pi2 * dmjd1 / 365.25)
        coshy = np.cos(pi4 * dmjd1 / 365.25)
        sinfy = np.sin(pi2 * dmjd1 / 365.25)
        sinhy = np.sin(pi4 * dmjd1 / 365.25)

    # Initialize output arrays
    p = 0
    T = 0
    dT = 0
    Tm = 0
    e = 0
    ah = 0
    aw = 0
    la = 0
    undu = 0
    undul = np.zeros(4)
    Ql = np.zeros(4)
    dTl = np.zeros(4)
    Tl = np.zeros(4)
    pl = np.zeros(4)
    ahl = np.zeros(4)
    awl = np.zeros(4)
    lal = np.zeros(4)
    Tml = np.zeros(4)
    el = np.zeros(4)

    # Read grid data
    (
        pgrid,
        Tgrid,
        Qgrid,
        dTgrid,
        u,
        Hs,
        ahgrid,
        awgrid,
        lagrid,
        Tmgrid,
    ) = read_4by5(gptfile)

    # Calculate parameters for each grid point
    for ll in [0, 1, 2, 3]:
        KL = ll
        # Transform ellipsoidal height to orthometric height
        undul[ll] = u[KL]
        hgt = hell - undul[ll]

        # Calculate temperature at grid height
        T0 = (
            Tgrid[KL, 0]
            + Tgrid[KL, 1] * cosfy
            + Tgrid[KL, 2] * sinfy
            + Tgrid[KL, 3] * coshy
            + Tgrid[KL, 4] * sinhy
        )

        # Calculate pressure at grid height
        p0 = (
            pgrid[KL, 0]
            + pgrid[KL, 1] * cosfy
            + pgrid[KL, 2] * sinfy
            + pgrid[KL, 3] * coshy
            + pgrid[KL, 4] * sinhy
        )

        # Calculate humidity
        Ql[ll] = (
            Qgrid[KL, 0]
            + Qgrid[KL, 1] * cosfy
            + Qgrid[KL, 2] * sinfy
            + Qgrid[KL, 3] * coshy
            + Qgrid[KL, 4] * sinhy
        )

        # Calculate height reduction
        Hs1 = Hs[KL]
        redh = hgt - Hs1

        # Calculate temperature lapse rate
        dTl[ll] = (
            dTgrid[KL, 0]
            + dTgrid[KL, 1] * cosfy
            + dTgrid[KL, 2] * sinfy
            + dTgrid[KL, 3] * coshy
            + dTgrid[KL, 4] * sinhy
        )

        # Calculate temperature at station height
        Tl[ll] = T0 + dTl[ll] * redh - 273.15

        # Calculate virtual temperature
        Tv = T0 * (1 + 0.6077 * Ql[ll])
        c = gm * dMtr / (Rg * Tv)

        # Calculate pressure at station height
        pl[ll] = (p0 * np.exp(-c * redh)) / 100

        # Calculate hydrostatic coefficient
        ahl[ll] = (
            ahgrid[KL, 0]
            + ahgrid[KL, 1] * cosfy
            + ahgrid[KL, 2] * sinfy
            + ahgrid[KL, 3] * coshy
            + ahgrid[KL, 4] * sinhy
        )

        # Calculate wet coefficient
        awl[ll] = (
            awgrid[KL, 0]
            + awgrid[KL, 1] * cosfy
            + awgrid[KL, 2] * sinfy
            + awgrid[KL, 3] * coshy
            + awgrid[KL, 4] * sinhy
        )

        # Calculate water vapor decrease factor
        lal[ll] = (
            lagrid[KL, 0]
            + lagrid[KL, 1] * cosfy
            + lagrid[KL, 2] * sinfy
            + lagrid[KL, 3] * coshy
            + lagrid[KL, 4] * sinhy
        )

        # Calculate mean temperature of water vapor
        Tml[ll] = (
            Tmgrid[KL, 0]
            + Tmgrid[KL, 1] * cosfy
            + Tmgrid[KL, 2] * sinfy
            + Tmgrid[KL, 3] * coshy
            + Tmgrid[KL, 4] * sinhy
        )

        # Calculate water vapor pressure
        e0 = Ql[ll] * p0 / (0.622 + 0.378 * Ql[ll]) / 100
        aa = 100 * pl[ll] / p0
        bb = lal[ll] + 1
        el[ll] = e0 * np.power(aa, bb)

    # Calculate interpolation weights
    dnpod1 = np.abs(diffpod)
    dnpod2 = 1 - dnpod1
    dnlon1 = np.abs(difflon)
    dnlon2 = 1 - dnlon1

    # Interpolate final values
    R1 = dnpod2 * pl[0] + dnpod1 * pl[1]
    R2 = dnpod2 * pl[2] + dnpod1 * pl[3]
    p = dnlon2 * R1 + dnlon1 * R2

    R1 = dnpod2 * Tl[0] + dnpod1 * Tl[1]
    R2 = dnpod2 * Tl[2] + dnpod1 * Tl[3]
    T = dnlon2 * R1 + dnlon1 * R2

    R1 = dnpod2 * dTl[0] + dnpod1 * dTl[1]
    R2 = dnpod2 * dTl[2] + dnpod1 * dTl[3]
    dT = (dnlon2 * R1 + dnlon1 * R2) * 1000

    R1 = dnpod2 * el[0] + dnpod1 * el[1]
    R2 = dnpod2 * el[2] + dnpod1 * el[3]
    e = dnlon2 * R1 + dnlon1 * R2

    R1 = dnpod2 * ahl[0] + dnpod1 * ahl[1]
    R2 = dnpod2 * ahl[2] + dnpod1 * ahl[3]
    ah = dnlon2 * R1 + dnlon1 * R2

    R1 = dnpod2 * awl[0] + dnpod1 * awl[1]
    R2 = dnpod2 * awl[2] + dnpod1 * awl[3]
    aw = dnlon2 * R1 + dnlon1 * R2

    R1 = dnpod2 * undul[0] + dnpod1 * undul[1]
    R2 = dnpod2 * undul[2] + dnpod1 * undul[3]
    undu = dnlon2 * R1 + dnlon1 * R2

    R1 = dnpod2 * lal[0] + dnpod1 * lal[1]
    R2 = dnpod2 * lal[2] + dnpod1 * lal[3]
    la = dnlon2 * R1 + dnlon1 * R2

    R1 = dnpod2 * Tml[0] + dnpod1 * Tml[1]
    R2 = dnpod2 * Tml[2] + dnpod1 * Tml[3]
    Tm = dnlon2 * R1 + dnlon1 * R2

    return p, T, dT, Tm, e, ah, aw, la, undu


def read_4by5(gptfile: str) -> Tuple[np.ndarray, ...]:
    """
    Read GPT grid data from a file.

    This function reads atmospheric parameters from a GPT grid file and organizes them
    into arrays for further processing.

    Parameters
    ----------
    gptfile : str
        Path to the GPT grid file

    Returns
    -------
    Tuple[np.ndarray, ...]
        Tuple containing arrays for:
        - pgrid: pressure grid
        - Tgrid: temperature grid
        - Qgrid: specific humidity grid
        - dTgrid: temperature lapse rate grid
        - u: geoid undulation
        - Hs: orthometric height
        - ahgrid: hydrostatic mapping function coefficient grid
        - awgrid: wet mapping function coefficient grid
        - lagrid: water vapor decrease factor grid
        - Tmgrid: mean temperature of water vapor grid
    """
    x = np.genfromtxt(gptfile, comments="%")

    # Initialize arrays for all parameters
    pgrid = np.zeros((4, 5))
    Tgrid = np.zeros((4, 5))
    Qgrid = np.zeros((4, 5))
    dTgrid = np.zeros((4, 5))
    u = np.zeros((4, 1))
    Hs = np.zeros((4, 1))
    ahgrid = np.zeros((4, 5))
    awgrid = np.zeros((4, 5))
    lagrid = np.zeros((4, 5))
    Tmgrid = np.zeros((4, 5))

    # Read data for each grid point
    for n in [0, 1, 2, 3]:
        ij = 0
        u[n] = x[n * 5, 6]
        Hs[n] = x[n * 5, 7]
        for m in range(n * 5, n * 5 + 5):
            pgrid[n, ij] = x[m, 2]
            Tgrid[n, ij] = x[m, 3]
            Qgrid[n, ij] = x[m, 4] / 1000
            dTgrid[n, ij] = x[m, 5] / 1000
            ahgrid[n, ij] = x[m, 8] / 1000
            awgrid[n, ij] = x[m, 9] / 1000
            lagrid[n, ij] = x[m, 10]
            Tmgrid[n, ij] = x[m, 11]
            ij += 1

    return pgrid, Tgrid, Qgrid, dTgrid, u, Hs, ahgrid, awgrid, lagrid, Tmgrid
