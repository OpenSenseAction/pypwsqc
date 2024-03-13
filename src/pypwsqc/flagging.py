"""A collection of functions for flagging problematic time steps."""


from __future__ import annotations

import numpy as np
import numpy.typing as npt
import xarray as xr


def FZ_filter(
    pws_data: npt.NDArray[np.float_], reference: npt.NDArray[np.float_], nint: int = 6
) -> npt.NDArray[np.float_]:
    """Flag faulty zeros based on a reference time series.

    This function applies the FZ filter from the R package PWSQC.
    The flag 1 means, a faulty zero has been detected. The flag -1
    means that no flagging was done because evaluation cannot be
    performed for the first `nint` values.

    Note that this code here is derived from the Python translation,
    done by Niek van Andel, of the original R code from Lotte de Vos.
    The Python code stems from here https://github.com/NiekvanAndel/QC_radar.
    Also note that the correctness of the Python code has not been
    verified and not all feature of the R implementation might be there.

    Parameters
    ----------
    pws_data
        The rainfall time series of the PWS that should be flagged
    reference
        The rainfall time series of the reference, which can be e.g.
        the median of neighboring PWS data.
    nint : optional
        The number of subsequent data points which have to be zero, while
        the reference has values larger than zero, to set the flag for
        this data point to 1.

    Returns
    -------
    npt.NDArray
        time series of flags
    """
    ref_array = np.zeros(np.shape(pws_data))
    ref_array[np.where(reference > 0)] = 1

    sensor_array = np.zeros(np.shape(pws_data))
    sensor_array[np.where(pws_data > 0)] = 1
    sensor_array[np.where(pws_data == 0)] = 0

    fz_array = np.ones(np.shape(pws_data), dtype=np.float_) * -1

    for i in np.arange(nint, np.shape(pws_data)[0]):
        if sensor_array[i] > 0:
            fz_array[i] = 0
        elif fz_array[i - 1] == 1:
            fz_array[i] = 1
        # TODO: check why `< nint + 1` is used here.
        #       should `nint`` be scaled with a selectable factor?
        elif (np.sum(sensor_array[i - nint : i + 1]) > 0) or (
            np.sum(ref_array[i - nint : i + 1]) < nint + 1
        ):
            fz_array[i] = 0
        else:
            fz_array[i] = 1

    return fz_array


def HI_filter(
    pws_data: npt.NDArray[np.float_],
    nbrs_not_nan: npt.NDArray[np.float_],
    reference: npt.NDArray[np.float_],
    HIthresA: npt.NDArray[np.float_],
    HIthresB: npt.NDArray[np.float_],
    nstat=npt.NDArray[np.float_],
) -> npt.NDArray[np.float_]:
    """High Influx filter

    This function applies the HI filter from the R package PWSQC,
    flagging unrealistically high rainfall amounts.

    https://github.com/LottedeVos/PWSQC/tree/master/R

    The function returns an array with [...]

    Parameters
    ----------
    pws_data
        The rainfall time series of the PWS that should be flagged
    distance_matrix
        The rainfall time series of the PWS that should be flagged
    reference
        The rainfall time series of the reference, which can be e.g.
        the median of neighboring PWS data within a specified range d
    HithresA
        threshold for median rainfall of stations within range d [mm]
    HithresB
        upper rainfall limit [mm]

    Returns
    -------
    npt.NDArray
        time series of flags
    """
    HI_array = xr.where(nbrs_not_nan < nstat, -1, 0)
    condition1 = HI_array != -1
    condition2 = (reference < HIthresA) & (pws_data > HIthresB)
    condition3 = (reference >= HIthresA) & (pws_data > reference * HIthresB / HIthresA)
    HI_array = xr.where(condition1 & (condition2 | condition3), 1, 0)
    print("app")

    return HI_array
