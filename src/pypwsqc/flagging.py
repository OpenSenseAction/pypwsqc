"""A collection of functions for flagging problematic time steps."""


from __future__ import annotations

import numpy as np
import numpy.typing as npt


def fz_filter(
    pws_data: npt.NDArray, reference: npt.NDArray, nint: int = 6
) -> npt.NDArray:
    """Flag faulty zeros based on a reference time series.

    This function applies the FZ filter from the R package PWSQC.
    The flag 1 means, a faulty zero has been detected. The flag -1
    means that no flagging was done because evaluation cannot be
    performed for the first `nint` values.

    Parameters
    ----------
    pws_data
        The rainfall time series of the PWS that should be flagged
    reference
        The rainfall time series of the reference, which can be e.g.
        the mediah of neighboring PWS data.
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

    fz_array = np.ones(np.shape(pws_data)) * -1

    for i in np.arange(nint, np.shape(pws_data)[0]):
        if sensor_array[i] > 0:
            fz_array[i] = 0
        elif fz_array[i - 1] == 1:
            fz_array[i] = 1
        elif (np.sum(sensor_array[i - nint : i + 1]) > 0) or (
            np.sum(ref_array[i - nint : i + 1]) < nint + 1
        ):
            fz_array[i] = 0
        else:
            fz_array[i] = 1

    return fz_array
