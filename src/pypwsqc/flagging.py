from __future__ import annotations

import numpy as np
import numpy.typing as npt


def fz_filter(
    pws_data: npt.NDArray, reference: npt.NDArray, nint: int = 6
) -> npt.NDArray:
    """Some doc string"""
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
