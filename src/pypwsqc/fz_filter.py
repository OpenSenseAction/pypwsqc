from __future__ import annotations

import numpy as np
import numpy.typing as npt


def FZ_filter(
    pws_data: npt.ArrayLike, reference: npt.ArrayLike, nint: int = 6
) -> npt.ArrayLike:
    """Some doc string"""
    Ref_array = np.zeros(np.shape(pws_data))
    Ref_array[np.where(reference > 0)] = 1

    Sensor_array = np.zeros(np.shape(pws_data))
    Sensor_array[np.where(pws_data > 0)] = 1
    Sensor_array[np.where(pws_data == 0)] = 0

    FZ_array = np.ones(np.shape(pws_data)) * -1

    for i in np.arange(nint, np.shape(pws_data)[0]):
        # print(i)
        if len(np.ma.compressed(Ref_array[i])) == 0:
            FZ_array[i] = -1
        elif len(np.ma.compressed(Sensor_array[i])) == 0:
            FZ_array[i] = FZ_array[i - 1]
        elif Sensor_array[i] > 0:
            FZ_array[i] = 0
        elif FZ_array[i - 1] == 1:
            FZ_array[i] = 1
        elif (np.sum(Sensor_array[i - nint : i + 1]) > 0) or (
            np.sum(Ref_array[i - nint : i + 1]) < nint + 1
        ):
            FZ_array[i] = 0
        else:
            FZ_array[i] = 1

    return FZ_array
