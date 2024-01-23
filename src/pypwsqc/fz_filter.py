from __future__ import annotations

import numpy as np


def FZ_filter(pws_data, reference, nint=6):
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
        else:
            if Sensor_array[i] > 0:
                FZ_array[i] = 0
            else:
                if (
                    FZ_array[i - 1] == 1
                ):  # als er geen regen valt en vorige meting ook al FZ, dan wederom afvlaggen
                    FZ_array[i] = 1
                elif (
                    np.sum(Sensor_array[i - nint : i + 1]) > 0
                ):  # als er geen regen valt, maar ergens in het voorgaande wel: neit afvlaggen
                    FZ_array[i] = 0
                else:  # als er een serie nullen is van nint lengte
                    if (
                        np.sum(Ref_array[i - nint : i + 1]) < nint + 1
                    ):  # als de radar ook minstens af en toe nul of aangeeft: dan is alles prima, of is bepaald zonder nstat stations
                        FZ_array[i] = 0
                    else:  # dus als de sensor nul aangeeft, maar de radar niet nul (continu) in die periode
                        FZ_array[i] = 1

    return FZ_array
