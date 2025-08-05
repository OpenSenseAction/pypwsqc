from unittest.mock import patch

import matplotlib as mpl
import numpy as np
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import pypwsqc.peak_removal_filter_plots as prfp

mpl.use("Agg")


def create_test_ds():
    """
    Create a test dataset for testing purposes.
    """
    start_time = np.datetime64("2025-04-01T00:00:00", "ns")
    end_time = np.datetime64("2025-04-01T08:00:00", "ns")
    time = np.arange(start_time, end_time, np.timedelta64(5, "m"))

    rainfall = []

    data_test_station = np.array(
        [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            7.7,
            1.1,
            1.2,
            1.7,
            0.5,
            1.2,
            1.2,
            1.2,
            0.3,
            1.9,
            1.5,
            1.6,
            2.0,
            0.2,
            1.3,
            1.2,
            1.2,
            1.4,
            0.8,
            0.2,
            0.6,
            1.3,
            0.4,
            0.2,
            0.0,
            0.0,
            0.8,
            1.8,
            0.5,
            1.0,
            0.5,
            1.6,
            0.7,
            0.3,
            1.8,
            1.9,
            0.3,
            0.4,
            0.2,
            1.9,
            1.9,
            1.8,
            0.5,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            9.8,
            0.7,
            0.0,
            1.7,
            0.0,
            0.2,
            1.5,
            1.4,
            0.2,
            0.9,
            1.6,
            1.7,
            0.1,
            0.7,
            1.0,
            0.1,
            0.1,
            0.1,
            0.9,
            0.3,
            1.7,
            0.6,
            0.9,
            1.8,
            1.8,
            1.1,
            0.5,
            0.5,
            0.9,
            0.1,
            0.6,
            0.9,
            0.5,
            0.6,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            11.5,
        ]
    )

    data_test_neighbor_1 = np.array(
        [
            1.2,
            0.8,
            1.5,
            1.8,
            0.3,
            0.5,
            1.3,
            0.6,
            0.5,
            0.5,
            1.3,
            1.5,
            0.6,
            0.8,
            0.3,
            0.6,
            0.4,
            0.1,
            0.6,
            0.8,
            1.5,
            1.5,
            0.2,
            1.6,
            1.4,
            1.8,
            0.4,
            1.1,
            0.4,
            1.1,
            1.8,
            0.3,
            0.4,
            1.6,
            0.8,
            0.3,
            0.1,
            2.0,
            1.1,
            0.8,
            0.6,
            1.1,
            0.6,
            0.0,
            1.1,
            0.5,
            0.2,
            1.9,
            1.6,
            1.1,
            1.1,
            0.4,
            0.4,
            0.7,
            1.7,
            0.5,
            1.2,
            2.0,
            1.9,
            0.9,
            0.8,
            1.9,
            0.5,
            1.7,
            1.0,
            0.4,
            1.7,
            0.5,
            1.7,
            2.0,
            1.2,
            1.8,
            1.2,
            0.5,
            0.8,
            1.9,
            0.9,
            0.3,
            0.6,
            1.3,
            0.1,
            0.8,
            0.8,
            1.5,
            1.8,
            0.5,
            1.6,
            0.8,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    data_test_neighbor_2 = np.array(
        [
            0.6,
            1.1,
            0.8,
            1.2,
            1.0,
            1.8,
            0.9,
            1.7,
            0.4,
            1.7,
            1.1,
            0.3,
            1.3,
            0.5,
            1.4,
            1.0,
            1.2,
            0.1,
            1.7,
            0.9,
            0.2,
            0.8,
            0.5,
            0.5,
            1.5,
            1.6,
            0.5,
            1.2,
            1.7,
            0.8,
            1.2,
            1.6,
            1.7,
            1.4,
            0.7,
            0.3,
            1.7,
            0.9,
            1.4,
            1.5,
            0.7,
            0.1,
            1.5,
            0.2,
            0.5,
            1.6,
            0.7,
            0.8,
            1.2,
            0.5,
            1.3,
            1.7,
            0.9,
            1.0,
            1.1,
            1.6,
            0.7,
            1.4,
            1.0,
            1.3,
            0.1,
            0.9,
            0.2,
            1.1,
            1.9,
            1.4,
            1.0,
            1.2,
            1.0,
            1.4,
            1.7,
            0.5,
            1.2,
            1.3,
            1.8,
            1.5,
            0.2,
            1.8,
            1.2,
            1.2,
            0.8,
            1.8,
            1.0,
            1.4,
            0.8,
            0.5,
            1.6,
            1.1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    data_test_neighbor_3 = np.array(
        [
            1.6,
            1.3,
            0.2,
            0.7,
            0.6,
            1.6,
            0.5,
            1.8,
            1.4,
            0.9,
            1.8,
            2.0,
            1.4,
            0.0,
            1.6,
            1.9,
            0.7,
            0.3,
            1.6,
            0.7,
            0.6,
            0.8,
            0.4,
            1.2,
            0.3,
            0.5,
            1.5,
            0.5,
            2.0,
            1.5,
            0.8,
            1.6,
            1.2,
            1.5,
            1.6,
            1.4,
            0.4,
            1.8,
            0.5,
            1.7,
            0.3,
            1.2,
            0.7,
            1.1,
            0.8,
            0.5,
            1.0,
            1.4,
            0.1,
            0.7,
            0.4,
            0.0,
            1.5,
            1.0,
            0.8,
            1.8,
            2.0,
            0.4,
            1.0,
            1.1,
            1.2,
            0.7,
            1.4,
            0.2,
            1.3,
            0.4,
            1.1,
            1.4,
            1.1,
            0.6,
            1.6,
            1.2,
            1.3,
            0.7,
            0.1,
            0.1,
            2.0,
            2.0,
            1.8,
            1.8,
            1.4,
            1.2,
            1.1,
            1.2,
            1.1,
            1.5,
            1.5,
            1.7,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    data_test_neighbor_4 = np.array(
        [
            0.9,
            1.7,
            1.2,
            1.0,
            0.5,
            0.1,
            1.9,
            0.3,
            1.3,
            0.5,
            1.2,
            1.1,
            0.1,
            0.9,
            1.1,
            1.8,
            0.1,
            0.2,
            0.5,
            1.9,
            1.4,
            1.5,
            0.2,
            0.0,
            1.8,
            1.1,
            1.8,
            1.3,
            1.5,
            1.9,
            0.4,
            0.3,
            0.3,
            1.5,
            1.5,
            0.1,
            2.0,
            2.0,
            1.8,
            0.2,
            1.7,
            0.2,
            1.9,
            0.1,
            1.0,
            0.6,
            0.9,
            1.6,
            1.0,
            0.4,
            1.3,
            1.3,
            0.5,
            1.1,
            0.5,
            0.2,
            0.7,
            0.3,
            0.8,
            1.9,
            1.9,
            0.1,
            0.1,
            1.7,
            1.7,
            1.1,
            1.9,
            2.0,
            0.3,
            1.9,
            1.0,
            0.4,
            0.8,
            0.7,
            1.4,
            0.3,
            1.3,
            1.2,
            0.9,
            0.9,
            1.7,
            0.0,
            1.6,
            0.2,
            0.8,
            0.5,
            0.7,
            0.8,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    rainfall.append(data_test_station)
    rainfall.append(data_test_neighbor_1)
    rainfall.append(data_test_neighbor_2)
    rainfall.append(data_test_neighbor_3)
    rainfall.append(data_test_neighbor_4)

    rainfall_reshape = [list(x) for x in zip(*rainfall)]
    rainfall_reshape_ar = np.array(rainfall_reshape)

    index = [
        "test_station",
        "test_neighbor_1",
        "test_neighbor_2",
        "test_neighbor_3",
        "test_neighbor_4",
    ]
    x = np.array([1, 0, 2, 2, 0])
    x = x.astype(np.float64)
    y = np.array([1, 2, 2, 0, 0])
    y = y.astype(np.float64)

    return xr.Dataset(
        data_vars={
            "rainfall": (["id", "time"], rainfall_reshape_ar.T),
        },
        coords={
            "id": index,
            "time": time,
            "x": (["id"], x),
            "y": (["id"], y),
        },
    )


def create_test_ds_ref():
    """
    Create a test dataset for testing purposes.
    """
    start_time = np.datetime64("2025-04-01T01:00:00", "ns")
    end_time = np.datetime64("2025-04-01T09:00:00", "ns")
    time = np.arange(start_time, end_time, np.timedelta64(5, "m"))

    data_test_ref_neighbor_1 = np.array(
        [
            1.4,
            1.2,
            1.9,
            0.9,
            0.8,
            0.9,
            0.1,
            1.7,
            0.5,
            0.7,
            0.6,
            1.7,
            2.0,
            0.3,
            0.3,
            0.9,
            0.3,
            1.7,
            1.2,
            0.4,
            1.5,
            1.6,
            0.8,
            0.8,
            2.0,
            1.3,
            0.4,
            1.5,
            1.6,
            1.7,
            0.8,
            1.9,
            1.4,
            1.4,
            0.0,
            1.5,
            0.2,
            1.5,
            0.4,
            0.4,
            0.2,
            1.5,
            0.7,
            2.0,
            0.5,
            0.8,
            1.4,
            1.0,
            0.3,
            0.8,
            0.1,
            0.7,
            0.4,
            1.9,
            1.3,
            0.2,
            0.0,
            1.1,
            0.5,
            0.1,
            1.7,
            0.1,
            1.0,
            1.9,
            1.8,
            0.0,
            1.7,
            1.1,
            0.6,
            0.1,
            0.8,
            0.6,
            1.9,
            0.1,
            1.1,
            1.0,
            0.9,
            0.4,
            1.5,
            np.nan,
            1.7,
            0.6,
            1.2,
            0.9,
            0.7,
            0.9,
            1.8,
            1.8,
            1.3,
            1.7,
            0.4,
            2.0,
            1.5,
            0.7,
            0.4,
            0.3,
        ]
    )

    data_test_ref_neighbor_2 = np.array(
        [
            0.7,
            1.0,
            1.5,
            0.6,
            2.0,
            0.2,
            1.7,
            0.0,
            1.4,
            0.6,
            1.4,
            1.1,
            0.6,
            1.2,
            1.4,
            1.4,
            0.5,
            1.2,
            1.1,
            0.4,
            0.8,
            1.4,
            0.4,
            1.4,
            0.5,
            1.4,
            0.7,
            1.9,
            1.0,
            0.2,
            1.2,
            1.9,
            1.1,
            1.2,
            0.2,
            0.3,
            0.6,
            0.8,
            1.0,
            0.0,
            1.4,
            1.7,
            0.3,
            0.9,
            1.3,
            0.8,
            1.2,
            1.4,
            1.1,
            0.3,
            0.8,
            0.3,
            1.3,
            1.9,
            0.8,
            0.0,
            0.2,
            1.5,
            0.4,
            0.6,
            1.6,
            0.9,
            1.6,
            1.4,
            1.4,
            1.2,
            1.8,
            1.8,
            1.8,
            0.1,
            0.8,
            0.7,
            1.8,
            1.4,
            0.4,
            0.9,
            1.9,
            1.6,
            2.0,
            1.8,
            0.5,
            1.5,
            0.5,
            1.3,
            0.2,
            0.0,
            0.5,
            1.1,
            1.3,
            1.4,
            1.9,
            0.6,
            1.3,
            1.6,
            0.8,
            1.2,
        ]
    )

    data_test_ref_neighbor_3 = np.array(
        [
            0.7,
            1.0,
            1.5,
            0.6,
            2.0,
            0.2,
            1.7,
            0.0,
            1.4,
            0.6,
            1.4,
            1.1,
            0.6,
            1.2,
            1.4,
            1.4,
            0.5,
            1.2,
            1.1,
            0.4,
            0.8,
            1.4,
            0.4,
            1.4,
            0.5,
            1.4,
            0.7,
            1.9,
            1.0,
            0.2,
            1.2,
            1.9,
            1.1,
            1.2,
            0.2,
            0.3,
            0.6,
            0.8,
            1.0,
            0.0,
            1.4,
            1.7,
            0.3,
            0.9,
            1.3,
            0.8,
            1.2,
            1.4,
            1.1,
            0.3,
            0.8,
            0.3,
            1.3,
            1.9,
            0.8,
            0.0,
            0.2,
            1.5,
            0.4,
            0.6,
            1.6,
            0.9,
            1.6,
            1.4,
            1.4,
            1.2,
            1.8,
            1.8,
            1.8,
            0.1,
            0.8,
            0.7,
            1.8,
            1.4,
            0.4,
            0.9,
            1.9,
            1.6,
            2.0,
            1.8,
            0.5,
            1.5,
            0.5,
            1.3,
            0.2,
            0.0,
            0.5,
            1.1,
            1.3,
            1.4,
            1.9,
            0.6,
            1.3,
            1.6,
            0.8,
            1.2,
        ]
    )

    data_test_ref_neighbor_4 = np.full(len(time), np.nan)

    rainfall = []
    rainfall.append(data_test_ref_neighbor_1)
    rainfall.append(data_test_ref_neighbor_2)
    rainfall.append(data_test_ref_neighbor_3)
    rainfall.append(data_test_ref_neighbor_4)

    rainfall_reshape = [list(x) for x in zip(*rainfall)]
    rainfall_reshape_ar = np.array(rainfall_reshape)

    index = [
        "test_ref_neighbor_1",
        "test_ref_neighbor_2",
        "test_ref_neighbor_3",
        "test_ref_neighbor_4",
    ]
    x = np.array([0.5, 2, 2, 0])
    x = x.astype(np.float64)
    y = np.array([1.5, 2, 0, 0])
    y = y.astype(np.float64)

    return xr.Dataset(
        data_vars={
            "rainfall": (["id", "time"], rainfall_reshape_ar.T),
        },
        coords={
            "id": index,
            "time": time,
            "x": (["id"], x),
            "y": (["id"], y),
        },
    )


def create_test_ds_corr():
    """
    Create a test dataset for testing purposes.
    """
    start_time = np.datetime64("2025-04-01T00:00:00", "ns")
    end_time = np.datetime64("2025-04-01T08:00:00", "ns")
    time = np.arange(start_time, end_time, np.timedelta64(5, "m"))

    rainfall = []

    data_test_station = np.array(
        [
            1.37958333,
            1.57208333,
            1.18708333,
            1.50791667,
            0.77,
            1.28333333,
            1.1,
            1.2,
            1.7,
            0.5,
            1.2,
            1.2,
            1.2,
            0.3,
            1.9,
            1.5,
            1.6,
            2.0,
            0.2,
            1.3,
            1.2,
            1.2,
            1.4,
            0.8,
            0.2,
            0.6,
            1.3,
            0.4,
            0.2,
            0.0,
            0.0,
            0.8,
            1.8,
            0.5,
            1.0,
            0.5,
            1.6,
            0.7,
            0.3,
            1.8,
            1.9,
            0.3,
            0.4,
            0.2,
            1.9,
            1.9,
            1.8,
            0.5,
            1.51067194,
            1.0458498,
            1.58814229,
            1.31699605,
            1.27826087,
            1.47193676,
            1.58814229,
            0.7,
            0.0,
            1.7,
            0.0,
            0.2,
            1.5,
            1.4,
            0.2,
            0.9,
            1.6,
            1.7,
            0.1,
            0.7,
            1.0,
            0.1,
            0.1,
            0.1,
            0.9,
            0.3,
            1.7,
            0.6,
            0.9,
            1.8,
            1.8,
            1.1,
            0.5,
            0.5,
            0.9,
            0.1,
            0.6,
            0.9,
            0.5,
            0.6,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            11.5,
        ]
    )

    data_test_neighbor_1 = np.array(
        [
            1.2,
            0.8,
            1.5,
            1.8,
            0.3,
            0.5,
            1.3,
            0.6,
            0.5,
            0.5,
            1.3,
            1.5,
            0.6,
            0.8,
            0.3,
            0.6,
            0.4,
            0.1,
            0.6,
            0.8,
            1.5,
            1.5,
            0.2,
            1.6,
            1.4,
            1.8,
            0.4,
            1.1,
            0.4,
            1.1,
            1.8,
            0.3,
            0.4,
            1.6,
            0.8,
            0.3,
            0.1,
            2.0,
            1.1,
            0.8,
            0.6,
            1.1,
            0.6,
            0.0,
            1.1,
            0.5,
            0.2,
            1.9,
            1.6,
            1.1,
            1.1,
            0.4,
            0.4,
            0.7,
            1.7,
            0.5,
            1.2,
            2.0,
            1.9,
            0.9,
            0.8,
            1.9,
            0.5,
            1.7,
            1.0,
            0.4,
            1.7,
            0.5,
            1.7,
            2.0,
            1.2,
            1.8,
            1.2,
            0.5,
            0.8,
            1.9,
            0.9,
            0.3,
            0.6,
            1.3,
            0.1,
            0.8,
            0.8,
            1.5,
            1.8,
            0.5,
            1.6,
            0.8,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    data_test_neighbor_2 = np.array(
        [
            0.6,
            1.1,
            0.8,
            1.2,
            1.0,
            1.8,
            0.9,
            1.7,
            0.4,
            1.7,
            1.1,
            0.3,
            1.3,
            0.5,
            1.4,
            1.0,
            1.2,
            0.1,
            1.7,
            0.9,
            0.2,
            0.8,
            0.5,
            0.5,
            1.5,
            1.6,
            0.5,
            1.2,
            1.7,
            0.8,
            1.2,
            1.6,
            1.7,
            1.4,
            0.7,
            0.3,
            1.7,
            0.9,
            1.4,
            1.5,
            0.7,
            0.1,
            1.5,
            0.2,
            0.5,
            1.6,
            0.7,
            0.8,
            1.2,
            0.5,
            1.3,
            1.7,
            0.9,
            1.0,
            1.1,
            1.6,
            0.7,
            1.4,
            1.0,
            1.3,
            0.1,
            0.9,
            0.2,
            1.1,
            1.9,
            1.4,
            1.0,
            1.2,
            1.0,
            1.4,
            1.7,
            0.5,
            1.2,
            1.3,
            1.8,
            1.5,
            0.2,
            1.8,
            1.2,
            1.2,
            0.8,
            1.8,
            1.0,
            1.4,
            0.8,
            0.5,
            1.6,
            1.1,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    data_test_neighbor_3 = np.array(
        [
            1.6,
            1.3,
            0.2,
            0.7,
            0.6,
            1.6,
            0.5,
            1.8,
            1.4,
            0.9,
            1.8,
            2.0,
            1.4,
            0.0,
            1.6,
            1.9,
            0.7,
            0.3,
            1.6,
            0.7,
            0.6,
            0.8,
            0.4,
            1.2,
            0.3,
            0.5,
            1.5,
            0.5,
            2.0,
            1.5,
            0.8,
            1.6,
            1.2,
            1.5,
            1.6,
            1.4,
            0.4,
            1.8,
            0.5,
            1.7,
            0.3,
            1.2,
            0.7,
            1.1,
            0.8,
            0.5,
            1.0,
            1.4,
            0.1,
            0.7,
            0.4,
            0.0,
            1.5,
            1.0,
            0.8,
            1.8,
            2.0,
            0.4,
            1.0,
            1.1,
            1.2,
            0.7,
            1.4,
            0.2,
            1.3,
            0.4,
            1.1,
            1.4,
            1.1,
            0.6,
            1.6,
            1.2,
            1.3,
            0.7,
            0.1,
            0.1,
            2.0,
            2.0,
            1.8,
            1.8,
            1.4,
            1.2,
            1.1,
            1.2,
            1.1,
            1.5,
            1.5,
            1.7,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    data_test_neighbor_4 = np.array(
        [
            0.9,
            1.7,
            1.2,
            1.0,
            0.5,
            0.1,
            1.9,
            0.3,
            1.3,
            0.5,
            1.2,
            1.1,
            0.1,
            0.9,
            1.1,
            1.8,
            0.1,
            0.2,
            0.5,
            1.9,
            1.4,
            1.5,
            0.2,
            0.0,
            1.8,
            1.1,
            1.8,
            1.3,
            1.5,
            1.9,
            0.4,
            0.3,
            0.3,
            1.5,
            1.5,
            0.1,
            2.0,
            2.0,
            1.8,
            0.2,
            1.7,
            0.2,
            1.9,
            0.1,
            1.0,
            0.6,
            0.9,
            1.6,
            1.0,
            0.4,
            1.3,
            1.3,
            0.5,
            1.1,
            0.5,
            0.2,
            0.7,
            0.3,
            0.8,
            1.9,
            1.9,
            0.1,
            0.1,
            1.7,
            1.7,
            1.1,
            1.9,
            2.0,
            0.3,
            1.9,
            1.0,
            0.4,
            0.8,
            0.7,
            1.4,
            0.3,
            1.3,
            1.2,
            0.9,
            0.9,
            1.7,
            0.0,
            1.6,
            0.2,
            0.8,
            0.5,
            0.7,
            0.8,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )

    rainfall.append(data_test_station)
    rainfall.append(data_test_neighbor_1)
    rainfall.append(data_test_neighbor_2)
    rainfall.append(data_test_neighbor_3)
    rainfall.append(data_test_neighbor_4)

    rainfall_reshape = [list(x) for x in zip(*rainfall)]
    rainfall_reshape_ar = np.array(rainfall_reshape)

    index = [
        "test_station",
        "test_neighbor_1",
        "test_neighbor_2",
        "test_neighbor_3",
        "test_neighbor_4",
    ]
    x = np.array([1, 0, 2, 2, 0])
    x = x.astype(np.float64)
    y = np.array([1, 2, 2, 0, 0])
    y = y.astype(np.float64)

    return xr.Dataset(
        data_vars={
            "rainfall": (["id", "time"], rainfall_reshape_ar.T),
        },
        coords={
            "id": index,
            "time": time,
            "x": (["id"], x),
            "y": (["id"], y),
        },
    )


def create_test_closest_neighbors_ds_ds():
    distances = [
        [0.0, 1.41421356, 1.41421356, 1.41421356, 1.41421356],
        [0.0, 1.41421356, 2.0, 2.0, 2.82842712],
        [0.0, 1.41421356, 2.0, 2.0, 2.82842712],
        [0.0, 1.41421356, 2.0, 2.0, 2.82842712],
        [0.0, 1.41421356, 2.0, 2.0, 2.82842712],
    ]
    n_closest = [
        [
            "test_station",
            "test_neighbor_2",
            "test_neighbor_3",
            "test_neighbor_4",
            "test_neighbor_1",
        ],
        [
            "test_neighbor_1",
            "test_station",
            "test_neighbor_2",
            "test_neighbor_4",
            "test_neighbor_3",
        ],
        [
            "test_neighbor_2",
            "test_station",
            "test_neighbor_1",
            "test_neighbor_3",
            "test_neighbor_4",
        ],
        [
            "test_neighbor_3",
            "test_station",
            "test_neighbor_2",
            "test_neighbor_4",
            "test_neighbor_1",
        ],
        [
            "test_neighbor_4",
            "test_station",
            "test_neighbor_1",
            "test_neighbor_3",
            "test_neighbor_2",
        ],
    ]
    index = [
        "test_station",
        "test_neighbor_1",
        "test_neighbor_2",
        "test_neighbor_3",
        "test_neighbor_4",
    ]
    return xr.Dataset(
        data_vars={
            "distance": (["id", "n_closest"], distances),
            "neighbor_id": (["id", "n_closest"], n_closest),
        },
        coords={"id": index},
    )


def create_test_closest_neighbors_ds_ds_ref():
    distances = [
        [0.70710678, 1.41421356, 1.41421356, 1.41421356, np.inf],
        [0.70710678, 2.0, 2.0, 2.82842712, np.inf],
        [0.0, 1.58113883, 2.0, 2.82842712, np.inf],
        [0.0, 2.0, 2.0, 2.12132034, np.inf],
        [0.0, 1.58113883, 2.0, 2.82842712, np.inf],
    ]
    n_closest = [
        [
            "test_ref_neighbor_1",
            "test_ref_neighbor_3",
            "test_ref_neighbor_4",
            "test_ref_neighbor_2",
            None,
        ],
        [
            "test_ref_neighbor_1",
            "test_ref_neighbor_2",
            "test_ref_neighbor_4",
            "test_ref_neighbor_3",
            None,
        ],
        [
            "test_ref_neighbor_2",
            "test_ref_neighbor_1",
            "test_ref_neighbor_3",
            "test_ref_neighbor_4",
            None,
        ],
        [
            "test_ref_neighbor_3",
            "test_ref_neighbor_2",
            "test_ref_neighbor_4",
            "test_ref_neighbor_1",
            None,
        ],
        [
            "test_ref_neighbor_4",
            "test_ref_neighbor_1",
            "test_ref_neighbor_3",
            "test_ref_neighbor_2",
            None,
        ],
    ]
    index = [
        "test_station",
        "test_neighbor_1",
        "test_neighbor_2",
        "test_neighbor_3",
        "test_neighbor_4",
    ]
    return xr.Dataset(
        data_vars={
            "distance": (["id", "n_closest"], distances),
            "neighbor_id": (["id", "n_closest"], n_closest),
        },
        coords={"id": index},
    )


def create_test_closest_neighbors_no_neighbors():
    distances = np.full((5, 5), np.inf)
    n_closest = np.full((5, 5), None)
    index = [
        "test_station",
        "test_neighbor_1",
        "test_neighbor_2",
        "test_neighbor_3",
        "test_neighbor_4",
    ]
    return xr.Dataset(
        data_vars={
            "distance": (["id", "n_closest"], distances),
            "neighbor_id": (["id", "n_closest"], n_closest),
        },
        coords={"id": index},
    )


def create_test_weights_da():
    weights = np.array(
        [
            [0.0, 0.25, 0.25, 0.25, 0.25],
            [0.0, 0.44444444, 0.22222222, 0.22222222, 0.11111111],
            [0.0, 0.44444444, 0.22222222, 0.22222222, 0.11111111],
            [0.0, 0.44444444, 0.22222222, 0.22222222, 0.11111111],
            [0.0, 0.44444444, 0.22222222, 0.22222222, 0.11111111],
        ]
    )
    index = [
        "test_station",
        "test_neighbor_1",
        "test_neighbor_2",
        "test_neighbor_3",
        "test_neighbor_4",
    ]
    return xr.DataArray(
        weights, dims=["id", "weights"], coords={"id": index}, name="weights"
    )


def create_test_weights_da_ref():
    weights = np.array(
        [
            [0.57142857, 0.14285714, 0.14285714, 0.14285714, 0.0],
            [0.76190476, 0.0952381, 0.0952381, 0.04761905, 0.0],
            [0.0, 0.51612903, 0.32258065, 0.16129032, 0.0],
            [0.0, 0.34615385, 0.34615385, 0.30769231, 0.0],
            [0.0, 0.51612903, 0.32258065, 0.16129032, 0.0],
        ]
    )
    index = [
        "test_station",
        "test_neighbor_1",
        "test_neighbor_2",
        "test_neighbor_3",
        "test_neighbor_4",
    ]
    return xr.DataArray(
        weights, dims=["id", "weights"], coords={"id": index}, name="weights"
    )


def test_plot_station_neighbors():
    # Arrange (setup)
    test_ds = create_test_ds()
    test_ds_ref = create_test_ds_ref()
    aa_closest_neighbors = create_test_closest_neighbors_ds_ds()
    ab_closest_neighbors = create_test_closest_neighbors_ds_ds_ref()

    # Act (execute)
    with patch("matplotlib.pyplot.show"):
        fig, ax = prfp.plot_station_neighbors(
            test_ds,
            test_ds_ref,
            "test_station",
            aa_closest_neighbors,
            ab_closest_neighbors,
            2,
            True,
        )
    figure_bool = isinstance(fig, Figure)
    axes_bool = isinstance(ax, Axes)

    figsize = fig.get_size_inches()

    title = fig.axes[0].get_title()

    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    scatter = ax.collections[0]

    facecolor = scatter.get_facecolor()
    edgecolor = scatter.get_edgecolor()
    sizes = scatter.get_sizes()
    alpha = scatter.get_alpha()

    # Assert (check)
    assert figure_bool is True, "Figure is not a Figure object"
    assert axes_bool is True, "Axes is not an Axes object"
    assert figsize[0] == np.array([6.4, 4.8])[0], "Figure size is not correct"
    assert title == "Neighbors of station test_station", "Title is not correct"
    assert xlabel == "longitude", "X label is not correct"
    assert ylabel == "latitude", "Y label is not correct"
    assert xlim == (np.float64(-5.0), np.float64(7.0)), "Xlim is not correct"
    assert ylim == (np.float64(-5.0), np.float64(7.0)), "Ylim is not correct"
    assert (
        facecolor[0][0] == np.array([0.0, 0.0, 1.0, 0.5])[0]
    ), "Facecolor is not correct"
    assert (
        edgecolor[0][0] == np.array([0.0, 0.0, 1.0, 0.5])[0]
    ), "Edgecolor is not correct"
    assert sizes[0] == 10, "Size is not correct"
    assert alpha == 0.5, "Alpha is not correct"


def test_plot_station_neighbors_ab_closest_neighbors_is_none():
    # Arrange (setup)
    test_ds = create_test_ds()
    test_ds_ref = None
    aa_closest_neighbors = create_test_closest_neighbors_ds_ds()
    ab_closest_neighbors = None

    # Act (execute)
    with patch("matplotlib.pyplot.show"):
        fig, ax = prfp.plot_station_neighbors(
            test_ds,
            test_ds_ref,
            "test_station",
            aa_closest_neighbors,
            ab_closest_neighbors,
            2,
            True,
        )
    figure_bool = isinstance(fig, Figure)
    axes_bool = isinstance(ax, Axes)

    figsize = fig.get_size_inches()

    title = fig.axes[0].get_title()

    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    scatter = ax.collections[0]

    facecolor = scatter.get_facecolor()
    edgecolor = scatter.get_edgecolor()
    sizes = scatter.get_sizes()
    alpha = scatter.get_alpha()

    # Assert (check)
    assert figure_bool is True, "Figure is not a Figure object"
    assert axes_bool is True, "Axes is not an Axes object"
    assert figsize[0] == np.array([6.4, 4.8])[0], "Figure size is not correct"
    assert title == "Neighbors of station test_station", "Title is not correct"
    assert xlabel == "longitude", "X label is not correct"
    assert ylabel == "latitude", "Y label is not correct"
    assert xlim == (np.float64(-5.0), np.float64(7.0)), "Xlim is not correct"
    assert ylim == (np.float64(-5.0), np.float64(7.0)), "Ylim is not correct"
    assert (
        facecolor[0][0] == np.array([0.0, 0.0, 1.0, 0.5])[0]
    ), "Facecolor is not correct"
    assert (
        edgecolor[0][0] == np.array([0.0, 0.0, 1.0, 0.5])[0]
    ), "Edgecolor is not correct"
    assert sizes[0] == 10, "Size is not correct"
    assert alpha == 0.5, "Alpha is not correct"


def test_plot_station_neighbors_neighbor_is_None_and_zoom_false():
    # Arrange (setup)
    test_ds = create_test_ds()
    test_ds_ref = create_test_ds_ref()
    aa_closest_neighbors = create_test_closest_neighbors_no_neighbors()
    ab_closest_neighbors = create_test_closest_neighbors_ds_ds_ref()

    # Act (execute)
    with patch("matplotlib.pyplot.show"):
        fig, ax = prfp.plot_station_neighbors(
            test_ds,
            test_ds_ref,
            "test_station",
            aa_closest_neighbors,
            ab_closest_neighbors,
            2,
            False,
        )
    figure_bool = isinstance(fig, Figure)
    axes_bool = isinstance(ax, Axes)

    figsize = fig.get_size_inches()

    title = fig.axes[0].get_title()

    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    scatter = ax.collections[0]

    facecolor = scatter.get_facecolor()
    edgecolor = scatter.get_edgecolor()
    sizes = scatter.get_sizes()
    alpha = scatter.get_alpha()

    # Assert (check)
    assert figure_bool is True, "Figure is not a Figure object"
    assert axes_bool is True, "Axes is not an Axes object"
    assert figsize[0] == np.array([6.4, 4.8])[0], "Figure size is not correct"
    assert title == "Neighbors of station test_station", "Title is not correct"
    assert xlabel == "longitude", "X label is not correct"
    assert ylabel == "latitude", "Y label is not correct"
    assert xlim == (np.float64(-1.2), np.float64(3.2)), "Xlim is not correct"
    assert ylim == (np.float64(-1.2), np.float64(3.2)), "Ylim is not correct"
    assert (
        facecolor[0][0] == np.array([0.0, 0.0, 1.0, 0.5])[0]
    ), "Facecolor is not correct"
    assert (
        edgecolor[0][0] == np.array([0.0, 0.0, 1.0, 0.5])[0]
    ), "Edgecolor is not correct"
    assert sizes[0] == 2, "Size is not correct"
    assert alpha == 0.5, "Alpha is not correct"


def test_plot_peak():
    # Arrange (setup)
    test_ds = create_test_ds()
    test_ds_corr = create_test_ds_corr()
    seq_start_lst = [
        np.datetime64("2025-04-01T00:00:00.000000000"),
        np.datetime64("2025-04-01T04:00:00.000000000"),
        np.datetime64("2025-04-01T07:20:00.000000000"),
    ]
    time_peak_lst = [
        np.datetime64("2025-04-01T00:25:00.000000000"),
        np.datetime64("2025-04-01T04:30:00.000000000"),
        np.datetime64("2025-04-01T07:55:00.000000000"),
    ]
    seq_end_lst = [
        np.datetime64("2025-04-01T00:20:00.000000000"),
        np.datetime64("2025-04-01T04:25:00.000000000"),
        np.datetime64("2025-04-01T07:50:00.000000000"),
    ]
    seq_len_lst = [5, 6, 7]
    index = 1

    # Act (execute)
    with patch("matplotlib.pyplot.show"):
        fig, ax = prfp.plot_peak(
            test_ds,
            test_ds_corr,
            "test_station",
            0.95,
            1,
            seq_start_lst,
            time_peak_lst,
            seq_end_lst,
            seq_len_lst,
            True,
        )
    figure_bool = isinstance(fig, Figure)
    axes_bool = isinstance(ax, Axes)

    figsize = fig.get_size_inches()

    title = fig.axes[0].get_title()
    lines = fig.axes[0].get_lines()

    marker = lines[index].get_marker()
    markersize = lines[index].get_markersize()
    color = lines[index].get_color()
    alpha = lines[index].get_alpha()

    line_data = lines[index].get_data()

    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Assert (check)
    assert figure_bool is True, "Figure is not a Figure object"
    assert axes_bool is True, "Axes is not an Axes object"
    assert figsize[0] == np.array([6.4, 4.8])[0], "Figure size is not correct"
    assert (
        title == "Corrected rainfall of station test_station with peak no. 2 out of 3"
    ), "Title is not correct"
    assert marker == "None", "Marker is not correct"
    assert markersize == 6.0, "Marker size is not correct"
    assert color == "#1f77b4", "Color is not correct"
    assert alpha is None, "Alpha is not correct"
    assert (
        line_data[0][0]
        == (np.array([20179.09375, 20179.26041667]), np.array([0, 0]))[0][0]
    ), "Line data is not correct"
    assert xlabel == "time", "X label is not correct"
    assert ylabel == "rainfall [mm]", "Y label is not correct"
    assert xlim == (
        np.float64(20179.09375),
        np.float64(20179.260416666668),
    ), "Xlim is not correct"
    assert ylim == (
        np.float64(-0.095),
        np.float64(1.9949999999999999),
    ), "Ylim is not correct"


def test_plot_peak_time_not_in_zoom_range_at_start():
    # Arrange (setup)
    test_ds = create_test_ds()
    test_ds_corr = create_test_ds_corr()
    seq_start_lst = [
        np.datetime64("2025-04-01T00:00:00.000000000"),
        np.datetime64("2025-04-01T04:00:00.000000000"),
        np.datetime64("2025-04-01T07:20:00.000000000"),
    ]
    time_peak_lst = [
        np.datetime64("2025-04-01T00:25:00.000000000"),
        np.datetime64("2025-04-01T04:30:00.000000000"),
        np.datetime64("2025-04-01T07:55:00.000000000"),
    ]
    seq_end_lst = [
        np.datetime64("2025-04-01T00:20:00.000000000"),
        np.datetime64("2025-04-01T04:25:00.000000000"),
        np.datetime64("2025-04-01T07:50:00.000000000"),
    ]
    seq_len_lst = [5, 6, 7]
    index = 1

    # Act (execute)
    with patch("matplotlib.pyplot.show"):
        fig, ax = prfp.plot_peak(
            test_ds,
            test_ds_corr,
            "test_station",
            0.95,
            0,
            seq_start_lst,
            time_peak_lst,
            seq_end_lst,
            seq_len_lst,
            True,
            3000,
        )
    figure_bool = isinstance(fig, Figure)
    axes_bool = isinstance(ax, Axes)

    figsize = fig.get_size_inches()

    title = fig.axes[0].get_title()
    lines = fig.axes[0].get_lines()

    marker = lines[index].get_marker()
    markersize = lines[index].get_markersize()
    color = lines[index].get_color()
    alpha = lines[index].get_alpha()

    line_data = lines[index].get_data()

    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Assert (check)
    assert figure_bool is True, "Figure is not a Figure object"
    assert axes_bool is True, "Axes is not an Axes object"
    assert figsize[0] == np.array([6.4, 4.8])[0], "Figure size is not correct"
    assert (
        title == "Corrected rainfall of station test_station with peak no. 1 out of 3"
    ), "Title is not correct"
    assert marker == "None", "Marker is not correct"
    assert markersize == 6.0, "Marker size is not correct"
    assert color == "#1f77b4", "Color is not correct"
    assert alpha is None, "Alpha is not correct"
    assert line_data[0][0] == 20179.0, "Line data is not correct"
    assert xlabel == "time", "X label is not correct"
    assert ylabel == "rainfall [mm]", "Y label is not correct"
    assert xlim == (
        np.float64(20179.0),
        np.float64(20179.32986111111),
    ), "Xlim is not correct"
    assert ylim == (
        np.float64(-0.5750000000000001),
        np.float64(12.075),
    ), "Ylim is not correct"


def test_plot_peak_time_not_in_zoom_range_at_end_plus_peak_stays_same_at_end():
    # Arrange (setup)
    test_ds = create_test_ds()
    test_ds_corr = create_test_ds_corr()
    seq_start_lst = [
        np.datetime64("2025-04-01T00:00:00.000000000"),
        np.datetime64("2025-04-01T04:00:00.000000000"),
        np.datetime64("2025-04-01T07:20:00.000000000"),
    ]
    time_peak_lst = [
        np.datetime64("2025-04-01T00:25:00.000000000"),
        np.datetime64("2025-04-01T04:30:00.000000000"),
        np.datetime64("2025-04-01T07:55:00.000000000"),
    ]
    seq_end_lst = [
        np.datetime64("2025-04-01T00:20:00.000000000"),
        np.datetime64("2025-04-01T04:25:00.000000000"),
        np.datetime64("2025-04-01T07:50:00.000000000"),
    ]
    seq_len_lst = [5, 6, 7]
    index = 1

    # Act (execute)
    with patch("matplotlib.pyplot.show"):
        fig, ax = prfp.plot_peak(
            test_ds,
            test_ds_corr,
            "test_station",
            0.95,
            2,
            seq_start_lst,
            time_peak_lst,
            seq_end_lst,
            seq_len_lst,
            True,
        )
    figure_bool = isinstance(fig, Figure)
    axes_bool = isinstance(ax, Axes)

    figsize = fig.get_size_inches()

    title = fig.axes[0].get_title()
    lines = fig.axes[0].get_lines()

    marker = lines[index].get_marker()
    markersize = lines[index].get_markersize()
    color = lines[index].get_color()
    alpha = lines[index].get_alpha()

    line_data = lines[index].get_data()

    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Assert (check)
    assert figure_bool is True, "Figure is not a Figure object"
    assert axes_bool is True, "Axes is not an Axes object"
    assert figsize[0] == np.array([6.4, 4.8])[0], "Figure size is not correct"
    assert (
        title == "Corrected rainfall of station test_station with peak no. 3 out of 3"
    ), "Title is not correct"
    assert marker == "None", "Marker is not correct"
    assert markersize == 6.0, "Marker size is not correct"
    assert color == "#1f77b4", "Color is not correct"
    assert alpha is None, "Alpha is not correct"
    assert line_data[0][0] == 20179.229166666668, "Line data is not correct"
    assert xlabel == "time", "X label is not correct"
    assert ylabel == "rainfall [mm]", "Y label is not correct"
    assert xlim == (
        np.float64(20179.229166666668),
        np.float64(20179.32986111111),
    ), "Xlim is not correct"
    assert ylim == (
        np.float64(-0.5750000000000001),
        np.float64(12.075),
    ), "Ylim is not correct"


def test_plot_peak_no_corrected_data():
    # Arrange (setup)
    test_ds = create_test_ds()
    test_ds_corr = None
    seq_start_lst = [
        np.datetime64("2025-04-01T00:00:00.000000000"),
        np.datetime64("2025-04-01T04:00:00.000000000"),
        np.datetime64("2025-04-01T07:20:00.000000000"),
    ]
    time_peak_lst = [
        np.datetime64("2025-04-01T00:25:00.000000000"),
        np.datetime64("2025-04-01T04:30:00.000000000"),
        np.datetime64("2025-04-01T07:55:00.000000000"),
    ]
    seq_end_lst = [
        np.datetime64("2025-04-01T00:20:00.000000000"),
        np.datetime64("2025-04-01T04:25:00.000000000"),
        np.datetime64("2025-04-01T07:50:00.000000000"),
    ]
    seq_len_lst = [5, 6, 7]
    index = 1

    # Act (execute)
    with patch("matplotlib.pyplot.show"):
        fig, ax = prfp.plot_peak(
            test_ds,
            test_ds_corr,
            "test_station",
            0.95,
            1,
            seq_start_lst,
            time_peak_lst,
            seq_end_lst,
            seq_len_lst,
            False,
            3000,
        )
    figure_bool = isinstance(fig, Figure)
    axes_bool = isinstance(ax, Axes)

    figsize = fig.get_size_inches()

    title = fig.axes[0].get_title()
    lines = fig.axes[0].get_lines()

    marker = lines[index].get_marker()
    markersize = lines[index].get_markersize()
    color = lines[index].get_color()
    alpha = lines[index].get_alpha()

    line_data = lines[index].get_data()

    xlabel = ax.get_xlabel()
    ylabel = ax.get_ylabel()

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Assert (check)
    assert figure_bool is True, "Figure is not a Figure object"
    assert axes_bool is True, "Axes is not an Axes object"
    assert figsize[0] == np.array([6.4, 4.8])[0], "Figure size is not correct"
    assert (
        title == "Rainfall of station test_station with peak no. 2 out of 3"
    ), "Title is not correct"
    assert marker == "None", "Marker is not correct"
    assert markersize == 6.0, "Marker size is not correct"
    assert color == "#1f77b4", "Color is not correct"
    assert alpha is None, "Alpha is not correct"
    assert line_data[0][0] == 20179.0, "Line data is not correct"
    assert xlabel == "time", "X label is not correct"
    assert ylabel == "rainfall [mm]", "Y label is not correct"
    assert xlim == (
        np.float64(20179.0),
        np.float64(20179.32986111111),
    ), "Xlim is not correct"
    assert ylim == (
        np.float64(-0.5750000000000001),
        np.float64(12.075),
    ), "Ylim is not correct"
