import numpy as np
import xarray as xr

import pypwsqc.peak_removal_filter as prf


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

    rainfall_reshape = [list(x) for x in zip(*rainfall, strict=False)]
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

    rainfall_reshape = [list(x) for x in zip(*rainfall, strict=False)]
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


def test_convert_to_utm():
    # Arrange (setup)
    start_time = np.datetime64("2025-04-01T00:00:00", "ns")
    end_time = np.datetime64("2025-04-01T08:00:00", "ns")
    time = np.arange(start_time, end_time, np.timedelta64(5, "m"))

    index = [
        "test_station",
        "test_neighbor_1",
        "test_neighbor_2",
        "test_neighbor_3",
        "test_neighbor_4",
    ]

    longitude = np.array([9.337699, 9.105562, 9.993434, 9.103393, 9.182794])
    longitude = longitude.astype(np.float64)
    latitude = np.array([48.628241, 48.744145, 48.3974, 48.744335, 48.764399])
    latitude = latitude.astype(np.float64)

    test_ds = xr.Dataset(
        coords={
            "id": index,
            "time": time,
            "lon": (["id"], longitude),
            "lat": (["id"], latitude),
        },
    )
    x_expected = np.array(
        [
            524883.44425925,
            507760.54114806,
            573534.35633562,
            507601.05538652,
            513432.95916921,
        ]
    )
    y_expected = np.array(
        [
            5386185.46713846,
            5399019.60369594,
            5360947.83933791,
            5399040.50555781,
            5401281.78784132,
        ]
    )

    # Act (execute)
    test_ds_utm = prf.convert_to_utm(test_ds, "lon", "lat", 32)

    # Assert (check)
    np.testing.assert_almost_equal(
        test_ds_utm.x.to_numpy(), x_expected, err_msg="\nThe data is not equal!\n"
    )
    np.testing.assert_almost_equal(
        test_ds_utm.y.to_numpy(), y_expected, err_msg="\nThe data is not equal!\n"
    )


def test_get_closest_neighbors():
    # Arrange (setup)
    test_ds = create_test_ds()
    expected_data = create_test_closest_neighbors_ds_ds()

    # Act (execute)
    new_data = prf.get_closest_points_to_point(test_ds, test_ds, 3000, 5)

    # Assert (check)
    np.testing.assert_almost_equal(
        new_data.distance.to_numpy(),
        expected_data.distance.to_numpy(),
        err_msg="\nThe data is not equal!\n",
    )


def test_get_nan_sequences():
    # Arrange (setup)
    test_ds = create_test_ds()
    expected_data = (
        [
            np.datetime64("2025-04-01T00:25:00.000000000"),
            np.datetime64("2025-04-01T04:30:00.000000000"),
            np.datetime64("2025-04-01T07:55:00.000000000"),
        ],
        [
            np.datetime64("2025-04-01T00:00:00.000000000"),
            np.datetime64("2025-04-01T04:00:00.000000000"),
            np.datetime64("2025-04-01T07:20:00.000000000"),
        ],
        [
            np.datetime64("2025-04-01T00:20:00.000000000"),
            np.datetime64("2025-04-01T04:25:00.000000000"),
            np.datetime64("2025-04-01T07:50:00.000000000"),
        ],
        [5, 6, 7],
    )

    # Act (execute)
    new_data = prf.get_nan_sequences(test_ds, "test_station", 0.95, 0)

    # Assert (check)
    np.testing.assert_almost_equal(
        new_data, expected_data, err_msg="\nThe data is not equal!\n"
    )


def test_print_info_only_pws():
    # Arrange (setup)
    test_ds = create_test_ds()
    time_peak_lst = [
        np.datetime64("2025-04-01T00:25:00.000000000"),
        np.datetime64("2025-04-01T04:30:00.000000000"),
        np.datetime64("2025-04-01T07:55:00.000000000"),
    ]
    seq_len_lst = [5, 6, 7]
    aa_closest_neighbors = create_test_closest_neighbors_ds_ds()
    ab_closest_neighbors = None

    expected_data = [
        np.float64(1.9149999999999991),
        3,
        np.float64(6.0),
        np.float64(100.0),
        4,
        0,
    ]

    # Act (execute)
    new_data = prf.print_info(
        test_ds,
        "test_station",
        3000,
        5,
        0.95,
        time_peak_lst,
        seq_len_lst,
        aa_closest_neighbors,
        ab_closest_neighbors,
    )

    # Assert (check)
    np.testing.assert_almost_equal(
        new_data, expected_data, err_msg="\nThe data is not equal!\n"
    )


def test_print_info_with_ref():
    # Arrange (setup)
    test_ds = create_test_ds()
    time_peak_lst = [
        np.datetime64("2025-04-01T00:25:00.000000000"),
        np.datetime64("2025-04-01T04:30:00.000000000"),
        np.datetime64("2025-04-01T07:55:00.000000000"),
    ]
    seq_len_lst = [5, 6, 7]
    aa_closest_neighbors = create_test_closest_neighbors_ds_ds()
    ab_closest_neighbors = create_test_closest_neighbors_ds_ds_ref()

    expected_data = [
        np.float64(1.9149999999999991),
        3,
        np.float64(6.0),
        np.float64(100.0),
        4,
        4,
    ]

    # Act (execute)
    new_data = prf.print_info(
        test_ds,
        "test_station",
        3000,
        5,
        0.95,
        time_peak_lst,
        seq_len_lst,
        aa_closest_neighbors,
        ab_closest_neighbors,
    )

    # Assert (check)
    np.testing.assert_almost_equal(
        new_data, expected_data, err_msg="\nThe data is not equal!\n"
    )


def test_inverse_distance_weighting_ds_ds():
    # Arrange (setup)
    closest_neighbors = create_test_closest_neighbors_ds_ds()
    expected_data = np.array(
        [
            [0.0, 0.25, 0.25, 0.25, 0.25],
            [0.0, 0.44444444, 0.22222222, 0.22222222, 0.11111111],
            [0.0, 0.44444444, 0.22222222, 0.22222222, 0.11111111],
            [0.0, 0.44444444, 0.22222222, 0.22222222, 0.11111111],
            [0.0, 0.44444444, 0.22222222, 0.22222222, 0.11111111],
        ]
    )

    # Act (execute)
    new_data = prf.inverse_distance_weighting(closest_neighbors)

    # Assert (check)
    np.testing.assert_almost_equal(
        new_data.to_numpy(), expected_data, err_msg="\nThe data is not equal!\n"
    )


def test_inverse_distance_weighting_ds_ds_ref():
    # Arrange (setup)
    closest_neighbors = create_test_closest_neighbors_ds_ds_ref()
    expected_data = np.array(
        [
            [0.57142857, 0.14285714, 0.14285714, 0.14285714, 0.0],
            [0.76190476, 0.0952381, 0.0952381, 0.04761905, 0.0],
            [0.0, 0.51612903, 0.32258065, 0.16129032, 0.0],
            [0.0, 0.34615385, 0.34615385, 0.30769231, 0.0],
            [0.0, 0.51612903, 0.32258065, 0.16129032, 0.0],
        ]
    )

    # Act (execute)
    new_data = prf.inverse_distance_weighting(closest_neighbors)

    # Assert (check)
    np.testing.assert_almost_equal(
        new_data.to_numpy(), expected_data, err_msg="\nThe data is not equal!\n"
    )


def test_inverse_distance_weighting_no_neighbors():
    # Arrange (setup)
    closest_neighbors = create_test_closest_neighbors_no_neighbors()
    expected_data = np.full((5, 5), np.nan)

    # Act (execute)
    new_data = prf.inverse_distance_weighting(closest_neighbors)

    # Assert (check)
    np.testing.assert_almost_equal(
        new_data.to_numpy(), expected_data, err_msg="\nThe data is not equal!\n"
    )


def test_interpolate_precipitation_ds_ds():
    # Arrange (setup)
    test_ds = create_test_ds()
    closest_neighbors = create_test_closest_neighbors_ds_ds()
    weights_da = create_test_weights_da()
    time_peak_lst = [
        np.datetime64("2025-04-01T00:25:00.000000000"),
        np.datetime64("2025-04-01T04:30:00.000000000"),
        np.datetime64("2025-04-01T07:55:00.000000000"),
    ]
    seq_start_lst = [
        np.datetime64("2025-04-01T00:00:00.000000000"),
        np.datetime64("2025-04-01T04:00:00.000000000"),
        np.datetime64("2025-04-01T07:20:00.000000000"),
    ]
    seq_len_lst = [5, 6, 7]
    expected_data = [
        np.array([1.075, 1.225, 0.925, 1.175, 0.6, 1.0]),
        np.array([0.975, 0.675, 1.025, 0.85, 0.825, 0.95, 1.025]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]

    # Act (execute)
    new_data = prf.interpolate_precipitation(
        test_ds,
        "test_station",
        closest_neighbors,
        weights_da,
        seq_start_lst,
        time_peak_lst,
        seq_len_lst,
        0.5,
        4,
    )

    # Assert (check)
    for i in range(len(expected_data)):
        np.testing.assert_almost_equal(
            new_data[i], expected_data[i], err_msg="\nThe data is not equal!\n"
        )


def test_interpolate_precipitation_ds_ds_ref():
    # Arrange (setup)
    test_ds_ref = create_test_ds_ref()
    closest_neighbors = create_test_closest_neighbors_ds_ds_ref()
    weights_da = create_test_weights_da_ref()
    time_peak_lst = [
        np.datetime64("2025-04-01T00:25:00.000000000"),
        np.datetime64("2025-04-01T04:30:00.000000000"),
        np.datetime64("2025-04-01T07:55:00.000000000"),
    ]
    seq_start_lst = [
        np.datetime64("2025-04-01T00:00:00.000000000"),
        np.datetime64("2025-04-01T04:00:00.000000000"),
        np.datetime64("2025-04-01T07:20:00.000000000"),
    ]
    seq_len_lst = [5, 6, 7]
    expected_data = [
        np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
        np.array(
            [
                0.28571429,
                1.08571429,
                0.51428571,
                0.22857143,
                0.51428571,
                1.34285714,
                0.48571429,
            ]
        ),
        np.array(
            [
                1.05714286,
                0.68571429,
                1.42857143,
                0.51428571,
                1.11428571,
                0.77142857,
                0.82857143,
                0.88571429,
            ]
        ),
    ]

    # Act (execute)
    new_data = prf.interpolate_precipitation(
        test_ds_ref,
        "test_station",
        closest_neighbors,
        weights_da,
        seq_start_lst,
        time_peak_lst,
        seq_len_lst,
        0.5,
        1,
    )

    # Assert (check)
    for i in range(len(expected_data)):
        np.testing.assert_almost_equal(
            new_data[i], expected_data[i], err_msg="\nThe data is not equal!\n"
        )


def test_distribute_peak_ds_ds():
    # Arrange (setup)
    test_ds = create_test_ds()
    time_peak_lst = [
        np.datetime64("2025-04-01T00:25:00.000000000"),
        np.datetime64("2025-04-01T04:30:00.000000000"),
        np.datetime64("2025-04-01T07:55:00.000000000"),
    ]
    seqs_lst = [
        np.array([1.075, 1.225, 0.925, 1.175, 0.6, 1.0]),
        np.array([0.975, 0.675, 1.025, 0.85, 0.825, 0.95, 1.025]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]
    expected_data = [
        np.array([1.37958333, 1.57208333, 1.18708333, 1.50791667, 0.77, 1.28333333]),
        np.array(
            [
                1.51067194,
                1.0458498,
                1.58814229,
                1.31699605,
                1.27826087,
                1.47193676,
                1.58814229,
            ]
        ),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ]

    # Act (execute)
    new_data = prf.distribute_peak(test_ds, "test_station", time_peak_lst, seqs_lst)

    # Assert (check)
    for i in range(len(expected_data)):
        np.testing.assert_almost_equal(
            new_data[i], expected_data[i], err_msg="\nThe data is not equal!\n"
        )


def test_distribute_peak_ds_ds_ref():
    # Arrange (setup)
    test_ds = create_test_ds()
    time_peak_lst = [
        np.datetime64("2025-04-01T00:25:00.000000000"),
        np.datetime64("2025-04-01T04:30:00.000000000"),
        np.datetime64("2025-04-01T07:55:00.000000000"),
    ]
    seqs_lst = [
        np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
        np.array(
            [
                0.28571429,
                1.08571429,
                0.51428571,
                0.22857143,
                0.51428571,
                1.34285714,
                0.48571429,
            ]
        ),
        np.array(
            [
                1.05714286,
                0.68571429,
                1.42857143,
                0.51428571,
                1.11428571,
                0.77142857,
                0.82857143,
                0.88571429,
            ]
        ),
    ]
    expected_data = [
        np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
        np.array(
            [
                0.62820513,
                2.38717949,
                1.13076923,
                0.5025641,
                1.13076923,
                2.9525641,
                1.06794872,
            ]
        ),
        np.array(
            [
                1.66862745,
                1.08235294,
                2.25490196,
                0.81176471,
                1.75882353,
                1.21764706,
                1.30784314,
                1.39803922,
            ]
        ),
    ]

    # Act (execute)
    new_data = prf.distribute_peak(test_ds, "test_station", time_peak_lst, seqs_lst)

    # Assert (check)
    for i in range(len(expected_data)):
        np.testing.assert_almost_equal(
            new_data[i], expected_data[i], err_msg="\nThe data is not equal!\n"
        )


def test_overwrite_seq_ds_ds():
    # Arrange (setup)
    test_ds = create_test_ds()
    seqs_corr_lst = [
        np.array([1.37958333, 1.57208333, 1.18708333, 1.50791667, 0.77, 1.28333333]),
        np.array(
            [
                1.51067194,
                1.0458498,
                1.58814229,
                1.31699605,
                1.27826087,
                1.47193676,
                1.58814229,
            ]
        ),
        np.array(
            [
                0.89563863,
                1.68380062,
                1.07476636,
                2.14953271,
                1.25389408,
                1.28971963,
                1.28971963,
                1.86292835,
            ]
        ),
    ]
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
    expected_data = np.array(
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
            0.89563863,
            1.68380062,
            1.07476636,
            2.14953271,
            1.25389408,
            1.28971963,
            1.28971963,
            1.86292835,
        ]
    )

    # Act (execute)
    new_data = (
        prf.overwrite_seq(
            test_ds, "test_station", seqs_corr_lst, seq_start_lst, time_peak_lst
        )
        .sel(id="test_station")
        .rainfall.to_numpy()
    )

    # Assert (check)
    np.testing.assert_almost_equal(
        new_data, expected_data, err_msg="\nThe data is not equal!\n"
    )


def test_overwrite_seq_ds_ds_ref():
    # Arrange (setup)
    test_ds = create_test_ds()
    seqs_corr_lst = [
        np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]),
        np.array(
            [
                0.62820513,
                2.38717949,
                1.13076923,
                0.5025641,
                1.13076923,
                2.9525641,
                1.06794872,
            ]
        ),
        np.array(
            [
                1.64285714,
                1.06563707,
                2.22007722,
                0.97683398,
                1.73166023,
                1.1988417,
                1.28764479,
                1.37644788,
            ]
        ),
    ]
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
    expected_data = np.array(
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
            0.62820513,
            2.38717949,
            1.13076923,
            0.5025641,
            1.13076923,
            2.9525641,
            1.06794872,
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
            1.64285714,
            1.06563707,
            2.22007722,
            0.97683398,
            1.73166023,
            1.1988417,
            1.28764479,
            1.37644788,
        ]
    )

    # Act (execute)
    new_data = (
        prf.overwrite_seq(
            test_ds, "test_station", seqs_corr_lst, seq_start_lst, time_peak_lst
        )
        .sel(id="test_station")
        .rainfall.to_numpy()
    )

    # Assert (check)
    np.testing.assert_almost_equal(
        new_data, expected_data, err_msg="\nThe data is not equal!\n"
    )
