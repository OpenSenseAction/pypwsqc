from __future__ import annotations

import numpy as np
import xarray as xr

import pypwsqc


def test_fz_filter():
    # fmt: off

    #Test 1. Station reports no rain, neighbours are reporting rain
    pws_data = np.array(
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
    0.   , 0., 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
    0.   , 0.   , 0., 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
    0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0., 0.   ,
    0.   ])

    pws_data = xr.DataArray(np.atleast_2d(pws_data), coords={'id': ['station_1',], 'time': range(len(pws_data))})

    nbrs_not_nan = np.array([24, 23, 24, 23, 26, 23, 25, 22, 23, 23, 23, 23, 23, 23, 22, 23, 23,
    26, 26, 25, 29, 22, 21, 25, 23, 24, 23, 24, 23, 2, 23, 25, 22, 23, 23, 2, 23])

    reference = np.array(
    [0.101     , 0.25136087, 0.1010425 , 0.1010425        , 0.05012029,
    0.101     , 0.101     , 0.101     , 0.303     , 0.20048115,
    0.202     , 0.202     , 0.202     , 0.303     , 0.202     ,
    0.202     , 0.202     , 0.202     , 0.101     , 0.05012029,
    0.101     , 0.10062029, 0.0505    ,   0.202         ,   0.202         ,
    0.101     , 0.101     , 0.101     , 0.0505    ,   0.202         ,
        0.202         ,   0.202         ,   0.202         ,   0.202         ,   0.202         ,
        0.202         ,   0.202         ])


    reference = xr.DataArray(np.atleast_2d(reference), coords={'id': ['station_1',], 'time': range(len(reference))})

    expected = np.array([-1., -1., -1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
        1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., -1., 1., 1., 1., 1.,
        1., -1., 1.])

    expected = xr.DataArray(np.atleast_2d(expected), coords={'id': ['station_1',], 'time': range(len(expected))})
        # fmt: on

    ds = xr.Dataset(
        {"pws_data": pws_data, "reference": reference, "expected": expected}
    )

    result = pypwsqc.flagging.fz_filter(
        ds.pws_data,
        nbrs_not_nan,
        ds.reference,
        nint=3,
        n_stat=5,
    )

    np.testing.assert_almost_equal(expected, result)

    # Test 2. same as test 1 but with different nint.
    # fmt: off

    pws_data = np.array(
    [ [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
        0.   , 0.101, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
        0.   , 0.   , 0.101, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
        0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.101, 0.   ,
        0.   ]])

    pws_data = xr.DataArray(np.atleast_2d(pws_data), coords={'id': ['station_2'], 'time': range(len(pws_data[0]))})

    nbrs_not_nan = np.array([24, 23, 24, 23, 26, 23, 25, 22, 23, 23, 23, 23, 23, 23, 22, 23, 23,
    26, 26, 25, 29, 22, 21, 25, 23, 24, 23, 24, 23, 2, 23, 25, 22, 23, 23, 2, 23])

    reference = np.array(
    [ [0.101     , 0.25136087, 0.1010425 , 0.        , 0.05012029,
        0.101     , 0.101     , 0.101     , 0.303     , 0.20048115,
        0.202     , 0.202     , 0.202     , 0.303     , 0.202     ,
        0.202     , 0.202     , 0.202     , 0.101     , 0.05012029,
        0.101     , 0.10062029, 0.0505    , 0.        , 0.        ,
        0.101     , 0.101     , 0.101     , 0.0505    , 0.        ,
        0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        ]])


    reference = xr.DataArray(np.atleast_2d(reference), coords={'id': ['station_2'], 'time': range(len(reference[0]))})

    expected = np.array([ [-1., -1., -1., -1., -1., -1.,  -1.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  -1.,  0.,  0.,  0.,  0.,  0.,  -1.,  0]])

    expected = xr.DataArray(
        np.atleast_2d(expected),
        coords={
            "id": [
                "station_2",
            ],
            "time": range(len(expected[0])),
        },
    )
    # fmt: on

    ds = xr.Dataset(
        {"pws_data": pws_data, "reference": reference, "expected": expected}
    )

    result = pypwsqc.flagging.fz_filter(
        ds.pws_data,
        nbrs_not_nan,
        ds.reference,
        nint=7,
        n_stat=5,
    )

    np.testing.assert_almost_equal(expected, result)


def test_hi_filter():
    # for max_distance = 10e3
    # fmt: off
    pws_data = np.array([[0.       , 0.       , 0.101    , 0.       , 0.101    , 0.       ,
       0.       , 0.       , 0.       , 0.       , 0.       , 1.3130001,
       3.7370002, 0.404    , 0.       , 0.       , 0.       , 0.       ,
       0.       , 0.       , 0.       , 0.       , 0.       , 0.       ,
       0.       ]])

    pws_data = xr.DataArray(np.atleast_2d(pws_data), coords={'id': ['station_1',], 'time': range(len(pws_data[0]))})

    nbrs_not_nan = np.array([24, 23, 24, 23, 26, 23, 25, 22, 23, 23, 23, 23, 23, 23, 22, 23, 23,
       26, 26, 25, 25, 23, 27, 25, 23])

    reference = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0.]])

    reference = xr.DataArray(np.atleast_2d(reference), coords={'id': ['station_1',], 'time': range(len(reference[0]))})

    expected = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0.]])

    expected = xr.DataArray(np.atleast_2d(expected), coords={'id': ['station_1',], 'time': range(len(expected[0]))})

    ds = xr.Dataset({"pws_data": pws_data, "reference": reference, "expected": expected})

    # fmt: on
    result = pypwsqc.flagging.hi_filter(
        ds.pws_data,
        nbrs_not_nan,
        ds.reference,
        hi_thres_a=0.4,
        hi_thres_b=10,
        n_stat=5,
    )

    np.testing.assert_almost_equal(expected, result.data)

    # the same test as above but with different `hi_thres_b`
    # fmt: off

    expected = np.array([[0.       , 0.       , 0.   , 0.       , 0.    , 0.       ,
       0.       , 0.       , 0.       , 0.       , 0.       , 0.,
       1, 0.    , 0.       , 0.       , 0.       , 0.       ,
       0.       , 0.       , 0.       , 0.       , 0.       , 0.       ,
       0.       ]])

    expected = xr.DataArray(np.atleast_2d(expected), coords={'id': ['station_1',], 'time': range(len(expected[0]))})

    ds = xr.Dataset({"pws_data": pws_data, "reference": reference, "expected": expected})

    # fmt: on
    result = pypwsqc.flagging.hi_filter(
        ds.pws_data,
        nbrs_not_nan,
        ds.reference,
        hi_thres_a=0.4,
        hi_thres_b=3,
        n_stat=5,
    )

    np.testing.assert_almost_equal(expected, result.data)

    # running test again with different IO

    # fmt: off
    pws_data = np.array([[0, 0, 0, 0, 15, 0, 15, 0]])
    pws_data = xr.DataArray(np.atleast_2d(pws_data), coords={'id': ['station_1',], 'time': range(len(pws_data[0]))})

    nbrs_not_nan = np.array([2, 2, 2, 2, 12, 12, 12, 12])

    reference = np.array([[0.1, 0.2, 0.35, 0.2, 0.1, 0.3, 0.2, 0.2]])
    reference = xr.DataArray(np.atleast_2d(reference), coords={'id': ['station_1',], 'time': range(len(reference[0]))})

    expected = np.array([[-1, -1, -1, -1, 1, 0, 1, 0]])

    expected = xr.DataArray(np.atleast_2d(expected), coords={'id': ['station_1',], 'time': range(len(expected[0]))})

    ds = xr.Dataset({"pws_data": pws_data, "reference": reference, "expected": expected})

    # fmt: on
    result = pypwsqc.flagging.hi_filter(
        ds.pws_data,
        nbrs_not_nan,
        ds.reference,
        hi_thres_a=0.4,
        hi_thres_b=10,
        n_stat=5,
    )
    np.testing.assert_almost_equal(expected, result.data)
