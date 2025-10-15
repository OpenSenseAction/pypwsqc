from __future__ import annotations

import numpy as np
import poligrain as plg
import xarray as xr

import pypwsqc


def test_fz_filter():
    # fmt: off

    ds_pws = xr.open_dataset("tests/test_dataset.nc")
    distance_matrix = plg.spatial.calc_point_to_point_distances(ds_pws, ds_pws)

    #Test 1. Station reports no rain, neighbours are reporting rain
    pws_data = np.array(
    [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
    0.   , 0., 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
    0.   , 0.   , 0., 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
    0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0., 0.   ,
    0.   ])

    rainfall = xr.DataArray(np.atleast_2d(pws_data), coords={'id': ['station_1',], 'time': range(len(pws_data))})

    nbrs_not_nan = np.array([24, 23, 24, 23, 26, 23, 25, 22, 23, 23, 23, 23, 23, 23, 22, 23, 23,
    26, 26, 25, 29, 22, 21, 25, 23, 24, 23, 24, 23, 2, 23, 25, 22, 23, 23, 2, 23])

    nbrs_not_nan = xr.DataArray(np.atleast_2d(nbrs_not_nan), coords={'id': ['station_1',], 'time': range(len(nbrs_not_nan))})

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
        {
            "rainfall": rainfall,
            "reference": reference,
            "expected": expected,
            "nbrs_not_nan": nbrs_not_nan,
        }
    )

    result = pypwsqc.flagging.fz_filter(
        ds,
        nint=3,
        n_stat=5,
        distance_matrix=distance_matrix,
        max_distance=10e3,
    )

    np.testing.assert_almost_equal(expected[0], result.fz_flag.data[0])

    # Test 2. same as test 1 but with different nint.
    # fmt: off

    pws_data = np.array(
    [ [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
        0.   , 0.101, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
        0.   , 0.   , 0.101, 0.   , 0.   , 0.   , 0.   , 0.   , 0.   ,
        0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.101, 0.   ,
        0.   ]])

    rainfall = xr.DataArray(np.atleast_2d(pws_data), coords={'id': ['station_2'], 'time': range(len(pws_data[0]))})

    nbrs_not_nan = np.array([24, 23, 24, 23, 26, 23, 25, 22, 23, 23, 23, 23, 23, 23, 22, 23, 23,
    26, 26, 25, 29, 22, 21, 25, 23, 24, 23, 24, 23, 2, 23, 25, 22, 23, 23, 2, 23])

    nbrs_not_nan = xr.DataArray(np.atleast_2d(nbrs_not_nan), coords={'id': ['station_2',], 'time': range(len(nbrs_not_nan))})

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
        {
            "rainfall": rainfall,
            "reference": reference,
            "expected": expected,
            "nbrs_not_nan": nbrs_not_nan,
        }
    )

    result = pypwsqc.flagging.fz_filter(
        ds,
        nint=7,
        n_stat=5,
        distance_matrix=distance_matrix,
        max_distance=10e3,
    )

    np.testing.assert_almost_equal(expected[0], result.fz_flag.data[0])

    # Test 3. Test when reference is not included in data set
    # reproduce the flags for Ams11, 2017-07-15 to 2017-07-30

    ds_pws = xr.open_dataset("tests/test_dataset.nc")
    ds_pws = ds_pws.sel(time=slice("2017-07-15", "2017-07-30"))
    expected_dataset = xr.open_dataset("tests/expected_array_fz_hi.nc")
    expected = expected_dataset.fz_flag
    distance_matrix = plg.spatial.calc_point_to_point_distances(ds_pws, ds_pws)
    pws_id = "ams11"

    result = pypwsqc.flagging.fz_filter(
        ds_pws,
        nint=6,
        n_stat=5,
        distance_matrix=distance_matrix,
        max_distance=10e3,
    )

    result_flags = result.fz_flag.sel(id=pws_id)

    np.testing.assert_almost_equal(expected.to_numpy(), result_flags.to_numpy())


def test_hi_filter():
    # fmt: off

    ds_pws = xr.open_dataset("tests/test_dataset.nc")
    distance_matrix = plg.spatial.calc_point_to_point_distances(ds_pws, ds_pws)

    pws_data = np.array([[0.       , 0.       , 0.101    , 0.       , 0.101    , 0.       ,
       0.       , 0.       , 0.       , 0.       , 0.       , 1.3130001,
       3.7370002, 0.404    , 0.       , 0.       , 0.       , 0.       ,
       0.       , 0.       , 0.       , 0.       , 0.       , 0.       ,
       0.       ]])

    rainfall = xr.DataArray(np.atleast_2d(pws_data), coords={'id': ['station_1',], 'time': range(len(pws_data[0]))})

    nbrs_not_nan = np.array([24, 23, 24, 23, 26, 23, 25, 22, 23, 23, 23, 23, 23, 23, 22, 23, 23,
       26, 26, 25, 25, 23, 27, 25, 23])
    nbrs_not_nan = xr.DataArray(np.atleast_2d(nbrs_not_nan), coords={'id': ['station_1',], 'time': range(len(nbrs_not_nan))})


    reference = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0.]])

    reference = xr.DataArray(np.atleast_2d(reference), coords={'id': ['station_1',], 'time': range(len(reference[0]))})

    expected = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0.]])

    expected = xr.DataArray(np.atleast_2d(expected), coords={'id': ['station_1',], 'time': range(len(expected[0]))})

    ds = xr.Dataset(
        {
            "rainfall": rainfall,
            "reference": reference,
            "expected": expected,
            "nbrs_not_nan": nbrs_not_nan,
        }
    )

    # fmt: on
    result = pypwsqc.flagging.hi_filter(
        ds,
        hi_thres_a=0.4,
        hi_thres_b=10,
        nint=8064,
        n_stat=5,
        distance_matrix=distance_matrix,
        max_distance=10e3,
    )

    np.testing.assert_almost_equal(expected, result.hi_flag.data)

    # the same test as above but with different `hi_thres_b`
    # fmt: off

    expected = np.array([[0.       , 0.       , 0.   , 0.       , 0.    , 0.       ,
       0.       , 0.       , 0.       , 0.       , 0.       , 0.,
       1, 0.    , 0.       , 0.       , 0.       , 0.       ,
       0.       , 0.       , 0.       , 0.       , 0.       , 0.       ,
       0.       ]])

    expected = xr.DataArray(np.atleast_2d(expected), coords={'id': ['station_1',], 'time': range(len(expected[0]))})

    ds = xr.Dataset(
        {
            "rainfall": rainfall,
            "reference": reference,
            "expected": expected,
            "nbrs_not_nan": nbrs_not_nan,
        }
    )
    # fmt: on
    result = pypwsqc.flagging.hi_filter(
        ds,
        hi_thres_a=0.4,
        hi_thres_b=3,
        nint=8064,
        n_stat=5,
        distance_matrix=distance_matrix,
        max_distance=10e3,
    )

    np.testing.assert_almost_equal(expected, result.hi_flag.data)

    # running test again with different IO

    # fmt: off
    pws_data = np.array([[0, 0, 0, 0, 15, 0, 15, 0]])
    rainfall = xr.DataArray(np.atleast_2d(pws_data), coords={'id': ['station_1',], 'time': range(len(pws_data[0]))})

    nbrs_not_nan = np.array([2, 2, 2, 2, 12, 12, 12, 12])
    nbrs_not_nan = xr.DataArray(np.atleast_2d(nbrs_not_nan), coords={'id': ['station_1',], 'time': range(len(nbrs_not_nan))})

    reference = np.array([[0.1, 0.2, 0.35, 0.2, 0.1, 0.3, 0.2, 0.2]])
    reference = xr.DataArray(np.atleast_2d(reference), coords={'id': ['station_1',], 'time': range(len(reference[0]))})

    expected = np.array([[-1, -1, -1, -1, 1, 0, 1, 0]])

    expected = xr.DataArray(np.atleast_2d(expected), coords={'id': ['station_1',], 'time': range(len(expected[0]))})

    ds = xr.Dataset({"rainfall": rainfall, "reference": reference, "expected": expected,  "nbrs_not_nan":nbrs_not_nan,})

    # fmt: on
    result = pypwsqc.flagging.hi_filter(
        ds,
        hi_thres_a=0.4,
        hi_thres_b=10,
        nint=8064,
        n_stat=5,
        distance_matrix=distance_matrix,
        max_distance=10e3,
    )

    np.testing.assert_almost_equal(expected, result.hi_flag.data)

    # Test 3. Test when reference is not included in data set
    # reproduce the flags for Ams11, 2017-07-15 to 2017-07-30

    ds_pws = xr.open_dataset("tests/test_dataset.nc")
    ds_pws = ds_pws.sel(time=slice("2017-07-15", "2017-07-30"))
    expected_dataset = xr.open_dataset("tests/expected_array_fz_hi.nc")
    expected = expected_dataset.hi_flag
    distance_matrix = plg.spatial.calc_point_to_point_distances(ds_pws, ds_pws)
    pws_id = "ams11"

    result = pypwsqc.flagging.hi_filter(
        ds_pws,
        hi_thres_a=0.4,
        hi_thres_b=10,
        nint=6,
        n_stat=5,
        distance_matrix=distance_matrix,
        max_distance=10e3,
    )

    result_flags = result.hi_flag.sel(id=pws_id)

    np.testing.assert_almost_equal(expected.to_numpy(), result_flags.to_numpy())


def test_so_filter():
    # reproduce the flags for Ams16, 2017-08-12 to 2017-10-15

    ds_pws = xr.open_dataset("tests/test_dataset.nc")
    expected_dataset = xr.open_dataset("tests/expected_array_so_bias.nc")
    expected = expected_dataset.so_flag
    distance_matrix = plg.spatial.calc_point_to_point_distances(ds_pws, ds_pws)
    evaluation_period = 8064
    pws_id = "ams16"

    ds_pws["so_flag"] = xr.DataArray(
        np.ones((len(ds_pws.id), len(ds_pws.time))) * -999, dims=("id", "time")
    )
    ds_pws["median_corr_nbrs"] = xr.DataArray(
        np.ones((len(ds_pws.id), len(ds_pws.time))) * -999, dims=("id", "time")
    )

    result = pypwsqc.flagging.so_filter(
        ds_pws=ds_pws,
        distance_matrix=distance_matrix,
        evaluation_period=evaluation_period,
        mmatch=200,
        gamma=0.15,
        n_stat=5,
        max_distance=10e3,
    )
    result_flags = result.so_flag.isel(time=slice(evaluation_period, None)).sel(
        id=pws_id
    )

    np.testing.assert_almost_equal(expected.to_numpy(), result_flags.to_numpy())

    # test for when Bias == True
    result = pypwsqc.flagging.so_filter(
        ds_pws=ds_pws,
        distance_matrix=distance_matrix,
        evaluation_period=evaluation_period,
        mmatch=200,
        gamma=0.15,
        n_stat=5,
        max_distance=10e3,
        bias_corr=True,
        beta=0.2,
        dbc=1,
    )

    result_flags = result.so_flag.isel(time=slice(evaluation_period, None)).sel(
        id=pws_id
    )

    np.testing.assert_almost_equal(expected.to_numpy(), result_flags.to_numpy())

    # test when there are no neighbors within max_distance
    # (because of small max_distance)
    ds_pws = xr.open_dataset("tests/test_dataset.nc")
    expected = expected_dataset.minus_one
    distance_matrix = plg.spatial.calc_point_to_point_distances(ds_pws, ds_pws)
    evaluation_period = 8064
    pws_id = "ams16"

    ds_pws["so_flag"] = xr.DataArray(
        np.ones((len(ds_pws.id), len(ds_pws.time))) * -999, dims=("id", "time")
    )
    ds_pws["median_corr_nbrs"] = xr.DataArray(
        np.ones((len(ds_pws.id), len(ds_pws.time))) * -999, dims=("id", "time")
    )

    result = pypwsqc.flagging.so_filter(
        ds_pws=ds_pws,
        distance_matrix=distance_matrix,
        evaluation_period=evaluation_period,
        mmatch=200,
        gamma=0.15,
        n_stat=5,
        max_distance=5,
    )
    result_flags = result.so_flag.isel(time=slice(evaluation_period, None)).sel(
        id=pws_id
    )

    np.testing.assert_almost_equal(expected.to_numpy(), result_flags.to_numpy())


def test_bias_corr():
    # reproduce the flags for Ams16, 2017-08-12 to 2017-10-15

    ds_pws = xr.open_dataset("tests/test_dataset.nc")
    expected_dataset = xr.open_dataset("tests/expected_array_so_bias.nc")
    expected = expected_dataset.bias_corr_factor
    distance_matrix = plg.spatial.calc_point_to_point_distances(ds_pws, ds_pws)
    evaluation_period = 8064
    pws_id = "ams16"
    dbc = 1

    # initialize
    ds_pws["BCF_new"] = xr.DataArray(
        np.ones((len(ds_pws.id), len(ds_pws.time))) * -999, dims=("id", "time")
    )

    ds_pws["bias_corr_factor"] = xr.DataArray(
        np.ones((len(ds_pws.id), len(ds_pws.time))) * dbc, dims=("id", "time")
    )

    result = pypwsqc.flagging.bias_correction(
        ds_pws,
        evaluation_period,
        distance_matrix,
        max_distance=10e3,
        beta=0.2,
        dbc=1,
    )

    result_flags = result.bias_corr_factor.isel(
        time=slice(evaluation_period, None)
    ).sel(id=pws_id)

    np.testing.assert_almost_equal(expected.to_numpy(), result_flags.to_numpy())

    # test when there are no neighbors within max_distance
    # (because of small max_distance)
    ds_pws = xr.open_dataset("tests/test_dataset.nc")
    expected = expected_dataset.minus_one
    distance_matrix = plg.spatial.calc_point_to_point_distances(ds_pws, ds_pws)

    # initialize
    ds_pws["BCF_new"] = xr.DataArray(
        np.ones((len(ds_pws.id), len(ds_pws.time))) * -999, dims=("id", "time")
    )

    ds_pws["bias_corr_factor"] = xr.DataArray(
        np.ones((len(ds_pws.id), len(ds_pws.time))) * dbc, dims=("id", "time")
    )

    result = pypwsqc.flagging.bias_correction(
        ds_pws,
        evaluation_period,
        distance_matrix,
        max_distance=5,
        beta=0.2,
        dbc=1,
    )

    result_flags = result.bias_corr_factor.isel(
        time=slice(evaluation_period, None)
    ).sel(id=pws_id)

    np.testing.assert_almost_equal(expected.to_numpy(), result_flags.to_numpy())
