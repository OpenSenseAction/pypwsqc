import numpy as np
import numpy.testing as npt
import poligrain as plg
import pytest
import xarray as xr

import pypwsqc.indicator_correlation as ic


def test_indicator_correlation():
    rng = np.random.default_rng()
    x = np.abs(rng.standard_normal(100))
    npt.assert_almost_equal(ic._indicator_correlation(x, x, prob=0.1), 1.0)
    npt.assert_almost_equal(ic._indicator_correlation(x, x * 0.7, prob=0.1), 1.0)

    npt.assert_almost_equal(
        ic._indicator_correlation(
            np.array([0, 1, 2, 3]),
            np.array([0, 2, 1, 4]),
            prob=0.75,
        ),
        1.0,
    )

    npt.assert_almost_equal(
        ic._indicator_correlation(
            np.array([0, 1, 2, 3]),
            np.array([0, 1, 2, 1]),
            prob=0.75,
        ),
        -0.33333333333333,
    )


def test_indicator_correlation_raise():
    # test with dataset a having negative values
    with pytest.raises(
        ValueError, match="input arrays must not contain negative values"
    ):
        ic._indicator_correlation(
            np.array([-1, -1, 1]),
            np.array([1, 0, 1]),
            prob=0.5,
        )
    # test with dataset b having negative values
    with pytest.raises(
        ValueError, match="input arrays must not contain negative values"
    ):
        ic._indicator_correlation(
            np.array([1, 0, 1]),
            np.array([-1, 0, 1]),
            prob=0.5,
        )

    with pytest.raises(ValueError, match="`a_dataset` has to be a 1D numpy.ndarray"):
        ic._indicator_correlation(
            np.array([[1, 0, 1], [1, 1, 1]]),
            np.array([-1, 0, 1]),
            prob=0.5,
        )

    with pytest.raises(
        ValueError, match="`a_dataset` and `b_dataset` have to have the same shape"
    ):
        ic._indicator_correlation(
            np.array([1, 0, 1, 1]),
            np.array([-1, 0, 1]),
            prob=0.5,
        )

    npt.assert_almost_equal(
        ic._indicator_correlation(
            np.array([np.nan, 1, 1]),
            np.array([1, np.nan, 1]),
            prob=0.5,
            min_valid_overlap=2,
        ),
        np.nan,
    )

    with pytest.raises(
        ValueError,
        match="No overlapping data. Define `min_valid_overlap` to return NaN in such cases.",
    ):
        ic._indicator_correlation(
            np.array([np.nan, np.nan]),
            np.array([np.nan, 1]),
            prob=0.5,
        )


def test_calc_indic_corr_all_stns():
    incorrtest = np.array(
        [
            [np.nan, np.nan, np.nan, np.nan],
            [1.0, 0.76393274, 0.59704723, 0.63568654],
            [0.76393274, 1.0, 0.63765614, 0.53883098],
            [0.59704723, 0.63765614, 1.0, 0.45904971],
        ]
    )

    distmatxtest = np.array(
        [
            [7746.71884635, 7687.70945578, 6795.89829624, 15988.10174513],
            [0.0, 3961.90905573, 13071.3586498, 9795.95383179],
            [3961.90905573, 0.0, 10958.53189906, 13483.70000333],
            [13071.3586498, 10958.53189906, 0.0, 22347.52572054],
        ]
    )

    ds_a = xr.open_dataset("./docs/notebooks/data/RadarRef_AMS.nc")
    ds_a.load()
    ds_a.coords["x"], ds_a.coords["y"] = plg.spatial.project_point_coordinates(
        ds_a.lon,
        ds_a.lat,
        target_projection="EPSG:25832",
    )

    dist_mtx, ind_corr_mtx = ic.indicator_distance_matrix(
        ds_a.rainfall,
        ds_a.rainfall,
        max_distance=30e3,
        prob=0.99,
        min_valid_overlap=2 * 24 * 30,
    )

    npt.assert_almost_equal(dist_mtx[3:7, 4:8].values, distmatxtest)
    npt.assert_almost_equal(ind_corr_mtx[3:7, 4:8].values, incorrtest)

    npt.assert_almost_equal(dist_mtx.data.sum(), 4883181.512681292)
    npt.assert_almost_equal(ind_corr_mtx.sum(), 212.76937646)


def test_indicator_correlation_filter():
    ds_a = xr.open_dataset("./docs/notebooks/data/RadarRef_AMS.nc")
    ds_a.load()
    ds_a.coords["x"], ds_a.coords["y"] = plg.spatial.project_point_coordinates(
        ds_a.lon,
        ds_a.lat,
        target_projection="EPSG:25832",
    )

    ds_a = ds_a.drop_sel(id="3")
    dist1, ind1 = ic.indicator_distance_matrix(
        ds_a.rainfall,
        ds_a.rainfall,
        max_distance=30e3,
        prob=0.99,
        min_valid_overlap=2 * 24 * 30,
    )

    ds_b = xr.DataArray.copy(ds_a)
    # calculate x-quantile for manipulating data
    quantile = (
        ds_b.isel(id=0).rainfall.quantile([0.98], dim="time", method="linear").data
    )
    ds_b.rainfall.data[0, ds_b.rainfall.isel(id=0) > quantile] = 0

    dist2, ind2 = ic.indicator_distance_matrix(
        ds_a.rainfall,
        ds_b.rainfall,
        max_distance=30e3,
        prob=0.99,
        min_valid_overlap=2 * 24 * 30,
    )

    indcorr_results_orig = ic.ic_filter(
        indicator_correlation_matrix_ref=ind1,
        distance_correlation_matrix_ref=dist1,
        indicator_correlation_matrix=ind1,
        distance_matrix=dist1,
        max_distance=30000,
        bin_size=1000,
        quantile_bin_ref=0.1,
        quantile_bin_pws=0.5,
        threshold=0.05,
    )

    indcorr_results_manip = ic.ic_filter(
        indicator_correlation_matrix_ref=ind1,
        distance_correlation_matrix_ref=dist1,
        indicator_correlation_matrix=ind2,
        distance_matrix=dist2,
        max_distance=30000,
        bin_size=1000,
        quantile_bin_ref=0.1,
        quantile_bin_pws=0.5,
        threshold=0.05,
    )

    npt.assert_almost_equal(indcorr_results_orig.indcorr_good.sum(), 19)
    npt.assert_almost_equal(indcorr_results_manip.indcorr_good.sum(), 18)
    assert indcorr_results_manip.indcorr_good.data[0] == False  # noqa: E712
