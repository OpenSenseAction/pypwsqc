import numpy as np
import numpy.testing as npt
import pytest

import pypwsqc.pwspyqc_dev_js as pyqc


def test_indicator_correlation():
    rng = np.random.default_rng()
    x = np.abs(rng.standard_normal(100))
    npt.assert_almost_equal(pyqc.calc_indicator_correlation(x, x, prob=0.1), 1.0)
    npt.assert_almost_equal(pyqc.calc_indicator_correlation(x, x * 0.7, prob=0.1), 1.0)

    npt.assert_almost_equal(
        pyqc.calc_indicator_correlation(
            np.array([0, 1, 2, 3]),
            np.array([0, 2, 1, 4]),
            prob=0.75,
        ),
        1.0,
    )

    npt.assert_almost_equal(
        pyqc.calc_indicator_correlation(
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
        pyqc.calc_indicator_correlation(
            np.array([-1, -1, 1]),
            np.array([1, 0, 1]),
            prob=0.5,
        )
    # test with dataset b having negative values
    with pytest.raises(
        ValueError, match="input arrays must not contain negative values"
    ):
        pyqc.calc_indicator_correlation(
            np.array([1, 0, 1]),
            np.array([-1, 0, 1]),
            prob=0.5,
        )

    with pytest.raises(ValueError, match="`a_dataset` has to be a 1D numpy.ndarray"):
        pyqc.calc_indicator_correlation(
            np.array([[1, 0, 1], [1, 1, 1]]),
            np.array([-1, 0, 1]),
            prob=0.5,
        )

    with pytest.raises(
        ValueError, match="`a_dataset` and `b_dataset` have to have the same shape"
    ):
        pyqc.calc_indicator_correlation(
            np.array([1, 0, 1, 1]),
            np.array([-1, 0, 1]),
            prob=0.5,
        )

    npt.assert_almost_equal(
        pyqc.calc_indicator_correlation(
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
        pyqc.calc_indicator_correlation(
            np.array([np.nan, np.nan]),
            np.array([np.nan, 1]),
            prob=0.5,
        )
