import numpy as np
import numpy.testing as npt

import pypwsqc.pwspyqc_dev_js as pyqc


def test_indicator_correlation():
    rng = np.random.default_rng()
    x = np.abs(rng.standard_normal(100))
    assert pyqc.calc_indicator_correlation(x, x, prob=0.1) == 1.0
    assert pyqc.calc_indicator_correlation(x, x * 0.7, prob=0.1) == 1.0

    assert (
        pyqc.calc_indicator_correlation(
            np.array([0, 1, 2, 3]),
            np.array([0, 2, 1, 4]),
            prob=0.75,
        )
        == 1.0
    )

    npt.assert_almost_equal(
        pyqc.calc_indicator_correlation(
            np.array([0, 1, 2, 3]),
            np.array([0, 1, 2, 1]),
            prob=0.75,
        ),
        -0.33333333333333,
    )


# def test_indicator_correlation_raise():
#     with pytest.raises(ValueError, match="input arrays must not contain negative values"):
#         pyqc.calc_indicator_correlation(
#             np.array([-1, 0, 1]),
#             np.array([-1, 0, 1]),
#         )
