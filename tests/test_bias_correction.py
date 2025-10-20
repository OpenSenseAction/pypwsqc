import numpy as np
import numpy.testing as npt
import pytest

from pypwsqc import bias_correction


def test_fit_gamma_with_threshold_empty_data():
    threshold = 0.1
    result = bias_correction.fit_gamma_with_threshold(np.array([]), threshold)
    assert all(np.isnan(x) for x in result)


def test_fit_gamma_with_threshold_p0_calculation():
    threshold = 0.1
    result = bias_correction.fit_gamma_with_threshold(
        np.array([0, 0, 0, 0, 1, 0.5, 0.8, 2, 0, 0, 0, 0]), threshold
    )
    assert isinstance(result[0], float)
    assert isinstance(result[1], float)
    assert np.isclose(result[2], 0.66, atol=0.01)


def test_fit_gamma_with_threshold_raises_invalid_mean_variance():
    threshold = 0.1
    with pytest.raises(ValueError):  # noqa: PT011
        bias_correction.fit_gamma_with_threshold(np.array([1, 1, 1]), threshold)


def test_fit_gamma_with_threshold_consistency():
    rng = np.random.default_rng(seed=42)
    shape_true, scale_true = 0.8, 2.0
    data = rng.gamma(shape=shape_true, scale=scale_true, size=50)
    threshold = 0.1

    a, b, p0 = bias_correction.fit_gamma_with_threshold(data, threshold)

    assert np.isfinite(a)
    assert a > 0
    assert np.isfinite(b)
    assert b > 0
    assert 0 <= p0 < 1


def test_fit_gamma_with_threshold_params_compared_method_moments():
    rng = np.random.default_rng(seed=123)
    shape_true, scale_true = 0.8, 2.0
    threshold = 0.001

    # Small synthetic sample, still enough for fitting
    data = rng.gamma(shape_true, scale_true, size=500)

    # Fit the censored gamma
    a_fit, b_fit, p0_fit = bias_correction.fit_gamma_with_threshold(data, threshold)

    # Positive values
    raindata = data[data > 0]
    raindata_trs = raindata[raindata > threshold]

    # Truncated method-of-moments
    mean_trs = np.mean(raindata_trs)
    var_trs = np.var(raindata_trs)
    b_mom = var_trs / mean_trs
    a_mom = mean_trs / b_mom

    # Assertions with wider tolerance for small sample
    assert np.isclose(a_fit, a_mom, rtol=0.1), f"Shape mismatch: {a_fit} vs {a_mom}"
    assert np.isclose(b_fit, b_mom, rtol=0.1), f"Scale mismatch: {b_fit} vs {b_mom}"


def test_qq_gamma_arbitrary_case():
    input_arr = np.array([0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0])
    expected = np.array([0, 0, 0, 0, 0, 0.0813236, 0, 0, 0, 0])
    shape_input, scale_input, p0_input = 0.89, 3.7, 0.92
    shape_ref, scale_ref, p0_ref = 0.87, 3.3, 0.92
    output = bias_correction.qq_gamma(
        input_arr, shape_input, scale_input, p0_input, shape_ref, scale_ref, p0_ref
    )
    npt.assert_array_almost_equal(output, expected, decimal=5)


def test_qq_gamma_identical_params_case():
    input_arr = np.array([0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0])
    shape_input, scale_input, p0_input = 0.89, 3.7, 0.92
    output = bias_correction.qq_gamma(
        input_arr,
        shape_input,
        scale_input,
        p0_input,
        shape_input,
        scale_input,
        p0_input,
    )
    npt.assert_array_almost_equal(output, input_arr, decimal=5)


def test_qq_gamma_number_of_zeroes_equal():
    input_arr = np.array([0, 0, 0, 0.3, 0.2, 0, 0, 0.4])
    shape_input, scale_input, p0_input = 1, 2.3, 0.625
    shape_ref, scale_ref, p0_ref = 1.2, 2.5, 0.7
    number_zero_input = np.sum(input_arr == 0)
    output = bias_correction.qq_gamma(
        input_arr, shape_input, scale_input, p0_input, shape_ref, scale_ref, p0_ref
    )
    number_zero_output = np.sum(output == 0)
    assert number_zero_input == number_zero_output
