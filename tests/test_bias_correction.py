import numpy as np
import numpy.testing as npt
import pytest
#
from pypwsqc import bias_correction


def test_data_preprocessing_empty_array():
    data = np.array([])
    threshold = 0.1
    result = bias_correction._data_preprocessing(data, threshold)
    assert all(np.isnan(x) for x in result)


def test_data_preprocessing_no_variance():
    data = np.array([0, 0, 0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0])
    threshold = 0.1
    with pytest.raises(ValueError, match="Invalid data: no variance in your data!"):
        bias_correction._data_preprocessing(data, threshold)


def test_data_preprocessing_too_little_data_above_threshold():
    data = np.array([0, 0, 0, 0.3, 0.5, 0.9, 0, 0, 0, 0.1])
    threshold = 0.1
    result = bias_correction._data_preprocessing(data, threshold)
    assert all(np.isnan(x) for x in result)


def test_data_preprocessing_p0_correct():
    data = np.array([0, 0, 0, 0, 1, 0.5, 0.8, 1, 1.6, 2, 0, 0, 0, 0, 0, 0, 0, 0])
    threshold = 0.1
    result = bias_correction._data_preprocessing(data, threshold)
    assert np.isclose(result[3], 0.66, atol=0.01)


def test_data_preprocessing_output_number_type_correct():
    data = np.array([0, 0, 0, 0, 1, 0.5, 0.8, 2, 0.3, 0, 0, 0])
    threshold = 0.1
    result = bias_correction._data_preprocessing(data, threshold)
    assert isinstance(result[0], list)
    assert isinstance(result[1], np.ndarray)
    assert isinstance(result[2], np.ndarray)
    assert isinstance(result[3], float)


def test_negative_log_likelihood_invalid_params():
    initial_guess = [0, 0.5]
    threshold = 0.1
    raindata = np.array(())
    raindata_trs = np.array(())
    with pytest.raises(ValueError, match="Gamma Dist not defined for params == 0!"):
        bias_correction._negative_log_likelihood(
            initial_guess, threshold, raindata, raindata_trs
        )


def test_fit_gamma_with_threshold_too_little_data():
    threshold = 0.1
    data = np.array([0, 0, 0, 0, 1, 0.5, 0.8])
    result = bias_correction.fit_gamma_params_with_threshold(data, threshold)
    assert all(np.isnan(x) for x in result)


def test_fit_gamma_with_threshold_raises_invalid_variance():
    threshold = 0.1
    data = np.array([0, 0, 0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0])
    result = bias_correction.fit_gamma_params_with_threshold(data, threshold)
    assert all(np.isnan(x) for x in result)


def test_fit_gamma_with_threshold_consistency():
    rng = np.random.default_rng(42)
    shape_true, scale_true = 0.8, 2.0
    data = rng.gamma(shape_true, scale_true, size=50)
    threshold = 0.1

    a, b, p0 = bias_correction.fit_gamma_params_with_threshold(data, threshold)

    assert np.isfinite(a)
    assert a > 0
    assert np.isfinite(b)
    assert b > 0
    assert 0 <= p0 < 1


def test_fit_gamma_with_threshold_params_compared_method_moments():
    rng = np.random.default_rng(42)
    shape_true, scale_true = 0.8, 2.0
    threshold = 0.001

    # Small synthetic sample, still enough for fitting
    data = rng.gamma(shape_true, scale_true, size=500)

    # Fit the censored gamma
    a_fit, b_fit, p0_fit = bias_correction.fit_gamma_params_with_threshold(
        data, threshold
    )

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
    data_arr = np.array([0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0])
    expected = np.array([0, 0, 0, 0, 0, 0.0813236, 0, 0, 0, 0])
    shape_data, scale_data, p0_data = 0.89, 3.7, 0.92
    shape_ref, scale_ref, p0_ref = 0.87, 3.3, 0.92
    output = bias_correction.qq_gamma(
        data_arr, shape_data, scale_data, p0_data, shape_ref, scale_ref, p0_ref
    )
    npt.assert_array_almost_equal(output, expected, decimal=5)


def test_qq_gamma_identical_params_case():
    data_arr = np.array([0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0])
    shape_data, scale_data, p0_data = 0.89, 3.7, 0.92
    output = bias_correction.qq_gamma(
        data_arr,
        shape_data,
        scale_data,
        p0_data,
        shape_data,
        scale_data,
        p0_data,
    )
    npt.assert_array_almost_equal(output, data_arr, decimal=5)


def test_qq_gamma_number_of_zeroes_equal():
    data_arr = np.array([0, 0, 0, 0.3, 0.2, 0, 0, 0.4])
    shape_data, scale_data, p0_data = 1, 2.3, 0.625
    shape_ref, scale_ref, p0_ref = 1.2, 2.5, 0.7
    number_zero_data = np.sum(data_arr == 0)
    output = bias_correction.qq_gamma(
        data_arr, shape_data, scale_data, p0_data, shape_ref, scale_ref, p0_ref
    )
    number_zero_output = np.sum(output == 0)
    assert number_zero_data == number_zero_output
