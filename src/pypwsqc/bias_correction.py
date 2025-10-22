"""Functions for bias correcting PWS data."""

import numpy as np
from scipy import optimize, stats


def _data_preprocessing(
    data: np.array,
    threshold: float,
) -> tuple[list[float, float], np.array, np.array, float]:
    """
    Prepare input from original data.

    Prepare the input for the _negative_log_likelihood and
    fit_gamma_params_with_threshold functions from raw data.

    Parameters
    ----------
    data : np.array
        Raw data to which the theoretical distribution function is fitted
    threshold : float
        Threshold for censoring data

    Returns
    -------
    initial_guess: [a_init, b_init]
        Method of Moment guess for shape and scale parameters of the gamma distribution
    raindata: np.array
        Data larger than zero
    raindata_trs: np.array
        Data larger than threshold
    p0: float
        Probability of no rainfall
    """
    data_valid = data[~np.isnan(data)]
    raindata = data_valid[data_valid > 0]
    raindata_trs = raindata[raindata > threshold]

    # First quick check for sufficient data length
    if data_valid.shape[0] < 10 or raindata_trs.shape[0] < 5:
        return np.nan, np.nan, np.nan, np.nan

    p0 = np.mean(data_valid <= 0)

    # statistics for initial guesses
    raindata_mean = np.mean(raindata)
    raindata_variance = np.var(raindata)

    if raindata_variance == 0:
        msg = "Invalid data: no variance in your data!"
        raise ValueError(msg)

    # initial guess for parameters (Method of Moments)
    b_init = raindata_variance / raindata_mean
    a_init = raindata_mean / b_init
    initial_guess = [a_init, b_init]

    return initial_guess, raindata, raindata_trs, p0


def _negative_log_likelihood(initial_guess, threshold, raindata, raindata_trs):
    """
    Define the log-likelihood function.

     Define the log-likelihood function which will be optimized in the
     fit_gamma_params_with_threshold function.

    Parameters
    ----------
    initial_guess: [a_init, b_init]
        Method of Moments guess for shape and scale parameters of the gamma distribution
    threshold : float
        Threshold for censoring data
    raindata: np.array
        Data larger than zero
    raindata_trs: np.array
        Data larger than threshold

    Returns
    -------
    float:
        The negative log-likelihood value for initial guess

    """
    aa, bb = initial_guess

    if aa == 0 or bb == 0:
        msg = "Invalid parameter(s): Gamma Dist. not defined for shape or scale == 0!"
        raise ValueError(msg)

    y = stats.gamma.pdf(raindata_trs, a=aa, loc=0, scale=bb)
    pa = stats.gamma.cdf(threshold, a=aa, loc=0, scale=bb)

    ltest = np.sum(np.log(y)) + np.log(pa) * (raindata.shape[0] - raindata_trs.shape[0])
    return -ltest


def fit_gamma_params_with_threshold(
    data,
    threshold,
) -> tuple[float, float, float]:
    """Fit a censored gamma distribution to rainfall data.

    Parameters
    ----------
    data : np.array
        Data for which the theoretical gamma distribution is fitted
    threshold : float
        Threshold for censoring data (detection threshold)

    Returns
    -------
    a_opt: float
        shape parameter for the gamma distribution
    b_opt: float
        scale parameter for the gamma distribution
    p0: float
        Probability of no rainfall
    """
    try:
        initial_guess, raindata, raindata_trs, p0 = _data_preprocessing(data, threshold)
    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return np.nan, np.nan, np.nan

    # Handle failed preprocessing
    if any(x is np.nan for x in [initial_guess, raindata, raindata_trs, p0]):
        return np.nan, np.nan, np.nan

    # Compute initial guess from method of moments
    result = optimize.minimize(
        lambda params: _negative_log_likelihood(
            params, threshold, raindata, raindata_trs
        ),
        initial_guess,
        bounds=[(1e-5, None), (1e-5, None)],
    )

    if result.success:
        a_opt, b_opt = result.x
        return a_opt, b_opt, p0
    return (
        np.nan,
        np.nan,
        np.nan,
    )


def qq_gamma(
    data: np.array,
    shape_input: float,
    scale_input: float,
    p0_input: float,
    shape_ref: float,
    scale_ref: float,
    p0_ref: float,
) -> np.array:
    """
    Map quantiles of PWS data with parameters.

    Map quantiles of PWS data with parameters (scale, shape) from theoretical gamma
    function and censor values from threshold of probability of precipitation
    occurrence, i.e. values > 0 (p0).

    Parameters
    ----------
    data : np.array
        Data to be bias corrected
    shape_input: float
        Shape parameter of input data
    scale_input : float
        Scale parameter of input data
    p0_input : float
        P0 of input data
    shape_ref : float
        Shape parameter of reference data
    scale_ref : float
        Scale parameter of reference data
    p0_ref : float
        P0 of reference data


    Returns
    -------
    np.array
        Bias corrected data
    """
    result = data.copy()
    mask = data > 0
    # Only compute for values > 0
    data_qt = p0_input + (1 - p0_input) * stats.gamma.cdf(
        data[mask], a=shape_input, scale=scale_input
    )
    data_bcorr = stats.gamma.ppf(
        (data_qt - p0_ref) / (1 - p0_ref), a=shape_ref, scale=scale_ref
    )
    result[mask] = data_bcorr
    return result
