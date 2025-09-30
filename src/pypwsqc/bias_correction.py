"""Functions for bias correcting PWS data."""
import numpy as np
from scipy import optimize, stats


def fit_gamma_with_threshold(
    data: np.array,
    threshold: float,
) -> tuple[float, float, float]:
    # to implement: Minimum data length required for calculating parameters
    """Fit a censored gamma distribution to rainfall data.

    Parameters
    ----------
    data : np.array
        Data for which the theoretical distribution function is fitted
    threshold : float
        Threshold for censoring data


    Returns
    -------
    shape and scale parameters for theoretical gamma function and p0 [float, float, float]
    """
    data_valid = data[~np.isnan(data)]
    raindata = data_valid[data_valid > 0]
    raindata_trs = raindata[raindata > threshold]

    # First quick check for sufficient data length, needs
    # to be adapted and modified!
    # Return nan if data_valid.shape[0] below threshold
    if data_valid.shape[0] < 1:
        return np.nan, np.nan, np.nan

    # calculate probability of no rainfall (p0)
    p0 = 1 - raindata.shape[0] / data_valid.shape[0]
    # statistics for initial guesses
    raindata_mean = np.mean(raindata)
    raindata_variance = np.var(raindata)

    if raindata_mean <= 0 or raindata_variance <= 0:
        raise ValueError("Invalid data: mean and variance must be positive.")

    # initial guess for parameters
    b_init = raindata_variance / raindata_mean
    a_init = raindata_mean / b_init
    initial_guess = [a_init, b_init]

    def negative_log_likelihood(params):
        aa, bb = params

        if aa <= 0 or bb <= 0:
            return np.inf

        y = stats.gamma.pdf(raindata_trs, a=aa, loc=0, scale=bb)
        pa = stats.gamma.cdf(threshold, a=aa, loc=0, scale=bb)

        if np.any(y <= 0) or pa <= 0:
            return np.inf

        ltest = np.sum(np.log(y)) + np.log(pa) * (
            raindata.shape[0] - raindata_trs.shape[0]
        )
        return -ltest

    # Compute initial guess from method of moments
    result = optimize.minimize(
        negative_log_likelihood, initial_guess, bounds=[(1e-5, None), (1e-5, None)]
    )

    if result.success:
        a_opt, b_opt = result.x
        return a_opt, b_opt, p0
    else:
        raise RuntimeError(f"Optimization failed: {result.message}")


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
    Function for bias correction of PWS data using quantile mapping with
    parameters (scale, shape) form theoretical gamma function and censor values
    from threshold of probability of precipitation occurrence, i.e. values > 0 (p0).

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
    Bias corrected data [np.array]
    """
    result = data.copy()  # Copy of the original array
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
