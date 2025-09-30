"""Functions for bias correting PWS data."""
import numpy as np
from scipy import stats, optimize

def fit_gamma_with_threshold(
    data: np.array,
    threshold: float,
    )  -> tuple [float, float, float]:
    #to implement: Minimum data length required for calculating parameters

    """Function to fit a censored gamma distribution .

    Parameters
    ----------
    data : np.array
        Data for which the theretical distribution function is fitted
    threshold : float
        Threshold for censoring data
    

    Returns
    -------
    parameters for theoretical gamma function and pO [float, float, float]
    """

    data_valid = data[~np.isnan(data)]
    raindata = data_valid[data_valid>0]
    raindata_trs = raindata[raindata > threshold]

    # First quick check for succient data length, needs
    # to be adapted and modiefied!
    # Return nan if data_valid.shape[0] below threshold
    if data_valid.shape[0] < 1:
        return np.nan, np.nan, np.nan

    # calculate probabilty of no rainfall (p0)
    p0 = 1- raindata.shape[0]/data_valid.shape[0]
    # statistics for inital guesses
    raindata_mean = np.mean(raindata)
    raindata_variance = np.var(raindata)

    if raindata_mean <= 0 or raindata_variance <= 0:
        raise ValueError("Invalid data: mean and variance must be positive.")

    # inital guess for parameters
    b_init = raindata_variance / raindata_mean
    a_init = raindata_mean / b_init
    initial_guess = [a_init, b_init]
    #
    def negative_log_likelihood(params):
        aa, bb = params

        if aa <= 0 or bb <= 0:
            return np.inf


        y = stats.gamma.pdf(raindata_trs, a=aa, loc=0, scale=bb)
        pa = stats.gamma.cdf(threshold, a=aa, loc=0, scale=bb)

        if np.any(y <= 0) or pa <= 0:
            return np.inf

        ltest = np.sum(np.log(y)) + np.log(pa) * (raindata.shape[0] - raindata_trs.shape[0])
        return -ltest

    # Compute initial guess from method of moments
    result = optimize.minimize(
        negative_log_likelihood,
        initial_guess,
        bounds=[(1e-5, None), (1e-5, None)]
    )

    if result.success:
        a_opt, b_opt = result.x
        return a_opt, b_opt, p0
    else:
        raise RuntimeError(f"Optimization failed: {result.message}")





#import scipy.stats as stats

def qq_gamma(pws_val, shape_pws, scale_pws, p0_pws, shape_ref, scale_ref, p0_ref):
    """
    Function for bias correction of PWS data using quantile mapping with 
    parameters (scale, shape) form theoretical gamma function and censor values 
    form probability of precipitaion occurence, i.e. values > 0 (p0)

    Parameters:
    - pws_input: Input value to transform (only if > 0)
    - shape_pws, scale_pws, p0_pws: Parameters of the original gamma distribution
    - shape_ref, scale_ref, p0_ref: Parameters of the target gamma distribution

    Returns:
    - Transformed value if pws_val > 0 (or threshhold to be set)
    """
    result = pws_val.copy()  # Copy of the original array
    mask = pws_val > 0
    # Only compute for values > 0
    pws_qt = p0_pws + (1 - p0_pws) * stats.gamma.cdf(pws_val[mask], a=shape_pws, scale=scale_pws)
    pws_bcorr = stats.gamma.ppf((pws_qt - p0_ref) / (1 - p0_ref), a=shape_ref, scale=scale_ref)
    result[mask] = pws_bcorr
    return result
