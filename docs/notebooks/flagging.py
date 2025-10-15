"""Functions for flagging problematic time steps in PWS time series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def fz_filter(
    ds_pws,
    nint,
    n_stat,
    distance_matrix,
    max_distance=10e3,
):
    """Faulty Zeros Filter.

    This function applies the FZ filter from the R package PWSQC.
    The flag 1 means, a faulty zero has been detected. The flag -1
    means that no flagging was done because evaluation cannot be
    performed for the first `nint` values.

    Note that this code here is derived from the Python translation,
    done by Niek van Andel, of the original R code from Lotte de Vos.
    The Python code stems from here https://github.com/NiekvanAndel/QC_radar.
    Also note that the correctness of the Python code has not been
    verified and not all feature of the R implementation might be there.

    Parameters
    ----------
    pws_data
        The rainfall time series of the PWS that should be flagged
    reference
        The rainfall time series of the reference, which can be e.g.
        the median of neighboring PWS data.
    nint : optional
        The number of subsequent data points which have to be zero, while
        the reference has values larger than zero, to set the flag for
        this data point to 1.

    Returns
    -------
    xarray.Dataset
        time series of flags
    """
    # calculate support variables
    if "reference" not in ds_pws:
        ds_pws["reference"], ds_pws["nbrs_not_nan"] = _calc_reference_and_nbrs_not_nan(
            ds_pws, distance_matrix, max_distance
        )

    pws_data = ds_pws.rainfall
    nbrs_not_nan = ds_pws.nbrs_not_nan
    reference = ds_pws.reference

    # initialize arrays
    sensor_array = np.zeros_like(pws_data)
    ref_array = np.zeros_like(pws_data)
    fz_array = np.zeros_like(pws_data)

    # Wet timestep at each station
    sensor_array[np.where(pws_data > 0)] = 1

    # Dry timestep at each station
    sensor_array[np.where(pws_data == 0)] = 0

    # Wet timesteps of the reference
    ref_array[np.where(reference > 0)] = 1

    for i in np.arange(len(pws_data.id.data)):
        for j in np.arange(len(pws_data.time.data)):
            if j < nint:
                fz_array[i, j] = -1
            elif sensor_array[i, j] > 0:
                fz_array[i, j] = 0
            elif fz_array[i, j - 1] == 1:
                fz_array[i, j] = 1
            elif (np.sum(sensor_array[i, j - nint : j + 1]) > 0) or (
                np.sum(ref_array[i, j - nint : j + 1]) < nint + 1
            ):
                fz_array[i, j] = 0
            else:
                fz_array[i, j] = 1

    fz_array = fz_array.astype(int)
    fz_flag = xr.where(nbrs_not_nan < n_stat, -1, fz_array)

    # add to dataset
    ds_pws["fz_flag"] = fz_flag

    # check if last nint timesteps are NaN in rolling window
    nan_in_last_nint = (
        ds_pws["rainfall"].rolling(time=nint, center=True).construct("window_dim")
    )
    all_nan_in_window = nan_in_last_nint.isnull().all(dim="window_dim")

    # Apply the mask to set fz_flag to -1 where the condition is met
    ds_pws["fz_flag"] = ds_pws["fz_flag"].where(~all_nan_in_window, -1)

    return ds_pws


def hi_filter(
    ds_pws,
    hi_thres_a,
    hi_thres_b,
    nint,
    n_stat,
    distance_matrix,
    max_distance=10e3,
):
    """High Influx filter.

    This function applies the HI filter from the R package PWSQC,
    flagging unrealistically high rainfall amounts.

    The Python code has been translated from the original R code,
    to be found here: https://github.com/LottedeVos/PWSQC/tree/master/R.

    The function returns an array with zeros, ones or -1 per time step
    and station.
    The flag 0 means that no high influx has been detected.
    The flag 1 means that high influx has been detected.
    The flag -1 means that no flagging was done because not enough
    neighbouring stations are reporting rainfall to make a reliable
    evaluation.

    Parameters
    ----------
    pws_data
        The rainfall time series of the PWS that should be flagged
    nbrs_not_nan
        Number of neighbouring stations reporting rainfall
    reference
        The rainfall time series of the reference, which can be e.g.
        the median of neighboring stations
    hi_thres_a
        threshold for median rainfall of neighbouring stations [mm]
    hi_thres_b
        upper rainfall limit [mm]
    n_stat
        threshold for number of neighbours reporting rainfall

    Returns
    -------
    xarray.Dataset
        time series of flags
    """
    # calculate support variables
    if "reference" not in ds_pws:
        ds_pws["reference"], ds_pws["nbrs_not_nan"] = _calc_reference_and_nbrs_not_nan(
            ds_pws, distance_matrix, max_distance
        )

    condition1 = (ds_pws.reference < hi_thres_a) & (ds_pws.rainfall > hi_thres_b)
    condition2 = (ds_pws.reference >= hi_thres_a) & (
        ds_pws.rainfall > ds_pws.reference * hi_thres_b / hi_thres_a
    )

    hi_array = (condition1 | condition2).astype(int)

    hi_flag = xr.where(ds_pws.nbrs_not_nan < n_stat, -1, hi_array)

    # add to dataset
    ds_pws["hi_flag"] = hi_flag

    # check if last nint timesteps are NaN in rolling window
    nan_in_last_nint = (
        ds_pws["rainfall"].rolling(time=nint, center=True).construct("window_dim")
    )
    all_nan_in_window = nan_in_last_nint.isnull().all(dim="window_dim")

    # Apply the mask to set hi_flag to -1 where the condition is met
    ds_pws["hi_flag"] = ds_pws["hi_flag"].where(~all_nan_in_window, -1)

    return ds_pws


def _so_filter_one_station(ds_station, ds_neighbors, evaluation_period, mmatch):
    """Support function to Station Outlier filter.

    Parameters
    ----------
    da_station
        rainfall time series of evaluated station.
    da_neighbors
        rainfall time series of neighboring stations.
    evaluation_period
        length of (rolling) window for correlation calculation
        [timesteps]
    mmatch
        threshold for number of matching rainy intervals in
        evaluation period [timesteps]

    Returns
    -------
    xarray.Dataset
        number of neighbors with enough wet time steps
    """
    # rolling pearson correlation
    s_rainfall = ds_station.rainfall.to_series()
    s_neighbors_rain = ds_neighbors.rainfall.to_series()
    corr = s_rainfall.rolling(evaluation_period, min_periods=1).corr(s_neighbors_rain)
    ds = xr.Dataset.from_dataframe(pd.DataFrame({"corr": corr}))

    # create dataframe of neighboring stations
    df_nbrs = ds_neighbors.to_dataframe()
    df_nbrs = df_nbrs["rainfall"].unstack("id")  # noqa: PD010

    # boolean arrays - True if a rainy time step, False if 0 or NaN.
    rainy_timestep_at_nbrs = df_nbrs > 0

    # rolling sum of number of rainy timesteps in
    # last evaluation_period period, per neighbor.
    wet_timesteps_last_evaluation_period_period = rainy_timestep_at_nbrs.rolling(
        evaluation_period, min_periods=1
    ).sum()

    # per time step and neighbor, does the nbr have more than
    # mmatch wet time steps in the last evaluation_period period? (true/false)
    enough_matches_per_nbr = wet_timesteps_last_evaluation_period_period > mmatch

    # summing how many neighbors that have enough matches per time step
    nr_nbrs_with_enough_matches = enough_matches_per_nbr.sum(axis=1)

    ds["matches"] = xr.DataArray.from_series(nr_nbrs_with_enough_matches)

    return ds


def so_filter(
    ds_pws,
    evaluation_period,
    mmatch,
    gamma,
    n_stat,
    distance_matrix,
    max_distance=10e3,
    bias_corr=False,
    beta=0.2,
    dbc=1,
):
    """Station Outlier filter.

    This function applies the SO filter from the R package PWSQC,
    flagging nonsensical rainfall measurements for a specific location.

    The Python code has been translated from the original R code,
    to be found here: https://github.com/LottedeVos/PWSQC/tree/master/R.

    In its original implementation, any interval with at least `mrain`
    intervals of nonzero rainfall measurements is evaluated.
    In this implementation, only a fixed rolling window of `evaluation_period`
    intervals is evaluated.

    The function returns an array with zeros, ones or -1 per time step
    and station.
    The flag 0 means that no station outlier has been detected.
    The flag 1 means that a station outlier has been detected.
    The flag -1 means that no flagging was done because not enough
    neighbouring stations are reporting rainfall to make a reliable
    evaluation or that the previous evaluation_period time steps was dry.

    The function also has the option to calculate a bias correction factor
    per time step.

    Parameters
    ----------
    ds_pws
        xarray data set
    nbrs_not_nan
        Number of neighbouring stations reporting rainfall
    evaluation_period
        length of (rolling) window for correlation calculation
        [timesteps]
    mmatch
        threshold for number of matching rainy intervals in
        evaluation period [timesteps]
    gamma
        threshold for rolling median pearson correlation [-]
    n_stat
        threshold for number of neighbours reporting rainfall
    distance_matrix
        matrix with distances between all stations in the data set
    max_distance
        considered range around each station [m]
    bias_corr (optional)
        boolean to decide if bias correction factor will be calculated. Default False
    beta
        bias correction parameter. Default 0.2
    dbc
        default bias correction factor. Default 1

    Returns
    -------
    xarray.Dataset
        Time series of flags.
    """
    # calculate support variables
    if "reference" not in ds_pws:
        ds_pws["reference"], ds_pws["nbrs_not_nan"] = _calc_reference_and_nbrs_not_nan(
            ds_pws, distance_matrix, max_distance
        )

    # For each station (ID), get the index of the first non-NaN rainfall value
    first_non_nan_index = ds_pws["rainfall"].notnull().argmax(dim="time")

    # initialize
    ds_pws["so_flag"] = xr.DataArray(
        np.ones((len(ds_pws.id), len(ds_pws.time))) * -999, dims=("id", "time")
    )
    ds_pws["median_corr_nbrs"] = xr.DataArray(
        np.ones((len(ds_pws.id), len(ds_pws.time))) * -999, dims=("id", "time")
    )

    for i in range(len(ds_pws.id)):
        ds_station = ds_pws.isel(id=i)
        pws_id = ds_station.id.to_numpy()

        # picking stations within max_distnance, excluding itself,
        # for the whole duration of the time series
        neighbor_ids = distance_matrix.id.data[
            (distance_matrix.sel(id=pws_id) < max_distance)
            & (distance_matrix.sel(id=pws_id) > 0)
        ]

        # if there are no neighbors, continue
        if len(neighbor_ids) == 0:
            ds_pws["so_flag"].loc[{"id": pws_id}] = -1
            continue

        # create data set for neighbors
        ds_neighbors = ds_pws.sel(id=neighbor_ids)

        # run so-filter
        ds_so_filter = _so_filter_one_station(
            ds_station, ds_neighbors, evaluation_period, mmatch
        )

        # calculate median correlation with nbrs, per time step
        median_correlation = ds_so_filter.corr.median(dim="id", skipna=True)
        ds_pws.median_corr_nbrs[i] = median_correlation

        so_array = (median_correlation < gamma).astype(int)

        # filter can not be applied if less than n_stat neighbors have enough matches
        ds_pws["so_flag"][i] = xr.where(ds_so_filter.matches < n_stat, -1, so_array)

        # find first valid time
        first_valid_time = first_non_nan_index[i].item()

        # disregard warm up period
        ds_pws["so_flag"].isel(id=i).loc[
            {
                "time": ds_pws.time[
                    first_valid_time : first_valid_time + evaluation_period
                ]
            }
        ] = -1

        if bias_corr:
            # run bias correction
            ds_pws = _calc_bias_corr_factor(
                ds_pws,
                evaluation_period,
                distance_matrix,
                max_distance,
                beta,
                dbc,
            )

    return ds_pws


def bias_correction(
    ds_pws,
    evaluation_period,
    distance_matrix,
    max_distance=10e3,
    beta=0.2,
    dbc=1,
):
    """Bias Correction Factor (BCF) Calculation.

    This function applies the BCF calculation from the R package PWSQC.

    The Python code has been translated from the original R code,
    to be found here: https://github.com/LottedeVos/PWSQC/tree/master/R.

    In its original implementation, the functionality is embedded in the
    Station Outlier filter. Here, the bias correction can be performed
    separately. It is recommended to apply the other QC filters first
    and only calculate BCF on filtered data.

    The default is to use the median rainfall of the neighboring stations
    as reference. To use another data source as reference, that data must
    be added as a variable named `reference` to the xarray data set.

    The function returns an array BCF values per station and time step.

    Parameters
    ----------
    ds_pws
        xarray data set
    nbrs_not_nan
        Number of neighbouring stations reporting rainfall
    evaluation_period
        length of (rolling) window for correlation calculation
        [timesteps]
    distance_matrix
        matrix with distances between all stations in the data set
    max_distance
        considered range around each station [m]
    beta
        bias correction parameter. Default 0.2
    dbc
        Start value of bias correction factor. Default 1

    Returns
    -------
    xarray.Dataset
        Time series with bias correction factors.
    """
    # calculate support variables
    if "reference" not in ds_pws:
        ds_pws["reference"], ds_pws["nbrs_not_nan"] = _calc_reference_and_nbrs_not_nan(
            ds_pws, distance_matrix, max_distance
        )

    # run bias correction
    return _calc_bias_corr_factor(
        ds_pws,
        evaluation_period,
        distance_matrix,
        max_distance,
        beta,
        dbc,
    )


def _calc_bias_corr_factor(
    ds_pws,
    evaluation_period,
    distance_matrix,
    max_distance,
    beta,
    dbc,
):
    """Support function to Bias Correction filter filter.

    Parameters
    ----------
    da_pws
        xarray data set
    evaluation_period
        rainfall time series of neighboring stations.
    distance_matrix
        matrix with distances between all stations in the
        data set.
    max_distance
        considered range around each station [m]
    beta
        bias correction parameter. Default 0.2
    dbc
        Start value of bias correction factor. Default 1

    Returns
    -------
    xarray.Dataset
        Time series with bias correction factors.
    """
    bcf_prev = dbc

    # For each station (ID), get the index of the first non-NaN rainfall value
    first_non_nan_index = ds_pws["rainfall"].notnull().argmax(dim="time")

    # initialize with default bias correction factor
    ds_pws["bias_corr_factor"] = xr.DataArray(
        np.ones((len(ds_pws.id), len(ds_pws.time))) * dbc, dims=("id", "time")
    )
    ds_pws["bcf_new"] = xr.DataArray(
        np.ones((len(ds_pws.id), len(ds_pws.time))) * -999, dims=("id", "time")
    )
    eps = 1e-6  # small number to prevent divide-by-zero

    for i in range(len(ds_pws.id)):
        bcf_prev = dbc
        ds_station = ds_pws.isel(id=i)
        pws_id = ds_station.id.to_numpy()

        # picking stations within max_distnance, excluding itself,
        # for the whole duration of the time series
        neighbor_ids = distance_matrix.id.data[
            (distance_matrix.sel(id=pws_id) < max_distance)
            & (distance_matrix.sel(id=pws_id) > 0)
        ]

        # if there are no neighbors, continue
        if len(neighbor_ids) == 0:
            ds_pws["bias_corr_factor"].loc[{id: pws_id}] = -1
            continue

        # find first valid time
        first_valid_time = first_non_nan_index[i].item()

        # disregard warm up period
        ds_pws["bias_corr_factor"].isel(id=i).loc[
            {
                "time": ds_pws.time[
                    first_valid_time : first_valid_time + evaluation_period
                ]
            }
        ] = -1

        s_rainfall = ds_station.rainfall.to_series()
        s_reference = ds_station.reference.to_series()
        diff = s_rainfall - s_reference
        mean_diff = diff.rolling(evaluation_period, min_periods=1, center=False).mean()
        mean_ref = s_reference.rolling(
            evaluation_period, min_periods=1, center=False
        ).mean()
        bias = mean_diff / mean_ref.replace(0, np.nan)

        bcf_new = 1 / (1 + bias)
        ds_pws["bcf_new"][i] = xr.DataArray.from_series(bcf_new)

        bcf_shifted = ds_pws["bcf_new"][i].shift(time=-1)

        # avoid log(<=0): replace invalid ratios with eps
        ratio = ds_pws.bcf_new[i] / bcf_prev
        safe_ratio = xr.where(ratio <= 0, eps, ratio)

        condition3 = (np.abs(np.log(safe_ratio)) > np.log(1 + beta)) & (
            ds_pws.bias_corr_factor[i] == 1
        )

        ds_pws.bias_corr_factor[i] = xr.where(
            condition3, ds_pws.bcf_new[i], bcf_shifted
        )

    return ds_pws


def _calc_reference_and_nbrs_not_nan(ds_pws, distance_matrix, max_distance):
    nbrs_not_nan = []
    reference = []
    for pws_id in ds_pws.id.data:
        neighbor_ids = distance_matrix.id.data[
            (distance_matrix.sel(id=pws_id) < max_distance)
            & (distance_matrix.sel(id=pws_id) > 0)
        ]

        nr_nbrs_not_nan = ds_pws.rainfall.sel(id=neighbor_ids).notnull().sum(dim="id")
        nbrs_not_nan.append(nr_nbrs_not_nan)

        median = ds_pws.sel(id=neighbor_ids).rainfall.median(dim="id")
        reference.append(median)

    ds_pws["nbrs_not_nan"] = xr.concat(nbrs_not_nan, dim="id")
    ds_pws["reference"] = xr.concat(reference, dim="id")

    return ds_pws.reference, ds_pws.nbrs_not_nan
