"""scratchpad for testing and implementing pwspyqc-functions."""

import numpy as np
import scipy
import tqdm
import xarray as xr


def _indicator_correlation(
    a_dataset, b_dataset, prob, exclude_nan=True, min_valid_overlap=None
):
    """Calculate indicator correlation two datasets.

    Parameters
    ----------
    a_dataset : np.array
        First data vector
    b_dataset : np.array
        Second data vector
    perc : float
        Percentile threshold
    exclude_nan : bool
        Default True, exculdes pairs where a least one value is NaN
    min_valid_overlap : int
        Minimum number of required overlapping data for calculating the
        indicator correlation

    Returns
    -------
    indicator correlation values [np.array]
    """
    if len(a_dataset.shape) != 1:
        msg = "`a_dataset` has to be a 1D numpy.ndarray"
        raise ValueError(msg)
    if a_dataset.shape != b_dataset.shape:
        msg = "`a_dataset` and `b_dataset` have to have the same shape"
        raise ValueError(msg)

    if (a_dataset < 0).any() or (b_dataset < 0).any():
        msg = "input arrays must not contain negative values"
        raise ValueError(msg)

    a_dataset = np.copy(a_dataset)
    b_dataset = np.copy(b_dataset)

    both_not_nan = ~np.isnan(a_dataset) & ~np.isnan(b_dataset)
    if exclude_nan:
        a_dataset = a_dataset[both_not_nan]
        b_dataset = b_dataset[both_not_nan]

    if min_valid_overlap is not None:
        if sum(both_not_nan) < min_valid_overlap:
            return np.nan
    else:  # noqa: PLR5501
        if sum(both_not_nan) == 0:
            msg = "No overlapping data. Define `min_valid_overlap` to return NaN in such cases."  # noqa: E501
            raise ValueError(msg)

    # Get index at quantile threshold `prob`
    ix = int(a_dataset.shape[0] * prob)

    # Set values below quantile threshold `prob` to 0
    # and above to 1
    a_sort = np.sort(a_dataset)
    b_sort = np.sort(b_dataset)
    a_dataset[a_dataset < a_sort[ix]] = 0
    b_dataset[b_dataset < b_sort[ix]] = 0
    a_dataset[a_dataset > 0] = 1
    b_dataset[b_dataset > 0] = 1

    # Calculate correlation of 0 and 1 time series
    cc = np.corrcoef(a_dataset, b_dataset)[0, 1]
    return cc  # noqa: RET504


def indicator_distance_matrix(
    da_a,
    da_b,
    max_distance=50000,  # this is in meters, assuming the projection units are also meters # noqa: E501
    prob=0.99,
    exclude_nan=True,
    min_valid_overlap=None,
):
    """Calculate indicator correlation and distance between reference and test stations.

    Parameters
    ----------
    da_a : xr.DataArray
        First data vector, has to be in the OpensSense data format standards [1]
        with rainfall as variable
    da_b : xr.DataArray
        Second data vector, has to be in the OpensSense data format standards [1]
        with rainfall as variable
    max_distance : int
        Maximum distance in meters for which the indicator correlation is returned
    prob : float
        Percentile threshold for indicator correlation
    exclude_nan : bool
        Default True, exculdes pairs where a least one value is NaN
    min_valid_overlap : int
        Minimum number of overlapping data for calculating the indicator correlation

    Literature
    ----------
    [1] Fencl M, Nebuloni R, C. M. Andersson J et al. Data formats and standards for
    opportunistic rainfall sensors [version 2; peer review: 2 approved].
    Open Res Europe 2024, 3:169 (https://doi.org/10.12688/openreseurope.16068.2)

    Returns
    -------
    Indicator correlation and distance matrices as tuple [xr.DataArray, xr.DataArray]

    """
    xy_a = list(zip(da_a.x.data, da_a.y.data, strict=False))
    xy_b = list(zip(da_b.x.data, da_b.y.data, strict=False))
    dist_mtx = scipy.spatial.distance.cdist(xy_a, xy_b, metric="euclidean")
    indcorr_mtx = np.full_like(dist_mtx, np.nan)
    # list_corr = []
    # list_dist = []
    for i in tqdm.tqdm(range(len(xy_a))):
        for j in range(len(xy_b)):
            # check if distance between stations is less than max_distance
            if dist_mtx[i, j] < max_distance:
                ts_a = da_a.isel(id=i)
                ts_b = da_b.isel(id=j)
                ts_b = ts_b.reindex({"time": ts_a.time})

                indcorr_mtx[i, j] = _indicator_correlation(
                    ts_a.data,
                    ts_b.data,
                    prob=prob,
                    exclude_nan=exclude_nan,
                    min_valid_overlap=min_valid_overlap,
                )
    da_dist_mtx = xr.DataArray(
        data=dist_mtx,
        dims=("id", "id_neighbor"),
        coords={
            "id": ("id", da_a.id.data),
            "id_neighbor": ("id_neighbor", da_b.id.data),
        },
    )
    da_indcorr_mtx = xr.DataArray(
        data=indcorr_mtx,
        dims=("id", "id_neighbor"),
        coords={
            "id": ("id", da_a.id.data),
            "id_neighbor": ("id_neighbor", da_b.id.data),
        },
    )
    return da_dist_mtx, da_indcorr_mtx


def ic_filter(
    indicator_correlation_matrix_ref,
    distance_correlation_matrix_ref,
    indicator_correlation_matrix,
    distance_matrix,
    max_distance=20e3,
    bin_size=1e3,
    quantile_bin_ref=0.1,
    quantile_bin_pws=0.5,
    threshold=0.01,
):
    """Apply indicator correlation filter [1].

    This function applies a modified version of the indicator correlation filter from
    Bárdossy el al. (2021). [1]

    Parameters
    ----------
    indicator_correlation_matrix_ref: xr.DataArray
        Indicator correlation matrix between reference stations (REF)
    distance_correlation_matrix_ref: xr.DataArray
        Distance matrix between reference stations (REF)
    indicator_correlation_matrix: xr.DataArray
        Indicator correlations matrix between REF and PWS
    distance_matrix: xr.DataArray
        Distance matrix between REF and PWS
    max_distance: int or float
        Range in meters for which the indicator correlation is evaluated
    bin_size: int or float
        Bin size in meters. This bin size is used to group data for the
        quantile calculation
    quantile_bin_ref: float
        Quantile for acceptance level based on reference data indicator correlation
    quantile_bin_pws: float
        Quantile of PWS data indicator correlation that has to be
        above `quantile_bin_ref` + `threshold`
    threshold: float
        Indicator correlation threshold below `quantile_bin_ref` where PWS are
        still accepted

    Literature
    ----------
    [1] Bárdossy, A., Seidel, J., and El Hachem, A.: The use of personal weather station
    observations to improve precipitation estimation and interpolation,
    Hydrol. Earth Syst. Sci., 25, 583-601,
    https://doi.org/10.5194/hess-25-583-2021, 2021.

    Returns
    -------
    distance matrix between PWS and reference stations [xr.DataArray]
    indicator correlation matrix between PWS and reference stations [xr.DataArray]
    boolean if station got accepted [xr.DataArray]
    indicator correlation score [xr.DataArray]
    """
    bins = np.arange(0, max_distance, bin_size)

    # quantile parameter not too low, otherwise the line becomes to wiggly - depends on data  # noqa: E501
    binned_indcorr_ref = (
        indicator_correlation_matrix_ref.groupby_bins(
            distance_correlation_matrix_ref, bins=bins
        )
        .quantile(quantile_bin_ref)
        .bfill(dim="group_bins")
    )

    # Function for Rank Sum Weights
    # Calculates weights according to length to data set
    def rsw(m):
        alphas = []  # Leere Liste
        for i in range(1, m + 1):  # Iteration über m Alternativen
            alpha = (m + 1.0 - i) / sum(range(1, m + 1))
            alphas.append(alpha)
        return alphas

    pws_indcorr_good_list = []
    pws_indcorr_score_list = []

    # iterates over REF (id)
    for pws_id in indicator_correlation_matrix["id_neighbor"].values:  # noqa: PD011
        binned_indcorr_pws = (
            indicator_correlation_matrix.sel(id_neighbor=pws_id)
            .groupby_bins(distance_matrix.sel(id_neighbor=pws_id), bins=bins)
            .quantile(quantile_bin_pws, skipna=True)
        )

        indcorr_good = binned_indcorr_pws + threshold > binned_indcorr_ref

        # Bool Information if PWS passed Indicator Correlation Test
        pws_indcorr_good_list.append(indcorr_good.any())

        # Valid bins for normed weights
        valid_bins = np.isfinite(binned_indcorr_pws.values)
        rank_sum_weights = rsw(len(indcorr_good))
        normed_weights = sum(valid_bins * np.array(rank_sum_weights))

        score = sum(indcorr_good.values * np.array(rank_sum_weights)) / normed_weights  # noqa: PD011
        pws_indcorr_score_list.append(score)

    result = indicator_correlation_matrix.to_dataset(name="indcorr")
    result["dist"] = distance_matrix
    result["indcorr_good"] = ("id_neighbor", pws_indcorr_good_list)
    result["indcorr_score"] = ("id_neighbor", pws_indcorr_score_list)

    return result
