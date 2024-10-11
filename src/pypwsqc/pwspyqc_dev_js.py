"""scratchpad for testing and implementing pwspyqc-functions."""

import matplotlib.pyplot as plt
import numpy as np
import scipy
import tqdm
import xarray as xr


def calc_indicator_correlation(
    a_dataset, b_dataset, prob, exclude_nan=True, min_valid_overlap=None
):
    """Calculate indicator correlation two datasets.

    Parameters
    ----------
    a_dataset: first data vector
    b_dataset: second data vector
    perc: percentile threshold

    Returns
    -------
    indicator correlation value
    """
    if len(a_dataset.shape) != 1:
        raise ValueError("`a_dataset` has to be a 1D numpy.ndarray")
    if a_dataset.shape != b_dataset.shape:
        raise ValueError("`a_dataset` and `b_dataset` have to have the same shape")

    a_dataset = np.copy(a_dataset)
    b_dataset = np.copy(b_dataset)

    both_not_nan = ~np.isnan(a_dataset) & ~np.isnan(b_dataset)
    if exclude_nan:
        a_dataset = a_dataset[both_not_nan]
        b_dataset = b_dataset[both_not_nan]

    if min_valid_overlap is not None:
        if sum(both_not_nan) < min_valid_overlap:
            return np.nan
    else:
        if sum(both_not_nan) == 0:
            raise ValueError(
                "No overlapping data. Define `min_valid_overlap` to return NaN in such cases."
            )

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
    return cc


def calc_indic_corr_all_stns(
    da_a,
    da_b,
    max_distance=50000,  # this is in meters, assuming the projection units are also meters # noqa: E501
    prob=0.99,
    exclude_nan=True,
    min_valid_overlap=None,
):
    """Calculate indicator correlation between reference and test stations.

    return: indicator correlation and distance values

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

                indcorr_mtx[i, j] = calc_indicator_correlation(
                    ts_a.data,
                    ts_b.data,
                    prob=prob,
                    exclude_nan=exclude_nan,
                    min_valid_overlap=min_valid_overlap,
                )
                # list_dist.append(dist_mtx[i, j])
                # list_corr.append(indi_corr)
    # dist_vals = np.asarray(list_dist)
    # corr_vals = np.asarray(list_corr)

    # Dimensionen benennen! StastionsID Als xarray.Darray
    # dist_vals=xr.DataArray(dist_vals, coords={'id' : da_a.id , 'id_neighbor' : da_b.id }) # noqa: E501
    # corr_cals=.....
    # return dist_vals, corr_vals
    # return dist_mtx, indcorr_mtx
    return (
        xr.DataArray(
            data=dist_mtx,
            dims=("id", "id_neighbor"),
            coords={
                "id": ("id", da_a.id.data),
                "id_neighbor": ("id_neighbor", da_b.id.data),
            },
        ),
        xr.DataArray(
            data=indcorr_mtx,
            dims=("id", "id_neighbor"),
            coords={
                "id": ("id", da_a.id.data),
                "id_neighbor": ("id_neighbor", da_b.id.data),
            },
        ),
    )


def indicator_correlation_filter(
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
    """Apply indicator correlation filer to filter out PWS that do not match the correlation structure of reference data set.

    Parameters
    ----------
    indicator_correlation_matrix_ref: xr.DataArray with indicator correlation matrix between reference stations (REF)
    distance_correlation_matrix_ref: xr.DataArray with distance matrix between reference stations (REF)
    indicator_correlation_matrix: xr.DataArray with indicator correlations matrix between REF and PWS
    distance_matrix: xr.DataArray with distance matrix between REF and PWS
    range: range in meters for which the indicator correlation is evaluated
    bin_size: bin size in meters
    acceptance_level: quantile for acceptance level of reference indicator correlation
    threshold: indicator correlation threshold below acceptance level where PWS are still accepted

    Returns
    -------
    boolean if station got accepted
    indicator correlation score
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

        IndCorrGood = binned_indcorr_pws + threshold > binned_indcorr_ref

        # Bool Information if PWS passed Indicator Correlation Test
        pws_indcorr_good_list.append(IndCorrGood.any())

        # Valid bins for normed weights
        ValidBins = np.isfinite(binned_indcorr_pws.values)
        RankSumWeights = rsw(len(IndCorrGood))
        NormedWeights = sum(ValidBins * np.array(RankSumWeights))

        score = sum(IndCorrGood.values * np.array(RankSumWeights)) / NormedWeights # noqa: PD011
        pws_indcorr_score_list.append(score)

    result = indicator_correlation_matrix.to_dataset(name="indcorr")
    result["dist"] = distance_matrix
    result["indcorr_good"] = ("id_neighbor", pws_indcorr_good_list)
    result["indcorr_score"] = ("id_neighbor", pws_indcorr_score_list)

    return result
